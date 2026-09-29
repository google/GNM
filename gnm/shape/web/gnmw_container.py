# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GNMW (Generative aNthropometric Model Web) binary reader and writer.

Container layout (all multi-byte integers and floats in little-endian):
  - bytes 0..3   : Magic signature (b'GNMW')
  - bytes 4..7   : uint32 format version (1)
  - bytes 8..11  : uint32 JSON header byte length (H)
  - bytes 12..N-1: UTF-8 JSON header padded with spaces (0x20) (N = 12 + H)
  - bytes N..end : 4-byte aligned, zero-padded binary array sections
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import json
import math
import struct
import sys
import types
from typing import Any, Final, Self

from etils import epath
import numpy as np

MAGIC: Final[bytes] = b'GNMW'
FORMAT_VERSION: Final[int] = 1
ALIGNMENT_BYTES: Final[int] = 4
_PREAMBLE_STRUCT: Final[struct.Struct] = struct.Struct('<4sII')
PREAMBLE_SIZE_BYTES: Final[int] = 12
SUPPORTED_DTYPES: Final[frozenset[str]] = frozenset(
    {'float32', 'int8', 'uint8', 'uint16', 'int32', 'uint32'}
)


def compute_padding_bytes(length: int, alignment: int = ALIGNMENT_BYTES) -> int:
  """Computes padding bytes needed to align length to alignment."""
  return (-length) % alignment


@dataclasses.dataclass(frozen=True, slots=True)
class SectionHeader:
  """Metadata descriptor for a single binary array section."""

  name: str
  dtype: str
  shape: tuple[int, ...]
  offset: int
  byte_length: int

  def __post_init__(self) -> None:
    """Validates section header attribute constraints."""
    if not self.name:
      raise ValueError('Section name cannot be empty.')
    if self.dtype not in SUPPORTED_DTYPES:
      raise ValueError(f'Unsupported dtype "{self.dtype}".')
    if self.offset < 0 or self.byte_length < 0:
      raise ValueError('Section offset and byte length must be non-negative.')
    if self.offset % ALIGNMENT_BYTES != 0:
      raise ValueError(f'Section offset {self.offset} must be 4-byte aligned.')
    if any(dim < 0 for dim in self.shape):
      raise ValueError('Section shape dimensions must be non-negative.')

  def to_json_dict(self) -> dict[str, Any]:
    """Converts the section header to a JSON-compatible dictionary."""
    return {
        'name': self.name,
        'dtype': self.dtype,
        'shape': list(self.shape),
        'offset': self.offset,
        'byteLength': self.byte_length,
    }

  @classmethod
  def from_json_dict(cls, data: Mapping[str, Any]) -> Self:
    """Constructs a SectionHeader instance from a JSON dictionary."""
    return cls(
        name=str(data['name']),
        dtype=str(data['dtype']),
        shape=tuple(int(dim) for dim in data['shape']),
        offset=int(data['offset']),
        byte_length=int(data['byteLength']),
    )


class ContainerWriter:
  """Builder for packing named NumPy arrays into a GNMW container file.

  Note: Instances of ContainerWriter are mutable and not thread-safe.
  """

  def __init__(self) -> None:
    """Initializes an empty ContainerWriter instance."""
    self._sections: list[SectionHeader] = []
    self._blobs: list[bytes] = []
    self._section_names: set[str] = set()
    self._current_offset: int = 0

  def add(self, name: str, array: np.ndarray) -> None:
    """Adds a named array section to the container."""
    if not name:
      raise ValueError('Section name cannot be empty.')
    if name in self._section_names:
      raise ValueError(f'Section "{name}" has already been added.')

    arr = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    if arr.dtype.byteorder == '>' or (
        arr.dtype.byteorder == '=' and sys.byteorder == 'big'
    ):
      arr = arr.astype(arr.dtype.newbyteorder('<'))

    dtype_name = arr.dtype.name
    if dtype_name not in SUPPORTED_DTYPES:
      raise TypeError(
          f'Array dtype "{dtype_name}" for "{name}" is unsupported.'
      )

    raw_bytes = arr.tobytes()
    byte_len = len(raw_bytes)
    pad_len = compute_padding_bytes(byte_len)

    self._sections.append(
        SectionHeader(
            name=name,
            dtype=dtype_name,
            shape=array.shape,
            offset=self._current_offset,
            byte_length=byte_len,
        )
    )
    self._blobs.append(raw_bytes + b'\x00' * pad_len)
    self._section_names.add(name)
    self._current_offset += byte_len + pad_len

  def to_bytes(self, meta: Mapping[str, Any] | None = None) -> bytes:
    """Serializes the container into an in-memory byte string."""
    hdr = {
        'meta': dict(meta) if meta is not None else {},
        'sections': [sec.to_json_dict() for sec in self._sections],
    }
    hdr_json = json.dumps(hdr, separators=(',', ':'), sort_keys=True).encode(
        'utf-8'
    )
    padded_hdr = hdr_json + b' ' * compute_padding_bytes(len(hdr_json))
    preamble = _PREAMBLE_STRUCT.pack(MAGIC, FORMAT_VERSION, len(padded_hdr))
    return preamble + padded_hdr + b''.join(self._blobs)

  def write(
      self,
      path: epath.PathLike,
      *,
      meta: Mapping[str, Any] | None = None,
  ) -> int:
    """Writes the container to the specified file path."""
    data = self.to_bytes(meta=meta)
    target = epath.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)

  def __repr__(self) -> str:
    """Returns a developer-readable string representation."""
    return f'ContainerWriter(sections={[sec.name for sec in self._sections]})'


class ContainerReader(Mapping[str, np.ndarray]):
  """Reader and read-only mapping for parsing GNMW binary containers."""

  def __init__(self, data: bytes | bytearray | memoryview[Any]) -> None:
    """Initializes the reader from a binary data buffer."""
    self._data = bytes(data)
    if len(self._data) < PREAMBLE_SIZE_BYTES:
      raise ValueError(f'Container too short: {len(self._data)} bytes.')

    magic, version, header_len = _PREAMBLE_STRUCT.unpack_from(self._data, 0)
    if magic != MAGIC:
      raise ValueError(f'Invalid magic: expected {MAGIC!r}, got {magic!r}.')
    if version != FORMAT_VERSION:
      raise ValueError(f'Unsupported version: expected 1, got {version}.')
    if header_len % ALIGNMENT_BYTES != 0:
      raise ValueError(f'Header length {header_len} is not 4-byte aligned.')

    header_end = PREAMBLE_SIZE_BYTES + header_len
    if len(self._data) < header_end:
      raise ValueError(f'Truncated container header: {len(self._data)} bytes.')

    try:
      raw = self._data[PREAMBLE_SIZE_BYTES:header_end].decode('utf-8')
      hdr = json.loads(raw)
      if not isinstance(hdr, dict):
        raise ValueError('JSON header root must be an object.')
      sections_raw = hdr.get('sections', ())
      if not isinstance(sections_raw, (list, tuple)):
        raise ValueError('Sections descriptor must be a list.')
      sections = tuple(SectionHeader.from_json_dict(s) for s in sections_raw)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
      raise ValueError(f'Malformed container JSON header: {e}') from e

    by_name: dict[str, SectionHeader] = {}
    for sec in sections:
      if sec.name in by_name:
        raise ValueError(f'Duplicate section name "{sec.name}".')
      by_name[sec.name] = sec

    self._metadata = types.MappingProxyType(dict(hdr.get('meta', {})))
    self._sections = types.MappingProxyType(by_name)
    self._section_names = tuple(sec.name for sec in sections)
    self._body_offset = header_end

  @classmethod
  def from_file(cls, path: epath.PathLike) -> Self:
    """Constructs a ContainerReader from a file path."""
    return cls(epath.Path(path).read_bytes())

  @property
  def metadata(self) -> Mapping[str, Any]:
    """Immutable view of the container metadata dictionary."""
    return self._metadata

  @property
  def section_names(self) -> tuple[str, ...]:
    """Tuple of all array section names in insertion order."""
    return self._section_names

  @property
  def sections(self) -> Mapping[str, SectionHeader]:
    """Immutable mapping of section names to SectionHeader descriptors."""
    return self._sections

  def has_section(self, name: str) -> bool:
    """Returns whether a section with the given name exists."""
    return name in self._sections

  def get_section(self, name: str) -> np.ndarray:
    """Reconstructs and returns the array for the named section."""
    sec = self._sections.get(name)
    if sec is None:
      raise KeyError(f'Section "{name}" not found.')
    if sec.dtype not in SUPPORTED_DTYPES:
      raise ValueError(f'Unknown dtype "{sec.dtype}".')

    np_dtype = np.dtype(sec.dtype).newbyteorder('<')
    expected_bytes = math.prod(sec.shape) * np_dtype.itemsize
    if sec.byte_length != expected_bytes:
      raise ValueError(
          f'Section "{name}" byte length ({sec.byte_length}) mismatch'
          f' (expected {expected_bytes}).'
      )

    start = self._body_offset + sec.offset
    end = start + sec.byte_length
    if len(self._data) < end:
      raise ValueError(f'Truncated payload for section "{name}".')

    count = sec.byte_length // np_dtype.itemsize
    return np.frombuffer(
        self._data, dtype=np_dtype, count=count, offset=start
    ).reshape(sec.shape)

  def __getitem__(self, name: str) -> np.ndarray:
    """Returns the array for the named section (alias for 'get_section')."""
    return self.get_section(name)

  def __contains__(self, name: object) -> bool:
    """Returns whether a section exists in the container."""
    return isinstance(name, str) and name in self._sections

  def __len__(self) -> int:
    """Returns the total number of array sections in the container."""
    return len(self._section_names)

  def __iter__(self) -> Iterator[str]:
    """Returns an iterator over section names in insertion order."""
    return iter(self._section_names)
