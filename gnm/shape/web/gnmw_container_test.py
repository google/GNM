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
"""Tests for GNMW container reader and writer."""
# pylint: disable=protected-access

from collections.abc import Callable, Mapping
import json
import pathlib
import struct
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from gnm.shape.web import gnmw_container
import numpy as np


def _create_container_bytes(
    sections: Mapping[str, np.ndarray],
    meta: Mapping[str, Any] | None = None,
) -> bytes:
  """Helper to construct a GNMW binary container buffer for unit tests."""
  sec_headers = []
  blobs = []
  offset = 0
  for name, arr in sections.items():
    arr_c = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder('<'))
    raw = arr_c.tobytes()
    pad = gnmw_container.compute_padding_bytes(len(raw))
    sec_headers.append(
        gnmw_container.SectionHeader(
            name=name,
            dtype=arr_c.dtype.name,
            shape=arr.shape,
            offset=offset,
            byte_length=len(raw),
        ).to_json_dict()
    )
    blobs.append(raw + (b'\x00' * pad))
    offset += len(raw) + pad

  hdr_dict = {'meta': dict(meta or {}), 'sections': sec_headers}
  hdr_json = json.dumps(hdr_dict, separators=(',', ':'), sort_keys=True).encode(
      'utf-8'
  )
  pad_hdr = hdr_json + (
      b' ' * gnmw_container.compute_padding_bytes(len(hdr_json))
  )
  preamble = gnmw_container._PREAMBLE_STRUCT.pack(
      gnmw_container.MAGIC, gnmw_container.FORMAT_VERSION, len(pad_hdr)
  )
  return preamble + pad_hdr + b''.join(blobs)


class SectionHeaderAndAlignmentTest(parameterized.TestCase):
  """Tests for alignment calculations and SectionHeader metadata."""

  @parameterized.named_parameters(
      ('zero', 0, 4, 0),
      ('unaligned_1', 1, 4, 3),
      ('unaligned_2', 2, 4, 2),
      ('unaligned_3', 3, 4, 1),
      ('aligned_4', 4, 4, 0),
      ('custom_align', 3, 8, 5),
  )
  def test_compute_padding_bytes(
      self, length: int, alignment: int, expected: int
  ) -> None:
    self.assertEqual(
        gnmw_container.compute_padding_bytes(length, alignment), expected
    )

  def test_section_header_to_and_from_json_dict_roundtrip(self) -> None:
    header = gnmw_container.SectionHeader('v', 'float32', (10, 3), 64, 120)
    data = header.to_json_dict()
    self.assertEqual(
        data,
        {
            'name': 'v',
            'dtype': 'float32',
            'shape': [10, 3],
            'offset': 64,
            'byteLength': 120,
        },
    )
    self.assertEqual(gnmw_container.SectionHeader.from_json_dict(data), header)

    scalar = gnmw_container.SectionHeader('s', 'int32', (), 0, 4)
    self.assertEqual(
        gnmw_container.SectionHeader.from_json_dict(scalar.to_json_dict()),
        scalar,
    )

  @parameterized.named_parameters(
      ('empty_name', '', 'float32', 0, 4, (1,), r'cannot be empty'),
      ('bad_dtype', 'sec', 'float64', 0, 8, (1,), r'Unsupported dtype'),
      ('negative_offset', 'sec', 'float32', -4, 4, (1,), r'non-negative'),
      ('unaligned', 'sec', 'float32', 2, 4, (1,), r'must be 4-byte aligned'),
      ('negative_len', 'sec', 'float32', 0, -4, (1,), r'non-negative'),
      ('negative_dim', 'sec', 'float32', 0, 4, (-1, 3), r'non-negative'),
  )
  def test_section_header_invalid_attributes_raise_value_error(
      self,
      name: str,
      dtype: str,
      offset: int,
      length: int,
      shape: tuple[int, ...],
      regex: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, regex):
      gnmw_container.SectionHeader(name, dtype, shape, offset, length)


class ContainerReaderTest(parameterized.TestCase):
  """Tests for container parsing and mapping protocols."""

  @parameterized.named_parameters(
      ('bytes', bytes),
      ('bytearray', bytearray),
      ('memoryview', memoryview),
  )
  def test_reader_supported_buffer_types(
      self, buffer_fn: Callable[[bytes], Any]
  ) -> None:
    expected = np.array([10, 20], dtype=np.int32)
    raw = _create_container_bytes({'data': expected})
    reader = gnmw_container.ContainerReader(buffer_fn(raw))
    np.testing.assert_array_equal(reader['data'], expected)

  @parameterized.named_parameters(
      ('str_path', str),
      ('epath_path', epath.Path),
      ('pathlib_path', pathlib.Path),
  )
  def test_reader_from_file_supported_path_types(
      self, path_fn: Callable[[str], epath.PathLike]
  ) -> None:
    expected = np.array([10, 20], dtype=np.int32)
    raw = _create_container_bytes({'data': expected})
    path = epath.Path(self.create_tempdir().full_path) / 'test.gnmw'
    path.write_bytes(raw)
    reader = gnmw_container.ContainerReader.from_file(path_fn(str(path)))
    np.testing.assert_array_equal(reader['data'], expected)

  @parameterized.named_parameters(
      ('empty', b'', r'Container too short'),
      (
          'bad_magic',
          b'BADM\x01\x00\x00\x00\x04\x00\x00\x00    ',
          r'Invalid magic',
      ),
      (
          'bad_version',
          struct.pack('<4sII4s', b'GNMW', 2, 4, b' ' * 4),
          r'Unsupported version',
      ),
      (
          'unaligned_len',
          struct.pack('<4sII7s', b'GNMW', 1, 7, b' ' * 7),
          r'not 4-byte aligned',
      ),
      (
          'truncated',
          struct.pack('<4sII', b'GNMW', 1, 100),
          r'Truncated container header',
      ),
      (
          'malformed_json',
          struct.pack('<4sII', b'GNMW', 1, 4) + b'{bad',
          r'Malformed container',
      ),
      (
          'root_list',
          struct.pack('<4sII', b'GNMW', 1, 4) + b'[]  ',
          r'root must be an object',
      ),
  )
  def test_reader_corrupt_data_raises_value_error(
      self, corrupt_bytes: bytes, regex: str
  ) -> None:
    with self.assertRaisesRegex(ValueError, regex):
      gnmw_container.ContainerReader(corrupt_bytes)

  def test_reader_mapping_protocols_and_immutability(self) -> None:
    raw = _create_container_bytes(
        {
            'a': np.array([1, 2], dtype=np.uint8),
            'b': np.array([3, 4], dtype=np.uint8),
        },
        meta={'k': 'v'},
    )
    reader = gnmw_container.ContainerReader(raw)

    self.assertLen(reader, 2)
    self.assertEqual(tuple(reader), ('a', 'b'))
    self.assertEqual(list(reader.keys()), ['a', 'b'])
    self.assertIn('a', reader)
    self.assertNotIn('c', reader)
    self.assertNotIn(123, reader)
    np.testing.assert_array_equal(
        reader.get('a'), np.array([1, 2], dtype=np.uint8)
    )
    self.assertIsNone(reader.get('c', None))

    arr = reader['a']
    self.assertFalse(arr.flags.writeable)
    with self.assertRaisesRegex(ValueError, r'destination is read-only'):
      arr[0] = 99

    reader_meta: Any = reader.metadata
    with self.assertRaisesRegex(TypeError, r"'mappingproxy'"):
      reader_meta['k'] = 'mutated'


class ContainerEndToEndReaderTest(parameterized.TestCase):
  """End-to-end multi-dtype and layout decoding tests."""

  @parameterized.named_parameters(
      ('float32', np.array([[1.5, -2.0], [3.0, 4.25]], dtype=np.float32)),
      ('int8', np.array([-128, 0, 127], dtype=np.int8)),
      ('uint8', np.array([0, 128, 255], dtype=np.uint8)),
      ('uint16', np.array([[0, 1000], [30000, 65535]], dtype=np.uint16)),
      ('int32', np.array([-100000, 0, 100000], dtype=np.int32)),
      ('uint32', np.array([0, 500000, 4000000], dtype=np.uint32)),
      ('scalar', np.array(42, dtype=np.int32)),
      ('zero_1d', np.zeros((0,), dtype=np.float32)),
      ('zero_3d', np.zeros((5, 0, 3), dtype=np.float32)),
      ('volume_3d', np.arange(24, dtype=np.float32).reshape(2, 3, 4)),
      ('tensor_4d', np.arange(16, dtype=np.uint16).reshape(2, 2, 2, 2)),
  )
  def test_decode_preserves_array_data(self, arr: np.ndarray) -> None:
    raw = _create_container_bytes({'sample': arr})
    reader = gnmw_container.ContainerReader(raw)
    recovered = reader['sample']

    self.assertEqual(recovered.dtype, arr.dtype.newbyteorder('<'))
    self.assertEqual(recovered.shape, arr.shape)
    self.assertTrue(recovered.flags.c_contiguous)
    np.testing.assert_array_equal(recovered, arr)


class ContainerWriterTest(parameterized.TestCase):
  """Tests for ContainerWriter validation, serialization, and file I/O."""

  def test_add_empty_name_raises_value_error(self) -> None:
    writer = gnmw_container.ContainerWriter()
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Section name cannot be empty.'
    ):
      writer.add('', np.array([1.0], dtype=np.float32))

  def test_add_duplicate_name_raises_value_error(self) -> None:
    writer = gnmw_container.ContainerWriter()
    writer.add('mesh', np.array([1.0, 2.0], dtype=np.float32))
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Section "mesh" has already been added.'
    ):
      writer.add('mesh', np.array([2.0], dtype=np.float32))

  def test_add_unsupported_dtype_raises_type_error(self) -> None:
    writer = gnmw_container.ContainerWriter()
    with self.assertRaisesRegex(TypeError, r'is unsupported'):
      writer.add('bad_dtype', np.zeros(5, dtype=np.float64))

  def test_to_bytes_invalid_metadata_raises_type_error(self) -> None:
    writer = gnmw_container.ContainerWriter()
    with self.assertRaisesRegex(
        TypeError, r'is not JSON serializable|Object of type'
    ):
      writer.to_bytes(meta={'invalid': object()})

  def test_to_bytes_none_metadata_produces_empty_metadata(self) -> None:
    writer = gnmw_container.ContainerWriter()
    reader = gnmw_container.ContainerReader(writer.to_bytes(meta=None))
    self.assertEmpty(reader.metadata)

  def test_add_big_endian_converts_to_little_endian(self) -> None:
    writer = gnmw_container.ContainerWriter()
    be_data = np.array([1.5, 2.5, 3.5], dtype='>f4')
    writer.add('be', be_data)
    reader = gnmw_container.ContainerReader(writer.to_bytes())

    self.assertIn(reader['be'].dtype.byteorder, ('<', '='))
    np.testing.assert_array_equal(reader['be'], be_data)

  def test_write_creates_file_and_preserves_content(self) -> None:
    writer = gnmw_container.ContainerWriter()
    writer.add('mesh', np.array([1.0, 2.0], dtype=np.float32))
    file_path = (
        epath.Path(self.create_tempdir().full_path) / 'sub' / 'test.gnmw'
    )
    bytes_written = writer.write(file_path, meta={'key': 'val'})
    reader = gnmw_container.ContainerReader.from_file(file_path)

    self.assertLen(file_path.read_bytes(), bytes_written)
    self.assertEqual(reader.metadata, {'key': 'val'})
    np.testing.assert_array_equal(
        reader['mesh'], np.array([1.0, 2.0], dtype=np.float32)
    )

  def test_repr_lists_section_names(self) -> None:
    writer = gnmw_container.ContainerWriter()
    writer.add('mesh', np.array([1.0, 2.0], dtype=np.float32))
    self.assertEqual(repr(writer), "ContainerWriter(sections=['mesh'])")

  @parameterized.named_parameters(
      ('float32', np.array([[1.5, -2.0], [3.0, 4.25]], dtype=np.float32)),
      ('int8', np.array([-128, 0, 127], dtype=np.int8)),
      ('uint8', np.array([0, 128, 255], dtype=np.uint8)),
      ('uint16', np.array([[0, 1000], [30000, 65535]], dtype=np.uint16)),
      ('int32', np.array([-100000, 0, 100000], dtype=np.int32)),
      ('uint32', np.array([0, 500000, 4000000], dtype=np.uint32)),
      (
          'fortran',
          np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3)),
      ),
      ('strided', np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2]),
  )
  def test_roundtrip_preserves_array_data(self, arr: np.ndarray) -> None:
    writer = gnmw_container.ContainerWriter()
    writer.add('sample', arr)
    reader = gnmw_container.ContainerReader(writer.to_bytes())
    recovered = reader['sample']

    self.assertEqual(recovered.dtype, arr.dtype.newbyteorder('<'))
    self.assertEqual(recovered.shape, arr.shape)
    np.testing.assert_array_equal(recovered, arr)


if __name__ == '__main__':
  absltest.main()
