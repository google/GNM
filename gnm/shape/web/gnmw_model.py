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

"""Serializes GNM mesh topology and quantized bases into GNMW containers."""

from __future__ import annotations

import dataclasses
from typing import Any, Final

from gnm.shape.web import basis_codec
from gnm.shape.web import gnmw_container
import numpy as np

_COORDS_PER_VERTEX: Final[int] = 3
_VERTICES_PER_TRIANGLE: Final[int] = 3


@dataclasses.dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class GnmwModel:
  """Web-exportable GNM mesh topology and linear blend-shape bases.

  Attributes:
    vertices: 2D float32 template vertex positions of shape (V, 3).
    triangles: 2D uint32 triangle vertex indices of shape (T, 3) in [0, V).
    identity_basis: 3D float32 identity basis of shape (I, V, 3).
    expression_basis: 3D float32 expression basis of shape (E, V, 3).
  """

  vertices: np.ndarray
  triangles: np.ndarray
  identity_basis: np.ndarray
  expression_basis: np.ndarray

  def __post_init__(self) -> None:
    """Validates cross-array shape, dtype, and triangle index bounds."""
    if not np.issubdtype(self.vertices.dtype, np.floating):
      raise TypeError(
          f'Expected floating-point vertices, got {self.vertices.dtype}.'
      )
    if (
        self.vertices.ndim != 2
        or self.vertices.shape[1] != _COORDS_PER_VERTEX
        or self.vertices.shape[0] == 0
    ):
      raise ValueError(
          f'Expected non-empty vertices of shape (V, {_COORDS_PER_VERTEX}), got'
          f' {self.vertices.shape}.'
      )
    if not np.isfinite(self.vertices).all():
      raise ValueError('Vertices must contain finite values.')

    if not np.issubdtype(self.triangles.dtype, np.integer):
      raise TypeError(
          f'Expected integer triangles, got {self.triangles.dtype}.'
      )
    if (
        self.triangles.ndim != 2
        or self.triangles.shape[1] != _VERTICES_PER_TRIANGLE
        or self.triangles.shape[0] == 0
    ):
      raise ValueError(
          'Expected non-empty triangles of shape (T,'
          f' {_VERTICES_PER_TRIANGLE}), got {self.triangles.shape}.'
      )

    num_vertices = self.vertices.shape[0]
    for name, basis in (
        ('identity_basis', self.identity_basis),
        ('expression_basis', self.expression_basis),
    ):
      if not np.issubdtype(basis.dtype, np.floating):
        raise TypeError(f'Expected floating-point {name}, got {basis.dtype}.')
      if (
          basis.ndim != 3
          or basis.shape[1:] != (num_vertices, _COORDS_PER_VERTEX)
          or basis.shape[0] == 0
      ):
        raise ValueError(
            f'Expected non-empty {name} of shape (components, {num_vertices},'
            f' {_COORDS_PER_VERTEX}), got {basis.shape}.'
        )
      if not np.isfinite(basis).all():
        raise ValueError(f'{name} must contain finite values.')

    min_idx, max_idx = int(self.triangles.min()), int(self.triangles.max())
    if min_idx < 0 or max_idx >= num_vertices:
      raise ValueError(
          f'Triangle indices must be in [0, {num_vertices}), got'
          f' [{min_idx}, {max_idx}].'
      )


def serialize_model(model: GnmwModel) -> bytes:
  """Packs a GnmwModel into a GNMW binary container byte buffer."""
  packed_identity = basis_codec.pack_basis(model.identity_basis)
  packed_expression = basis_codec.pack_basis(model.expression_basis)

  writer = gnmw_container.ContainerWriter()
  writer.add('vertices', model.vertices.astype(np.float32, copy=False))
  writer.add('triangles', model.triangles.astype(np.uint32, copy=False))
  writer.add('identity_quantized', packed_identity.quantized)
  writer.add('identity_scales', packed_identity.scales)
  writer.add('expression_quantized', packed_expression.quantized)
  writer.add('expression_scales', packed_expression.scales)
  return writer.to_bytes(meta={'num_vertices': model.vertices.shape[0]})


def deserialize_model(
    data: bytes | bytearray | memoryview[Any],
) -> GnmwModel:
  """Reconstructs a GnmwModel from a GNMW binary container byte buffer."""
  reader = gnmw_container.ContainerReader(data)
  vertices = reader.get_section('vertices').astype(np.float32)
  num_vertices = vertices.shape[0]
  return GnmwModel(
      vertices=vertices,
      triangles=reader.get_section('triangles').astype(np.uint32),
      identity_basis=basis_codec.unpack_basis(
          reader.get_section('identity_quantized'),
          reader.get_section('identity_scales'),
          num_vertices=num_vertices,
      ),
      expression_basis=basis_codec.unpack_basis(
          reader.get_section('expression_quantized'),
          reader.get_section('expression_scales'),
          num_vertices=num_vertices,
      ),
  )
