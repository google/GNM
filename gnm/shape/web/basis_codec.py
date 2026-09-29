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

"""Pack and unpack 3D GNM bases as per-component quantized int8 matrices."""

from __future__ import annotations

from typing import Final, NamedTuple

from gnm.shape.web import quantization
import numpy as np

_COORDS_PER_VERTEX: Final[int] = 3


class PackedBasis(NamedTuple):
  """Quantized representation of a 3D GNM basis.

  Attributes:
    quantized: 2D int8 array of shape (components, vertices * 3).
    scales: 1D float32 array of per-component scale factors.
    metrics: Reconstruction error metrics of the packed rows against the basis.
  """

  quantized: np.ndarray
  scales: np.ndarray
  metrics: quantization.ReconstructionMetrics


def pack_basis(basis: np.ndarray) -> PackedBasis:
  """Quantizes a 3D GNM basis into one int8 row per component.

  Dtype and finiteness constraints are enforced by `quantization.quantize_int8`
  on the flattened rows.

  Args:
    basis: 3D floating-point array of shape (components, vertices, 3).

  Returns:
    PackedBasis holding the int8 rows, the per-component scales, and the
    reconstruction error of those rows against `basis`.

  Raises:
    TypeError: If `basis` does not have a floating-point dtype.
    ValueError: If `basis` is not a non-empty (components, vertices, 3) array,
      or contains non-finite values.
  """
  if basis.ndim != 3 or basis.shape[2] != _COORDS_PER_VERTEX or basis.size == 0:
    raise ValueError(
        'Expected non-empty basis of shape (components, vertices,'
        f' {_COORDS_PER_VERTEX}), got {basis.shape}.'
    )

  flattened = basis.reshape((basis.shape[0], -1))
  quantized, scales = quantization.quantize_int8(flattened)
  restored = quantization.dequantize_int8(quantized, scales)
  metrics = quantization.compute_reconstruction_error(flattened, restored)
  return PackedBasis(quantized=quantized, scales=scales, metrics=metrics)


def unpack_basis(
    quantized: np.ndarray,
    scales: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
  """Restores a 3D GNM basis from packed int8 rows and per-component scales.

  Dtype, dimension, and scale value constraints are enforced by
  `quantization.dequantize_int8`.

  Args:
    quantized: 2D int8 array of shape (components, num_vertices * 3).
    scales: 1D floating-point array of shape (components,).
    num_vertices: Number of vertices per component in the original basis.

  Returns:
    3D float32 array of shape (components, num_vertices, 3).

  Raises:
    TypeError: If `quantized` is not int8 or `scales` is not floating-point.
    ValueError: If `num_vertices` is not positive, if the column count does
      not match `num_vertices`, or if the packed arrays are otherwise invalid.
  """
  if num_vertices <= 0:
    raise ValueError(f'Expected positive num_vertices, got {num_vertices}.')

  restored = quantization.dequantize_int8(quantized, scales)
  expected_columns = num_vertices * _COORDS_PER_VERTEX
  if restored.shape[1] != expected_columns:
    raise ValueError(
        f'Expected {expected_columns} columns for {num_vertices} vertices, got'
        f' {restored.shape[1]}.'
    )

  return restored.reshape((restored.shape[0], num_vertices, _COORDS_PER_VERTEX))
