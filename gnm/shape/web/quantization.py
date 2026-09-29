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

"""Per-row symmetric int8 quantization and error validation for GNM bases."""

from __future__ import annotations

from typing import Final, NamedTuple

import numpy as np

_INT8_MAX: Final[int] = 127
_MIN_SCALE: Final[float] = 1e-12


class QuantizedMatrix(NamedTuple):
  """Quantized int8 matrix and per-row float32 scale factors."""

  quantized: np.ndarray
  scales: np.ndarray


class ReconstructionMetrics(NamedTuple):
  """Maximum and mean absolute reconstruction error metrics."""

  max_abs_error: float
  mean_abs_error: float


def quantize_int8(matrix: np.ndarray) -> QuantizedMatrix:
  """Quantizes a 2D floating-point matrix to symmetric per-row int8.

  Args:
    matrix: 2D floating-point NumPy array of shape (rows, cols) to quantize.

  Returns:
    QuantizedMatrix with int8 `quantized` and float32 `scales`.

  Raises:
    TypeError: If `matrix` does not have a floating-point dtype.
    ValueError: If `matrix` is not a non-empty 2D array or out of float32 range.
  """
  if not np.issubdtype(matrix.dtype, np.floating):
    raise TypeError(f'Expected floating-point matrix, got {matrix.dtype}.')
  if matrix.ndim != 2 or matrix.size == 0:
    raise ValueError(f'Expected non-empty 2D matrix, got {matrix.shape}.')
  max_abs = np.abs(matrix).max(axis=1)
  if not np.isfinite(max_abs.astype(np.float32)).all():
    raise ValueError('Matrix must contain finite values within float32 range.')
  scales = np.maximum(max_abs / _INT8_MAX, _MIN_SCALE).astype(np.float32)

  quantized = np.round(matrix / scales[:, np.newaxis]).astype(np.int8)
  return QuantizedMatrix(quantized=quantized, scales=scales)


def dequantize_int8(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
  """Reconstructs a float32 matrix from per-row int8 values and scales.

  Args:
    quantized: 2D int8 NumPy array of shape (rows, cols).
    scales: 1D floating-point NumPy array of shape (rows,).

  Returns:
    2D float32 NumPy array of shape (rows, cols).

  Raises:
    TypeError: If `quantized` is not int8 or `scales` is not floating-point.
    ValueError: If array dimensions or scale values are invalid.
  """
  if quantized.dtype != np.int8 or not np.issubdtype(scales.dtype, np.floating):
    raise TypeError(
        'Expected int8 quantized matrix and floating-point scales, got'
        f' {quantized.dtype} and {scales.dtype}.'
    )
  if quantized.ndim != 2 or quantized.size == 0:
    raise ValueError(f'Expected non-empty 2D matrix, got {quantized.shape}.')
  if scales.ndim != 1:
    raise ValueError(f'Expected 1D scales array, got {scales.shape}.')
  if quantized.shape[0] != scales.shape[0]:
    raise ValueError(
        f'Row mismatch: {quantized.shape[0]} vs {scales.shape[0]}.'
    )
  scales = scales.astype(np.float32)
  if not np.isfinite(scales).all() or np.any(scales <= 0):
    raise ValueError('Scales must be finite and strictly positive.')

  return quantized.astype(np.float32) * scales[:, np.newaxis]


def compute_reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> ReconstructionMetrics:
  """Computes maximum and mean absolute reconstruction error between arrays.

  Args:
    original: Ground-truth NumPy array.
    reconstructed: Reconstructed NumPy array of matching shape.

  Returns:
    ReconstructionMetrics containing (max_abs_error, mean_abs_error) as floats.

  Raises:
    ValueError: If shapes mismatch, arrays are empty, or values are non-finite.
  """
  if original.shape != reconstructed.shape:
    raise ValueError(
        f'Shape mismatch: {original.shape} vs {reconstructed.shape}.'
    )
  if original.size == 0:
    raise ValueError('Cannot compute error on empty arrays.')
  if not (np.isfinite(original).all() and np.isfinite(reconstructed).all()):
    raise ValueError('Arrays must contain only finite values.')

  abs_diff = np.abs(np.subtract(original, reconstructed, dtype=np.float64))
  return ReconstructionMetrics(float(abs_diff.max()), float(abs_diff.mean()))
