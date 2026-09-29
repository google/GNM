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
"""Tests for GNM basis quantization."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape.web import quantization
import numpy as np

_Shape = tuple[int, ...]


class QuantizationTest(parameterized.TestCase):
  """Tests for int8 quantization, dequantization, and error calculation."""

  def test_quantize_int8_maps_row_maxima_to_int8_bounds(self) -> None:
    matrix = np.array(
        [[127.0, -63.5], [0.0, 0.0], [12.7, -12.7]], dtype=np.float32
    )

    result = quantization.quantize_int8(matrix)

    self.assertEqual(result.quantized.dtype, np.int8)
    self.assertEqual(result.scales.dtype, np.float32)
    np.testing.assert_allclose(result.scales, [1.0, 1e-12, 0.1])
    expected = np.array([[127, -64], [0, 0], [127, -127]], dtype=np.int8)
    np.testing.assert_array_equal(result.quantized, expected)

  def test_quantize_int8_non_floating_dtype_raises_type_error(self) -> None:
    matrix = np.zeros((2, 2), dtype=np.int32)

    with self.assertRaisesRegex(TypeError, r'Expected floating-point'):
      quantization.quantize_int8(matrix)

  @parameterized.named_parameters(
      ('non_2d', (4,), 0.0, r'non-empty 2D matrix'),
      ('empty', (0, 5), 0.0, r'non-empty 2D matrix'),
      ('non_finite', (2, 2), float('nan'), r'finite values within float32'),
      ('float64_overflow', (2, 2), 1e39, r'finite values within float32'),
  )
  def test_quantize_int8_invalid_input_raises_value_error(
      self, shape: _Shape, fill_value: float, regex: str
  ) -> None:
    invalid_matrix = np.full(shape, fill_value, dtype=np.float64)

    with self.assertRaisesRegex(ValueError, regex):
      quantization.quantize_int8(invalid_matrix)

  def test_dequantize_int8_scales_rows_to_float32(self) -> None:
    quantized = np.array([[100, -50], [10, -20]], dtype=np.int8)
    scales = np.array([0.5, 2.0], dtype=np.float64)

    reconstructed = quantization.dequantize_int8(quantized, scales)

    self.assertEqual(reconstructed.dtype, np.float32)
    expected = np.array([[50.0, -25.0], [20.0, -40.0]], dtype=np.float32)
    np.testing.assert_allclose(reconstructed, expected)

  @parameterized.named_parameters(
      ('non_int8_quantized', np.float32, np.float32),
      ('non_floating_scales', np.int8, np.int32),
  )
  def test_dequantize_int8_invalid_dtype_raises_type_error(
      self, q_dtype: type[np.number], s_dtype: type[np.number]
  ) -> None:
    quantized = np.zeros((2, 2), dtype=q_dtype)
    scales = np.ones(2, dtype=s_dtype)

    with self.assertRaisesRegex(TypeError, r'Expected int8 quantized'):
      quantization.dequantize_int8(quantized, scales)

  @parameterized.named_parameters(
      ('non_2d', (4,), (4,), 1.0, r'non-empty 2D matrix'),
      ('empty', (0, 3), (0,), 1.0, r'non-empty 2D matrix'),
      ('non_1d_scales', (2, 3), (2, 1), 1.0, r'Expected 1D scales'),
      ('row_mismatch', (2, 3), (3,), 1.0, r'Row mismatch'),
      ('non_positive_scale', (2, 3), (2,), 0.0, r'strictly positive'),
      ('nan_scale', (2, 3), (2,), float('nan'), r'finite and strictly'),
  )
  def test_dequantize_int8_invalid_inputs_raise_value_error(
      self, q_shape: _Shape, s_shape: _Shape, scale: float, regex: str
  ) -> None:
    quantized = np.zeros(q_shape, dtype=np.int8)
    scales = np.full(s_shape, scale, dtype=np.float32)

    with self.assertRaisesRegex(ValueError, regex):
      quantization.dequantize_int8(quantized, scales)

  def test_compute_reconstruction_error_returns_max_and_mean_diff(self) -> None:
    original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reconstructed = np.array([[1.5, 2.0], [2.0, 4.5]], dtype=np.float32)

    err = quantization.compute_reconstruction_error(original, reconstructed)

    self.assertAlmostEqual(err.max_abs_error, 1.0)
    self.assertAlmostEqual(err.mean_abs_error, 0.5)

  @parameterized.named_parameters(
      ('shape_mismatch', (2, 3), (3, 2), r'Shape mismatch'),
      ('empty_array', (0, 3), (0, 3), r'empty arrays'),
  )
  def test_compute_reconstruction_error_invalid_shapes_raise_value_error(
      self, o_shape: _Shape, r_shape: _Shape, regex: str
  ) -> None:
    original = np.zeros(o_shape)
    reconstructed = np.zeros(r_shape)

    with self.assertRaisesRegex(ValueError, regex):
      quantization.compute_reconstruction_error(original, reconstructed)

  @parameterized.named_parameters(
      ('nan_original', float('nan'), 0.0),
      ('inf_reconstructed', 0.0, float('inf')),
  )
  def test_compute_reconstruction_error_non_finite_values_raise_value_error(
      self, orig_val: float, recon_val: float
  ) -> None:
    original = np.full((2, 2), orig_val)
    reconstructed = np.full((2, 2), recon_val)

    with self.assertRaisesRegex(ValueError, r'only finite'):
      quantization.compute_reconstruction_error(original, reconstructed)

  @parameterized.named_parameters(
      ('small_basis', (8, 32), 0.1),
      ('large_basis', (50, 300), 10.0),
  )
  def test_roundtrip_bounds_max_error_within_half_step(
      self, shape: tuple[int, int], bound: float
  ) -> None:
    rng = np.random.default_rng(42)
    original = rng.uniform(-bound, bound, size=shape).astype(np.float32)

    quantized, scales = quantization.quantize_int8(original)
    reconstructed = quantization.dequantize_int8(quantized, scales)
    err = quantization.compute_reconstruction_error(original, reconstructed)

    # Half step size is bound / (2 * 127); 0.501 adds float32 slack.
    self.assertLessEqual(err.max_abs_error, (bound / 127.0) * 0.501)


if __name__ == '__main__':
  absltest.main()
