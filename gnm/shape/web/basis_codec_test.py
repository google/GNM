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
"""Tests for GNM basis packing and unpacking."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape.web import basis_codec
import numpy as np


class BasisCodecTest(parameterized.TestCase):

  def test_pack_basis_flattens_components_to_int8_rows(self) -> None:
    basis = np.array(
        [
            [[127.0, -63.5, 0.0], [63.5, -127.0, 31.75]],
            [[12.7, -6.35, 0.0], [6.35, -12.7, 3.175]],
        ],
        dtype=np.float32,
    )
    expected_rows = np.array(
        [[127, -64, 0, 64, -127, 32], [127, -64, 0, 64, -127, 32]],
        dtype=np.int8,
    )

    packed = basis_codec.pack_basis(basis)

    self.assertEqual(packed.quantized.dtype, np.int8)
    self.assertEqual(packed.scales.dtype, np.float32)
    np.testing.assert_allclose(packed.scales, [1.0, 0.1])
    np.testing.assert_array_equal(packed.quantized, expected_rows)

  def test_pack_basis_reports_error_within_half_quantization_step(self) -> None:
    rng = np.random.default_rng(42)
    basis = rng.uniform(-2.0, 2.0, size=(4, 16, 3)).astype(np.float32)

    packed = basis_codec.pack_basis(basis)

    # Half step size is 2.0 / (2 * 127); 0.501 adds float32 slack.
    self.assertLessEqual(packed.metrics.max_abs_error, (2.0 / 127.0) * 0.501)
    self.assertGreaterEqual(packed.metrics.mean_abs_error, 0.0)
    self.assertLessEqual(
        packed.metrics.mean_abs_error, packed.metrics.max_abs_error
    )

  @parameterized.named_parameters(
      ('rank_2', (4, 3)),
      ('rank_4', (2, 4, 3, 1)),
      ('trailing_dim_two', (2, 4, 2)),
      ('trailing_dim_four', (2, 4, 4)),
      ('empty_vertices', (2, 0, 3)),
      ('empty_components', (0, 4, 3)),
  )
  def test_pack_basis_invalid_shape_raises_value_error(
      self, shape: tuple[int, ...]
  ) -> None:
    basis = np.zeros(shape, dtype=np.float32)

    with self.assertRaisesRegex(ValueError, r'components, vertices, 3'):
      basis_codec.pack_basis(basis)

  def test_pack_basis_non_floating_dtype_raises_type_error(self) -> None:
    basis = np.zeros((2, 4, 3), dtype=np.int32)

    with self.assertRaisesRegex(TypeError, r'Expected floating-point'):
      basis_codec.pack_basis(basis)

  def test_pack_basis_non_finite_values_raise_value_error(self) -> None:
    basis = np.ones((2, 4, 3), dtype=np.float32)
    basis[0, 0, 0] = float('nan')

    with self.assertRaisesRegex(ValueError, r'finite values within float32'):
      basis_codec.pack_basis(basis)

  def test_unpack_basis_dequantizes_and_reshapes_to_float32_basis(self) -> None:
    quantized = np.array(
        [[127, -64, 0, 32, -127, 64], [10, -20, 30, -40, 50, -60]],
        dtype=np.int8,
    )
    scales = np.array([0.5, 2.0], dtype=np.float32)
    expected = np.array(
        [
            [[63.5, -32.0, 0.0], [16.0, -63.5, 32.0]],
            [[20.0, -40.0, 60.0], [-80.0, 100.0, -120.0]],
        ],
        dtype=np.float32,
    )

    restored = basis_codec.unpack_basis(quantized, scales, num_vertices=2)

    self.assertEqual(restored.shape, (2, 2, 3))
    self.assertEqual(restored.dtype, np.float32)
    np.testing.assert_allclose(restored, expected)

  def test_unpack_basis_non_int8_quantized_dtype_raises_type_error(
      self,
  ) -> None:
    quantized = np.zeros((2, 6), dtype=np.float32)
    scales = np.ones(2, dtype=np.float32)

    with self.assertRaisesRegex(TypeError, r'Expected int8 quantized'):
      basis_codec.unpack_basis(quantized, scales, num_vertices=2)

  @parameterized.named_parameters(
      ('zero_vertices', 0, r'positive num_vertices'),
      ('negative_vertices', -1, r'positive num_vertices'),
      ('column_deficit', 5, r'Expected 15 columns'),
      ('column_excess', 1, r'Expected 3 columns for 1 vertices, got 6'),
  )
  def test_unpack_basis_invalid_num_vertices_raises_value_error(
      self, num_vertices: int, expected_regex: str
  ) -> None:
    quantized = np.zeros((2, 6), dtype=np.int8)
    scales = np.ones(2, dtype=np.float32)

    with self.assertRaisesRegex(ValueError, expected_regex):
      basis_codec.unpack_basis(quantized, scales, num_vertices)

  @parameterized.named_parameters(
      ('single_vertex_boundary', 1, 1, 1.0),
      ('all_zero_basis', 2, 4, 0.0),
      ('small_basis', 4, 8, 0.1),
      ('large_basis', 16, 64, 10.0),
  )
  def test_pack_unpack_roundtrip_restores_shape_and_approximates_basis(
      self, components: int, num_vertices: int, bound: float
  ) -> None:
    rng = np.random.default_rng(42)
    original = rng.uniform(
        -bound, bound, size=(components, num_vertices, 3)
    ).astype(np.float32)

    packed = basis_codec.pack_basis(original)
    restored = basis_codec.unpack_basis(
        packed.quantized, packed.scales, num_vertices
    )

    self.assertEqual(restored.shape, original.shape)
    # Half step size is bound / (2 * 127); 0.501 adds float32 slack.
    np.testing.assert_allclose(restored, original, atol=(bound / 127.0) * 0.501)


if __name__ == '__main__':
  absltest.main()
