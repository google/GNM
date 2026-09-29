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
"""Tests for GNM web-exportable model and serialization."""

from __future__ import annotations

import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape.web import gnmw_container
from gnm.shape.web import gnmw_model
import numpy as np

_EXPECTED_SECTIONS: tuple[str, ...] = (
    'vertices',
    'triangles',
    'identity_quantized',
    'identity_scales',
    'expression_quantized',
    'expression_scales',
)


def _make_valid_model(**kwargs: Any) -> gnmw_model.GnmwModel:
  defaults = {
      'vertices': np.ones((2, 3), dtype=np.float32),
      'triangles': np.zeros((1, 3), dtype=np.uint32),
      'identity_basis': np.zeros((2, 2, 3), dtype=np.float32),
      'expression_basis': np.zeros((3, 2, 3), dtype=np.float32),
  }
  defaults.update(kwargs)
  return gnmw_model.GnmwModel(**defaults)


def _ramp(shape: tuple[int, ...], bound: float) -> np.ndarray:
  return np.linspace(-bound, bound, math.prod(shape), dtype=np.float32).reshape(
      shape
  )


class GnmwModelTest(parameterized.TestCase):

  def test_serialize_model_writes_num_vertices_metadata_and_exact_sections(
      self,
  ) -> None:
    model = _make_valid_model()

    payload = gnmw_model.serialize_model(model)
    reader = gnmw_container.ContainerReader(payload)

    self.assertEqual(reader.metadata, {'num_vertices': 2})
    self.assertSequenceEqual(reader.section_names, _EXPECTED_SECTIONS)
    np.testing.assert_array_equal(
        reader.get_section('vertices'), model.vertices
    )
    self.assertEqual(reader.get_section('vertices').dtype, np.float32)
    np.testing.assert_array_equal(
        reader.get_section('triangles'), model.triangles
    )
    self.assertEqual(reader.get_section('triangles').dtype, np.uint32)

  @parameterized.named_parameters(
      ('single_vertex_zero_bases', 1, 1, 1, 1, 0.0, 0.0),
      ('multi_component_mesh', 6, 4, 3, 5, 2.0, 0.01),
  )
  def test_roundtrip_restores_mesh_and_approximates_bases(
      self,
      num_vertices: int,
      num_triangles: int,
      num_identity_components: int,
      num_expression_components: int,
      bound: float,
      atol: float,
  ) -> None:
    original = gnmw_model.GnmwModel(
        vertices=_ramp((num_vertices, 3), bound=1.0),
        triangles=np.zeros((num_triangles, 3), dtype=np.uint32),
        identity_basis=_ramp(
            (num_identity_components, num_vertices, 3), bound=bound
        ),
        expression_basis=_ramp(
            (num_expression_components, num_vertices, 3), bound=bound
        ),
    )

    payload = gnmw_model.serialize_model(original)
    restored = gnmw_model.deserialize_model(payload)

    with self.subTest('vertices'):
      np.testing.assert_array_equal(restored.vertices, original.vertices)
    with self.subTest('triangles'):
      np.testing.assert_array_equal(restored.triangles, original.triangles)
    with self.subTest('identity_basis'):
      np.testing.assert_allclose(
          restored.identity_basis, original.identity_basis, atol=atol
      )
    with self.subTest('expression_basis'):
      np.testing.assert_allclose(
          restored.expression_basis, original.expression_basis, atol=atol
      )

  @parameterized.named_parameters(
      ('int_vertices', 'vertices', np.zeros((2, 3), dtype=np.int32)),
      ('float_triangles', 'triangles', np.zeros((1, 3), dtype=np.float32)),
      ('int_identity', 'identity_basis', np.zeros((1, 2, 3), dtype=np.int32)),
  )
  def test_gnmw_model_invalid_dtype_raises_type_error(
      self,
      field_name: str,
      invalid_array: np.ndarray,
  ) -> None:
    with self.assertRaisesRegex(TypeError, f'Expected .*{field_name}'):
      _make_valid_model(**{field_name: invalid_array})

  @parameterized.named_parameters(
      ('rank_1_vertices', 'vertices', (3,)),
      ('wrong_cols_vertices', 'vertices', (2, 2)),
      ('empty_vertices', 'vertices', (0, 3)),
      ('zero_component_identity', 'identity_basis', (0, 2, 3)),
      ('identity_mismatch', 'identity_basis', (1, 3, 3)),
      ('expression_mismatch', 'expression_basis', (1, 3, 3)),
  )
  def test_gnmw_model_invalid_shape_raises_value_error(
      self,
      field_name: str,
      invalid_shape: tuple[int, ...],
  ) -> None:
    invalid_array = np.zeros(invalid_shape, dtype=np.float32)

    with self.assertRaisesRegex(ValueError, field_name):
      _make_valid_model(**{field_name: invalid_array})

  @parameterized.named_parameters(
      (
          'nan_vertices',
          'vertices',
          np.array([[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0]]),
      ),
      (
          'inf_expression_basis',
          'expression_basis',
          np.array([[[np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
      ),
  )
  def test_gnmw_model_non_finite_values_raises_value_error(
      self,
      field_name: str,
      array: np.ndarray,
  ) -> None:
    with self.assertRaisesRegex(ValueError, 'finite'):
      _make_valid_model(**{field_name: array.astype(np.float32)})

  @parameterized.named_parameters(
      ('wrong_columns', np.zeros((1, 2), dtype=np.int32), r'shape \(T, 3\)'),
      ('empty', np.zeros((0, 3), dtype=np.int32), r'shape \(T, 3\)'),
      ('negative_index', np.array([[-1, 0, 1]], dtype=np.int32), r'\[0, 2\)'),
      ('upper_bound_index', np.array([[0, 1, 2]], dtype=np.int32), r'\[0, 2\)'),
  )
  def test_gnmw_model_invalid_triangles_raises_value_error(
      self,
      triangles: np.ndarray,
      expected_regex: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, expected_regex):
      _make_valid_model(triangles=triangles)


if __name__ == '__main__':
  absltest.main()
