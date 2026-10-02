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

"""Tests for gnm_xnp."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any

from absl.testing import absltest
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_numpy
from gnm.shape import gnm_xnp
import numpy as np


def _get_dummy_model_data() -> dict[str, Any]:
  """Returns a minimal GNM model data dict with float32/int32 arrays."""
  model_data = {}
  for name in gnm_data_schema.GNM_DATA_ATTRIBUTES:
    model_data[name] = np.zeros((1, 3), dtype=np.float32)
  model_data.update({
      'version': '3.0',
      'variant': 'head',
      'identity_names': ['id1'],
      'joint_names': ['joint1'],
      'expression_names': ['exp1'],
      'joint_parent_indices': np.array([0], dtype=np.int32),
      'quads': np.zeros((1, 4), dtype=np.int32),
      'triangles': np.zeros((1, 3), dtype=np.int32),
      'mirror_indices': np.array([0], dtype=np.int32),
      'mesh_component_names': ['part1'],
      'vertex_group_names': ['group1'],
  })
  return model_data


class GNMXnpTest(absltest.TestCase):

  def test_annotations_match_schema(self):
    expected_attributes = set(gnm_data_schema.GNM_DATA_ATTRIBUTES)
    actual_attributes = set(gnm_xnp.GNM.__annotations__.keys())
    self.assertEqual(actual_attributes, expected_attributes)

  def test_dataclass_fields_match_schema(self):
    expected_attributes = set(gnm_data_schema.GNM_DATA_ATTRIBUTES)
    actual_fields = {f.name for f in dataclasses.fields(gnm_xnp.GNM)}
    self.assertEqual(actual_fields, expected_attributes)

  def test_cannot_instantiate_abstract_gnm(self):
    with self.assertRaises(TypeError):
      gnm_xnp.GNM()  # pytype: disable=not-instantiable,missing-parameter  # pylint: disable=abstract-class-instantiated

  def test_from_model_data_raises_not_implemented(self):
    with self.assertRaises(NotImplementedError):
      gnm_xnp.GNM._from_model_data({})  # pylint: disable=protected-access

  def test_cannot_instantiate_concrete_gnm_via_constructor(self):
    with self.assertRaises(TypeError):
      gnm_numpy.GNM()  # pytype: disable=not-instantiable,missing-parameter

  def test_to_numpy_data_dict_does_not_share_memory_with_instance(self):
    gnm = gnm_numpy.GNM._from_model_data(_get_dummy_model_data())  # pylint: disable=protected-access
    data_dict = gnm.to_numpy_data_dict()

    data_dict['template_vertex_positions'][0, 0] = 1.0
    data_dict['quads'][0, 0] = 1
    data_dict['joint_parent_indices'][0] = 1
    data_dict['joint_names'].append('joint2')
    data_dict['vertex_group_names'].append('group2')

    np.testing.assert_array_equal(
        gnm.template_vertex_positions, np.zeros((1, 3))
    )
    np.testing.assert_array_equal(gnm.quads, np.zeros((1, 4)))
    np.testing.assert_array_equal(gnm.joint_parent_indices, [0])
    self.assertEqual(gnm.joint_names, ['joint1'])
    self.assertEqual(gnm.vertex_group_names, ['group1'])

  def test_from_gnm_does_not_share_memory_with_source_instance(self):
    source = gnm_numpy.GNM._from_model_data(_get_dummy_model_data())  # pylint: disable=protected-access
    gnm = gnm_numpy.GNM.from_gnm(source)

    gnm.template_vertex_positions[0, 0] = 1.0
    gnm.quads[0, 0] = 1
    typing.cast(list[str], gnm.joint_names).append('joint2')
    source.template_joint_positions[0, 0] = 1.0
    typing.cast(list[str], source.vertex_group_names).append('group2')

    np.testing.assert_array_equal(
        source.template_vertex_positions, np.zeros((1, 3))
    )
    np.testing.assert_array_equal(source.quads, np.zeros((1, 4)))
    self.assertEqual(source.joint_names, ['joint1'])
    np.testing.assert_array_equal(
        gnm.template_joint_positions, np.zeros((1, 3))
    )
    self.assertEqual(gnm.vertex_group_names, ['group1'])


if __name__ == '__main__':
  absltest.main()
