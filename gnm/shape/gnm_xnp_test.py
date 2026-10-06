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

from absl.testing import absltest
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_numpy
from gnm.shape import gnm_testing_factory_mixins
from gnm.shape import gnm_xnp
import numpy as np


class PlaceholderGNM(
    gnm_testing_factory_mixins.GNMTestingFactoryMethodsMixin,
    gnm_numpy.GNM,
):
  """Concrete GNM subclass with testing factory methods."""


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
      gnm_xnp.GNM()  # pylint: disable=abstract-class-instantiated  # pyrefly: ignore[bad-instantiation]

  def test_cannot_instantiate_concrete_gnm_via_constructor(self):
    with self.assertRaises(TypeError):
      PlaceholderGNM()

  def test_to_numpy_data_dict_does_not_share_memory_with_instance(self):
    gnm = PlaceholderGNM.from_placeholder()
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
    source = PlaceholderGNM.from_placeholder()
    gnm = PlaceholderGNM.from_gnm(source)

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

  def test_from_model_data_fails_for_missing_fields(self):
    gnm = PlaceholderGNM.from_placeholder()
    data_dict = gnm.to_numpy_data_dict()
    del data_dict['template_vertex_positions']

    with self.assertRaisesRegex(
        ValueError, "Missing:.*'template_vertex_positions'"
    ):
      PlaceholderGNM._from_model_data(data_dict)

  def test_from_model_data_fails_for_extra_fields(self):
    with self.assertRaisesRegex(ValueError, "Extra:.*'unexpected_field'"):
      PlaceholderGNM.from_placeholder(
          unexpected_field=1,
      )


if __name__ == '__main__':
  absltest.main()
