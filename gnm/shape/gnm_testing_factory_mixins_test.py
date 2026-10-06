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

"""Unit tests for GNM testing factory methods mixin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from absl.testing import absltest
from gnm.shape import gnm_data_loader
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_testing_factory_mixins
from gnm.shape.data.versions import gnm_specs
from gnm.shape.data.versions import gnm_test_catalog
import numpy as np


class _PlaceholderGNM(gnm_testing_factory_mixins.GNMTestingFactoryMethodsMixin):
  """Placeholder GNM class for testing factory mixin methods."""

  def __init__(self, model_data: Mapping[str, Any]):
    self.model_data = model_data

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> Self:
    return cls(model_data=model_data)


class GNMTestingFactoryMethodsMixinTest(absltest.TestCase):

  def test_from_placeholder_populates_default_model_data(self):
    gnm = _PlaceholderGNM.from_placeholder()
    expected_major_version = gnm_specs.GNMMajorVersion(
        gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[-1].removeprefix('v')
    )
    expected_version = gnm_data_loader.major_to_newest_full_version(
        expected_major_version
    )
    self.assertIsInstance(gnm, _PlaceholderGNM)
    self.assertEqual(
        set(gnm.model_data.keys()),
        set(gnm_data_schema.GNM_DATA_ATTRIBUTES),
    )
    self.assertEqual(gnm.model_data['version'], expected_version)
    self.assertEqual(gnm.model_data['variant'], gnm_specs.GNMVariant.HEAD)

  def test_from_placeholder_overrides_default_fields_with_kwargs(self):
    custom_vertices = np.ones((2, 3), dtype=np.float32)
    gnm = _PlaceholderGNM.from_placeholder(
        identity_names=['custom_id'],
        template_vertex_positions=custom_vertices,
    )
    self.assertEqual(gnm.model_data['variant'], gnm_specs.GNMVariant.HEAD)
    self.assertEqual(gnm.model_data['identity_names'], ['custom_id'])
    np.testing.assert_array_equal(
        gnm.model_data['template_vertex_positions'],
        custom_vertices,
    )
    self.assertEqual(gnm.model_data['joint_names'], ['joint1'])


if __name__ == '__main__':
  absltest.main()
