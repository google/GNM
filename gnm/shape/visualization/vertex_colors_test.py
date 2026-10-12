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

"""Tests for vertex colors visualization."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
from gnm.shape import gnm_test_utils
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import vertex_colors
import numpy as np

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_test_catalog.MAJOR_VERSION_TO_VARIANTS_MAP

_SUPPORTED_VARIANTS = frozenset([
    gnm_numpy.GNMVariant.HEAD,
])


class VertexColorsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.models = gnm_test_utils.LazyGNMDict(
        load_fn=functools.partial(gnm_test_utils.load_gnm, gnm_numpy.GNM),
        variants_by_version={
            version: [
                variant
                for variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]
                if variant in _SUPPORTED_VARIANTS
            ]
            for version in _MAINTAINED_MAJOR_GNM_VERSIONS
        },
    )
    cls.addClassCleanup(cls.models.clear)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_get_vertex_colors(self, version, variant):
    """Tests we can get per-vertex colors."""
    gnm_np = self.models.get_or_skip(version, variant)
    colors = vertex_colors.get_vertex_colors(gnm_np)

    # Sanity check on color values.
    self.assertTrue(np.all(0.0 <= colors) and np.all(colors <= 1.0))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=[gnm_numpy.GNMVariant.HEAD],
  )
  def test_gets_inner_head_colors(self, version, variant):
    gnm_np = self.models.get_or_skip(version, variant)
    colors = vertex_colors.get_vertex_colors_for_inner_head(gnm_np)
    groups = (
        ('mouth_sock',),
        ('upper_teeth_and_gums',),
        ('lower_teeth_and_gums', 'tongue'),
    )

    with self.subTest('Correct shape'):
      self.assertEqual(colors.shape, (gnm_np.num_vertices, 3))

    # Check that the colors in each interst group are the same.
    with self.subTest('Same color in each group'):
      for group in groups:
        indices = gnm_np.vertex_group_indices(*group)
        self.assertTrue(np.all(colors[indices] == colors[indices[0]]))

    # Check that each group has a different color.
    with self.subTest('Different colors across groups'):
      group_colors = [
          colors[gnm_np.vertex_group_indices(*g)[0]] for g in groups
      ]
      self.assertLen(np.unique(np.array(group_colors), axis=1), len(groups))


if __name__ == '__main__':
  absltest.main(testLoader=gnm_test_utils.ModelOrderedTestLoader())
