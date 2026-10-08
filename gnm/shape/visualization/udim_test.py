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

from absl.testing import absltest
from gnm.shape.visualization import udim
import numpy as np


class SplitTrianglesByTileTest(absltest.TestCase):

  def test_splits_by_tile_and_localizes_uvs(self):
    unit_triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    triangle_uvs = np.stack([
        unit_triangle,  # Tile 1001, with corners on the upper tile boundary.
        unit_triangle * 0.5 + [2.0, 1.0],  # Tile 1013.
        unit_triangle * 0.5 + [0.25, 0.25],  # Tile 1001.
    ])

    split = udim.split_triangles_by_tile(triangle_uvs)

    self.assertEqual(list(split), [1001, 1013])
    np.testing.assert_array_equal(split[1001][0], [0, 2])
    np.testing.assert_array_equal(split[1013][0], [1])
    np.testing.assert_allclose(split[1001][1], triangle_uvs[[0, 2]])
    np.testing.assert_allclose(split[1013][1], unit_triangle[None] * 0.5)


if __name__ == '__main__':
  absltest.main()
