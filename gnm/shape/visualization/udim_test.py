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

"""Tests for UDIM tile splitting and geometry builders in udim."""

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


class BuildTileGeometryTest(absltest.TestCase):

  def test_empty_triangles_returns_empty_dict(self):
    triangles = np.zeros((0, 3), dtype=np.int32)
    triangle_uvs = np.zeros((0, 3, 2), dtype=np.float32)
    self.assertEqual(udim.build_tile_geometry(triangles, triangle_uvs), {})

  def test_builds_tile_geometry_and_unwelds_corners(self):
    unit_triangle = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32
    )
    triangle_uvs = np.stack([
        unit_triangle,  # Tile 1001.
        unit_triangle * 0.5 + [2.0, 1.0],  # Tile 1013.
        unit_triangle * 0.5 + [0.25, 0.25],  # Tile 1001.
    ])
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)

    tile_geometry = udim.build_tile_geometry(triangles, triangle_uvs)

    self.assertEqual(list(tile_geometry.keys()), [1001, 1013])

    corners_1001, faces_1001, unwelded_uvs_1001 = tile_geometry[1001]
    np.testing.assert_array_equal(corners_1001, [0, 1, 2, 6, 7, 8])
    np.testing.assert_array_equal(faces_1001, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_allclose(
        unwelded_uvs_1001, triangle_uvs[[0, 2]].reshape(-1, 2)
    )

    corners_1013, faces_1013, unwelded_uvs_1013 = tile_geometry[1013]
    np.testing.assert_array_equal(corners_1013, [3, 4, 5])
    np.testing.assert_array_equal(faces_1013, [[0, 1, 2]])
    np.testing.assert_allclose(
        unwelded_uvs_1013, (unit_triangle * 0.5).reshape(-1, 2)
    )


if __name__ == '__main__':
  absltest.main()
