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

"""Utilities for rendering meshes with UDIM texture coordinates.

In the UDIM convention, the integer part of a texture coordinate selects a
texture (a tile) and the fractional part addresses it, see
`gnm_udim.UdimTile`. Renderers that take one texture per primitive split such
a mesh by tile.
"""

from typing import TypeAlias

from gnm.shape import gnm_typing
from gnm.shape import gnm_udim
import numpy as np
import numpy.typing as npt

enpt = gnm_typing.enpt

TileGeometry: TypeAlias = tuple[
    enpt.IntArray['C_tile'],  # pyrefly: ignore[unknown-name]
    enpt.IntArray['F_tile 3'],  # pyrefly: ignore[unknown-name]
    enpt.FloatArray['C_tile 2'],  # pyrefly: ignore[unknown-name]
]


def split_triangles_by_tile(
    triangle_uvs: npt.ArrayLike,
) -> dict[int, tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]]:
  """Splits triangles with UDIM texture coordinates by tile.

  Args:
    triangle_uvs: Per-corner texture coordinates in UDIM layout, (F, 3, 2).

  Returns:
    For each UDIM tile, the indices of the triangles lying in it, (F_tile,), and
    their texture coordinates in the unit square of the tile, (F_tile, 3, 2).
  """
  triangle_uvs = np.asarray(triangle_uvs)
  # Take the tile from the component-wise minimum of the corners, as a corner on
  # the upper boundary of a tile has an integer coordinate that would place it
  # in the next tile.
  tile_offsets = np.floor(triangle_uvs.min(axis=1)).astype(np.int32)
  tiles = (
      gnm_udim.UDIM_FIRST_TILE
      + tile_offsets[:, 0]
      + gnm_udim.UDIM_TILES_PER_ROW * tile_offsets[:, 1]
  )
  split = {}
  for tile in np.unique(tiles).tolist():
    indices = np.flatnonzero(tiles == tile)
    split[tile] = (indices, triangle_uvs[indices] - tile_offsets[indices, None])
  return split


def build_tile_geometry(
    triangles: enpt.IntArray['F 3'],
    triangle_uvs: enpt.FloatArray['F 3 2'],
) -> dict[int, TileGeometry]:
  """Splits triangles by UDIM tile and unwelds corners per tile."""
  if len(triangles) == 0:
    return {}

  tile_geometry: dict[int, TileGeometry] = {}
  for tile, (tile_triangles, tile_uvs) in split_triangles_by_tile(
      triangle_uvs
  ).items():
    corners = np.asarray(triangles[tile_triangles].ravel(), dtype=np.int32)
    faces = np.arange(corners.size, dtype=np.int32).reshape(-1, 3)
    unwelded_uvs = np.asarray(tile_uvs.reshape(-1, 2), dtype=np.float32)
    tile_geometry[int(tile)] = (corners, faces, unwelded_uvs)
  return tile_geometry
