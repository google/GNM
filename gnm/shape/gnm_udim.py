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

"""UDIM texture coordinate layout and tile assignment utilities for GNM."""

from collections.abc import Mapping
import enum
from typing import Any

import immutabledict
import jaxtyping as jt
import numpy as np

# The first UDIM tile, as defined by the Mari/OpenUSD UDIM convention: a tile
# whose lower-left corner is at integer texture coordinate (u, v) is numbered
# `1001 + u + 10 * v`, and tiles are laid out ten per row.
UDIM_FIRST_TILE: int = 1001
UDIM_TILES_PER_ROW: int = 10


class UdimTile(enum.IntEnum):
  """The UDIM tile of each textured region of the mesh.

  GNM's texture coordinates all lie in the unit square, and distinct surfaces
  overlap there so that each can be given its own texture in tools like
  Blender. The UDIM convention expresses the same intent in a standard,
  interchangeable way by placing each surface in its own tile, which
  additionally allows each tile to have its own resolution.

  The eyes are split because an eye's interior lies entirely inside its
  exterior in UV space, so a tile per eye would still hold two surfaces
  competing for the same texels. The two teeth components are merged because
  they are already disjoint in UV and so cost nothing to share.

  Being an `IntEnum`, a tile compares and hashes equal to its tile number, so
  it can be used interchangeably with the plain integers of `quad_udim_tiles`
  and `triangle_udim_tiles`.
  """

  SKIN = 1001
  LEFT_EYE_INTERIOR = 1002
  LEFT_EYE_EXTERIOR = 1003
  RIGHT_EYE_INTERIOR = 1004
  RIGHT_EYE_EXTERIOR = 1005
  TEETH_AND_GUMS = 1006
  TONGUE = 1007


# The region of the mesh assigned to each UDIM tile, as a vertex group
# expression in the syntax accepted by `GNM.vertex_group_mask`, so `'&name'`
# intersects and a bare name unions.
#
# The table is fixed rather than derived from the order of
# `mesh_component_names`, because that order differs between model variants
# (the head and body variants swap `tongue` and `lower_teeth_and_gums`). A
# fixed table keeps a region on the same tile, and so in the same texture
# file, for every variant. Entries whose groups are absent from a variant are
# skipped, which is how the hand model gets a single skin tile.
UDIM_TILE_REGIONS: Mapping[UdimTile, tuple[str, ...]] = (
    immutabledict.immutabledict({
        UdimTile.SKIN: ('skin',),
        UdimTile.LEFT_EYE_INTERIOR: ('left_eye', '&eye_interiors'),
        UdimTile.LEFT_EYE_EXTERIOR: ('left_eye', '&eye_exteriors'),
        UdimTile.RIGHT_EYE_INTERIOR: ('right_eye', '&eye_interiors'),
        UdimTile.RIGHT_EYE_EXTERIOR: ('right_eye', '&eye_exteriors'),
        UdimTile.TEETH_AND_GUMS: (
            'upper_teeth_and_gums',
            'lower_teeth_and_gums',
        ),
        UdimTile.TONGUE: ('tongue',),
    })
)


def assign_tiles(
    model: Any,
    faces: jt.Int[np.ndarray, 'F C'],
) -> jt.Int32[np.ndarray, 'F']:
  """Assigns each face the tile of the `UDIM_TILE_REGIONS` region it lies in.

  Args:
    model: The model whose vertex groups define the regions.
    faces: The vertex indices of each face, (F, C), for F faces with C corners
      each.

  Returns:
    The UDIM tile number of each face, (F,).

  Raises:
    ValueError: If two regions claim the same face, or if some face is claimed
      by none and so cannot be assigned a tile.
  """
  faces = np.asarray(faces)
  tiles = np.zeros(faces.shape[0], dtype=np.int32)
  defined = set(model.vertex_group_names)
  for tile, groups in UDIM_TILE_REGIONS.items():
    if not all(g.lstrip('|&-~') in defined for g in groups):
      continue
    mask = model.vertex_group_mask(*groups)
    indices = np.where(np.all(mask[faces], axis=-1))[0]
    claimed = tiles[indices]
    if np.any(claimed):
      raise ValueError(
          f'UDIM tile {tile} claims {int(np.count_nonzero(claimed))} faces '
          f'already claimed by tile(s) {sorted(set(claimed[claimed != 0]))}. '
          'The regions of UDIM_TILE_REGIONS must be disjoint.'
      )
    tiles[indices] = tile

  num_unassigned = int(np.count_nonzero(tiles == 0))
  if num_unassigned:
    raise ValueError(
        f'{num_unassigned} of {tiles.size} faces lie in no region of '
        'UDIM_TILE_REGIONS, so they cannot be assigned a UDIM tile.'
    )
  return tiles


def offset_uvs_into_tiles(
    uvs: jt.Float[np.ndarray, 'F C 2'],
    tiles: jt.Int32[np.ndarray, 'F'],
) -> jt.Float[np.ndarray, 'F C 2']:
  """Translates per-face texture coordinates into their UDIM tiles.

  Args:
    uvs: Texture coordinates in the unit square, (F, C, 2), for F faces with C
      corners each.
    tiles: The UDIM tile number of each face, (F,).

  Returns:
    The texture coordinates offset into their tiles, (F, C, 2).
  """
  tile_indices = tiles - UDIM_FIRST_TILE
  offsets = np.stack(
      [tile_indices % UDIM_TILES_PER_ROW, tile_indices // UDIM_TILES_PER_ROW],
      axis=-1,
  ).astype(uvs.dtype)
  return uvs + offsets[:, None, :]


def quad_tiles(model: Any) -> jt.Int32[np.ndarray, 'Q']:
  """Returns the UDIM tile number of each quad in the model."""
  return assign_tiles(model, np.asarray(model.quads))


def triangle_tiles(model: Any) -> jt.Int32[np.ndarray, 'T']:
  """Returns the UDIM tile number of each triangle in the model."""
  return assign_tiles(model, np.asarray(model.triangles))


def quad_uvs(model: Any) -> jt.Float[np.ndarray, 'Q 4 2']:
  """Returns per-quad texture coordinates in UDIM layout."""
  return offset_uvs_into_tiles(np.asarray(model.quad_uvs), quad_tiles(model))


def triangle_uvs(model: Any) -> jt.Float[np.ndarray, 'T 3 2']:
  """Returns per-triangle texture coordinates in UDIM layout."""
  return offset_uvs_into_tiles(
      np.asarray(model.triangle_uvs), triangle_tiles(model)
  )
