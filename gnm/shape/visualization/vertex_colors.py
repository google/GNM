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

"""Per-vertex colors (V, 3) for GNM, highlighting eyes and teeth."""

from collections.abc import Sequence

from gnm.shape import gnm_numpy
import numpy as np

# UINT8 color values kept for easier pasting into Color Picker tools.
DEFAULT_COLOR = tuple([c / 255 for c in [50, 156, 237]])
ORANGE = tuple([c / 255 for c in [232, 138, 14]])
GREEN = tuple([c / 255 for c in [83, 227, 39]])
CYAN = tuple([c / 255 for c in [39, 227, 227]])

# Defines a mapping for shading particular regions. It is from a vertex group to
# a tuple of modifiers that scales and adds an offset to the given color value.
_VERTEX_GROUP_COLOR_MODIFIERS = {
    'skin': (1.0, 0.0),
    'scleras': (0.6, 0.4),
    'irises': (0.6, 0.0),
    'gums': (0.7, 0.0),
    'teeth': (0.6, 0.4),
    'tongue': (0.7, 0.0),
    'mouth_sock': (0.7, 0.0),
}


def get_vertex_colors(
    gnm_np: gnm_numpy.GNM,
    color: Sequence[float] = DEFAULT_COLOR,
) -> np.ndarray:
  """Per-vertex colors (V, 3) for GNM, highlighting eyes and teeth.

  Args:
    gnm_np: The GNM instance for which the colors will be generated. If None,
      then use the GNM face model.
    color: The RGB color of the skin. Color values should be in [0.0, 1.0]. The
      irises will be darker, and the scleras and teeth will be brighter.

  Returns:
    Per-vertex colors (V, 3) for GNM, highlighting eyes and teeth.
  """
  color = np.array(color)  # pyrefly: ignore[bad-assignment]
  colors = np.zeros((gnm_np.num_vertices, 3))

  for region, (scale, offset) in _VERTEX_GROUP_COLOR_MODIFIERS.items():
    if region in gnm_np.vertex_group_names:
      colors[gnm_np.vertex_group_indices(region)] = (
          color * scale + offset  # pyrefly: ignore[unsupported-operation]
      )
  return colors


def get_vertex_colors_for_inner_head(
    gnm_np: gnm_numpy.GNM,
    base_color: Sequence[float] = DEFAULT_COLOR,
) -> np.ndarray:
  """Per-vertex GNM colors, highlighting mouth sock and teeth/tongue.

  Useful for visualizing the inner part of the head where we need to distinguish
  between the mouth sock, upper teeth, lower teeth, and tongue.

  Args:
    gnm_np: The GNM instance for which the colors will be generated.
    base_color: The RGB color of the skin (see `get_vertex_colors`).

  Returns:
    Per-vertex colors (V, 3) for GNM.

  Raises:
    ValueError: The GNM model does not include a head.
  """
  # The function only makes sense for GNM models includling a head.
  colored_groups = [
      'mouth_sock',
      'upper_teeth_and_gums',
      'lower_teeth_and_gums',
      'tongue',
  ]
  if not np.all([g in gnm_np.vertex_group_names for g in colored_groups]):
    raise ValueError('The GNM model does not include a head.')

  colors = get_vertex_colors(color=base_color, gnm_np=gnm_np)
  colors[gnm_np.vertex_group_indices('mouth_sock')] = ORANGE
  colors[gnm_np.vertex_group_indices('upper_teeth_and_gums')] = GREEN
  colors[gnm_np.vertex_group_indices('lower_teeth_and_gums', 'tongue')] = CYAN
  return colors
