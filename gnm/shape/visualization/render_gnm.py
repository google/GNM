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

"""Visualization utilities for GNM."""

from collections.abc import Callable
from typing import Any

from gnm.shape import gnm_numpy
from gnm.shape.visualization import gnm_pyrender
from gnm.shape.visualization import render_common
import numpy as np
import numpy.typing as npt

GNMMeshRenderer = gnm_pyrender.GNMMeshRenderer
MeshRenderOutput = render_common.MeshRenderOutput


def render_gnm(
    gnm_np: gnm_numpy.GNM,
    vertices: render_common.FloatArray | None = None,
    world_to_camera: render_common.FloatArray | None = None,
    camera_to_image: render_common.FloatArray | None = None,
    image_size: tuple[int, int] = render_common.DEFAULT_IMAGE_SIZE,
    triangles: render_common.TriangleGroups = '~eye_exteriors',
    texture: render_common.Texture = render_common.DEFAULT_TEXTURE,
    multisample_antialiasing: int = 2,
    background_color: render_common.ColorOrImage = (
        render_common.DEFAULT_BACKGROUND_COLOR
    ),
    alpha: float = 1.0,
    vertex_colors: npt.NDArray[np.floating] | None = None,
    multiple_gnms: bool = False,
    include_shading: bool = True,
    verbose: bool = False,
    renderer_factory: Callable[..., Any] = (
        gnm_pyrender.DEFAULT_RENDERER_FACTORY
    ),
) -> render_common.FloatArray:
  """Render GNM meshes.

  Uses a pyrender backend (`GNMMeshRenderer`) to render GNM meshes.

  All arguments specified with (...) in their shape are batchable. They can have
  an arbitrary number of leading dimensions, but they must all be broadcastable
  to the batch size of the biggest of them. e.g. if vertices has shape (3, 100,
  3) and world_to_camera has shape (4, 4), then world_to_camera will be
  broadcast over 3 frames.

  The user can pass the world-to-camera and camera-to-image transformations. If
  they are not given, then cameras are placed in front of each face, looking at
  the face. Note that this function assumes that the world-to-camera and
  camera-to-image transformations follow OpenCV's convention. This means
  that the camera coordinate system has X pointing to the right, Y downwards,
  and Z towards the scene. The camera_to_image should project points in the
  camera coordinate system to pixel coordinates where (0, 0) is the coordinate
  of the top left pixel's top left corner.

  Recall the following GNM dimension notation (from gnm_numpy.py):
  * N: Size of batch.
  * V: Number of vertices.
  * J: Number of joints.
  * I: Identity basis dimensionality.
  * E: Expression basis dimensionality.

  Additionally, we use:
  * M: Number of GNMs per image.

  Args:
    gnm_np: The NumPy GNM model, e.g., if you have a custom version of GNM you
      wish to pose with.
    vertices: Optional posed GNM vertices shaped (..., [M], V, 3). If not
      provided, use the template vertices.
    world_to_camera: Optional world-to-camera transformations, (..., 4, 4). If
      not given, use a default look-at transformation.
    camera_to_image: Optional camera-to-image transformations, (..., 4, 4). If
      not given, use a default fill factor.
    image_size: The width and height of the rendered image in pixels: (W, H).
    triangles: Determines which triangles to render in GNM. Either a vertex
      group name: triangles in that GNM vertex group will be rendered. Or an
      array of triangle indices.
    texture: Optional texture map(s) for GNM. If None, no texture will be used.
      Defaults to a skin edge-flow texture. If given as a single array, this
      will be used for the skin UDIM tile only. If given as a mapping, then the
      keys should be UDIM tiles (see `gnm_udim.UdimTile`; GNM part names are
      deprecated), and the values should be arrays. All arrays should be float32
      [0-1], (..., H, W, 3).
    multisample_antialiasing: Internally render with e.g., double resolution,
      and then downsample for anti-aliasing.
    background_color: Color for the background. Can either be float (gray), an
      RGB tuple, or a background image (..., H, W, 3).
    alpha: An optional float value in [0, 1] range. If provided, it will be used
      to blend the rasterized GNM meshes with the background images.
    vertex_colors: Per-vertex colors, (..., [M], V, 3). In the case of multiple
      GNMs, 'M' will be inferred - if it matches the M dimension in vertices, it
      will be used, otherwise vertex_colors will be broadcast over the M
      dimension in vertices.
    multiple_gnms: If True, vertices is expected to be shape (..., M, V, 3), and
      we render M GNMs per image. Default cameras will be set relative to the
      first GNM in the sequence.
    include_shading: Whether to include shading. If False, the mesh will be
      rendered without light or shading.
    verbose: Whether to print progress bars.
    renderer_factory: Optional factory function returning a
      pyrender.OffscreenRenderer.

  Returns:
    A rendered image of GNM, (..., H, W, 3).

  Raises:
    ValueError: If multiple_gnms=True, but vertices is only 2D.
  """
  vertices, world_to_camera, camera_to_image = (
      render_common.resolve_gnm_camera_and_vertices(
          gnm_np,
          vertices=vertices,
          world_to_camera=world_to_camera,
          camera_to_image=camera_to_image,
          image_size=image_size,
          multiple_gnms=multiple_gnms,
      )
  )
  # Ensure `renderer.close()` frees the offscreen OpenGL context on exit.
  with GNMMeshRenderer.from_gnm(
      gnm_np=gnm_np,
      triangles=triangles,
      texture=texture,
      multisample_antialiasing=multisample_antialiasing,
      vertex_colors=vertex_colors,
      include_shading=include_shading,
      verbose=verbose,
      renderer_factory=renderer_factory,
  ) as renderer:
    render_output: MeshRenderOutput = renderer.render(
        vertices=vertices,
        world_to_camera=world_to_camera,
        camera_to_image=camera_to_image,
        image_size=image_size,
        multiple_gnms=multiple_gnms,
    )
  return render_output.composite(
      background=background_color,
      alpha=alpha,
      dtype=np.float32,
  )
