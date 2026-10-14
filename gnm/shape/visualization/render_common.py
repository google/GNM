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

"""Shared helpers for render_gnm backends (Pyrender, etc.).

This module factors out the backend-agnostic pieces that are common to the
different `render_gnm` implementations: default scene parameters, texture
loading, camera/projection helpers and batching utilities.
"""

import abc
from collections.abc import Sequence
import dataclasses
import functools
from typing import Protocol, TypeAlias

from etils import epath
from gnm.shape import gnm_numpy
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import vertex_colors as vertex_colors_module
import imageio
import immutabledict
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import tqdm.autonotebook as tqdm

FloatArray = npt.NDArray[np.floating]
ColorOrImage = npt.NDArray[np.uint8] | FloatArray | Sequence[float] | float

_pkg = __package__ or 'gnm.shape.visualization'
_TEXTURES_DIR = epath.resource_path(_pkg).parent / 'data' / 'textures'
EDGEFLOW_TEXTURE_BY_BODY_PART = immutabledict.immutabledict({
    gnm_numpy.GNMBodyPart.HEAD: _TEXTURES_DIR / 'edgeflow_bw_4k.png',
})

# Default parameters for scene.
DEFAULT_IMAGE_SIZE = (240, 320)
DEFAULT_CAMERA_DISTANCE = 2.0
DEFAULT_NEAR = 0.01
DEFAULT_FAR = 100.0
DEFAULT_TARGET_FILL_FACTOR = 0.4
DEFAULT_BACKGROUND_COLOR = (0.95, 0.95, 0.95)


# Placeholder for lazy loading the default texture.
class DefaultTexture:
  """Placeholder for lazy loading the default texture."""


DEFAULT_TEXTURE = DefaultTexture()
Texture = FloatArray | DefaultTexture | dict[str, FloatArray] | None
TriangleGroups: TypeAlias = str | Sequence[str]


class BackendRenderFn(Protocol):
  """Protocol for backend rendering functions.

  Backend render functions accept batched scene parameters and rasterize or
  raytrace the scene, returning rendered images of shape (N, H, W, 3).
  """

  def __call__(
      self,
      vertices: FloatArray,
      triangles: dict[str, np.ndarray],
      world_to_camera: FloatArray,
      camera_to_image: FloatArray,
      vertex_normals: FloatArray,
      vertex_uvs: FloatArray,
      vertex_colors: FloatArray,
      image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
      texture: dict[str, np.ndarray] | None = None,
      multisample_antialiasing: int = 1,
      background_color: FloatArray | None = None,
      alpha: float = 1.0,
      include_shading: bool = True,
      verbose: bool = False,
  ) -> np.ndarray:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class MeshRenderOutput:
  """Output of the stateful GNM 3D mesh renderer.

  Attributes:
    image: Rendered RGB image as a uint8 NumPy array of shape (..., H, W, 3).
    alpha: Rendered coverage/opacity mask in [0.0, 1.0] as a float32 NumPy array
      of shape (..., H, W, 1) after multisample anti-aliasing.
  """

  image: jt.UInt8[np.ndarray, '... H W 3']
  alpha: jt.Float[np.ndarray, '... H W 1']

  def composite(
      self,
      background_image: jt.UInt8[np.ndarray, '... H W 3'] | None = None,
      *,
      alpha: float = 1.0,
      default_background_color: tuple[float, float, float] = (
          DEFAULT_BACKGROUND_COLOR
      ),
  ) -> jt.UInt8[np.ndarray, '... H W 3']:
    """Composites the rendered mesh foreground over a background image.

    Args:
      background_image: Optional uint8 RGB background image array of shape (...,
        H, W, 3). If None, `default_background_color` is used.
      alpha: Global opacity multiplier applied to `self.alpha`. Defaults to 1.0.
      default_background_color: RGB float tuple in [0.0, 1.0] for the solid
        background when `background_image` is None.

    Returns:
      Composited uint8 RGB image array of shape (..., H, W, 3).
    """
    foreground_f32 = np.asarray(self.image, dtype=np.float32) / 255.0
    height, width = foreground_f32.shape[-3:-1]
    if background_image is not None:
      background_f32 = np.asarray(background_image, dtype=np.float32) / 255.0
    else:
      background_f32 = np.broadcast_to(
          np.asarray(default_background_color, dtype=np.float32),
          (height, width, 3),
      )
    effective_alpha = float(alpha) * np.asarray(self.alpha, dtype=np.float32)
    composited_f32 = foreground_f32 * effective_alpha + background_f32 * (
        1.0 - effective_alpha
    )
    return np.clip(composited_f32 * 255.0, 0.0, 255.0).astype(np.uint8)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BaseGNMMeshRenderer(abc.ABC):
  """Abstract base class for stateful 3D GNM mesh renderers.

  Caches static mesh topology, UV coordinates, vertex colors, and loaded
  textures for a `gnm_numpy.GNM` instance across frames and repeated render
  calls.

  Attributes:
    gnm_np: NumPy GNM shape model used for topology, normals, and textures.
    triangles: Vertex group arithmetic expression string (e.g.
      `'~eye_exteriors'`) or sequence of group expression strings (e.g.
      `('~eye_exteriors', '-teeth')`).
    texture: Texture configuration (`DEFAULT_TEXTURE`, custom texture
      array/mapping, or `None` to disable texture).
    multisample_antialiasing: Supersampling factor for anti-aliasing. Defaults
      to 1 (native resolution).
    verbose: Default flag controlling whether to display a progress bar.
  """

  gnm_np: gnm_numpy.GNM
  triangles: TriangleGroups = '~eye_exteriors'
  texture: Texture = DEFAULT_TEXTURE
  multisample_antialiasing: int = 1
  verbose: bool = False

  _triangle_dict: dict[str, jt.Int[np.ndarray, 'F_part 3']] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _vertex_colors: jt.Float[np.ndarray, 'V 3'] = dataclasses.field(
      default_factory=lambda: np.empty((0, 3), dtype=np.float32),
      init=False,
      repr=False,
      compare=False,
  )
  _vertex_uvs: jt.Float[np.ndarray, 'V 2'] = dataclasses.field(
      default_factory=lambda: np.empty((0, 2), dtype=np.float32),
      init=False,
      repr=False,
      compare=False,
  )
  _raw_textures: dict[str, jt.UInt8[np.ndarray, '... H_tex W_tex 3']] = (
      dataclasses.field(
          default_factory=dict, init=False, repr=False, compare=False
      )
  )

  def __post_init__(self) -> None:
    object.__setattr__(
        self,
        '_triangle_dict',
        self._compute_triangle_dict(self.triangles),
    )

    raw_vertex_colors = vertex_colors_module.get_vertex_colors(
        self.gnm_np, vertex_colors_module.DEFAULT_COLOR
    )
    if raw_vertex_colors.shape[-1] == 4:
      raw_vertex_colors = raw_vertex_colors[..., :3]
    object.__setattr__(
        self,
        '_vertex_colors',
        np.asarray(raw_vertex_colors, dtype=np.float32),
    )
    object.__setattr__(
        self,
        '_vertex_uvs',
        np.asarray(self.gnm_np.vertex_uvs, dtype=np.float32),
    )
    object.__setattr__(
        self,
        '_raw_textures',
        load_texture(self.gnm_np, self.texture),
    )

  def update_triangles(
      self,
      triangles: TriangleGroups,
  ) -> None:
    """Updates active mesh triangles and syncs or invalidates backend state."""
    new_triangle_dict = self._compute_triangle_dict(triangles)
    if self._triangle_dict and all(
        part_name in self._triangle_dict
        and np.array_equal(self._triangle_dict[part_name], new_faces)
        for part_name, new_faces in new_triangle_dict.items()
    ):
      object.__setattr__(self, 'triangles', triangles)
      return

    object.__setattr__(self, 'triangles', triangles)
    object.__setattr__(self, '_triangle_dict', new_triangle_dict)
    self._on_triangles_updated()

  @abc.abstractmethod
  def update_vertices(
      self,
      vertices: jt.Float[jt.ArrayLike, 'V 3'],
      vertex_normals: jt.Float[jt.ArrayLike, 'V 3'] | None = None,
      *,
      max_tex_dim: int = 1024,
  ) -> None:
    """Updates scene mesh vertices and normals in-place."""

  @abc.abstractmethod
  def update_camera(
      self,
      world_to_camera: jt.Float[np.ndarray, '4 4'],
      camera_to_image: (
          jt.Float[np.ndarray, '4 4'] | jt.Float[np.ndarray, '3 3']
      ),
      image_size: tuple[int, int],
  ) -> None:
    """Updates camera transform and projection parameters in OpenCV convention."""

  def render(
      self,
      vertices: jt.Float[jt.ArrayLike, '... V 3'] | None = None,
      world_to_camera: jt.Float[np.ndarray, '... 4 4'] | None = None,
      camera_to_image: (
          jt.Float[np.ndarray, '... 4 4']
          | jt.Float[np.ndarray, '... 3 3']
          | None
      ) = None,
      *,
      image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
      triangles: TriangleGroups | None = None,
      verbose: bool | None = None,
  ) -> MeshRenderOutput:
    """Renders single-frame or batched 3D GNM meshes.

    Args:
      vertices: Optional 3D mesh vertices of shape (..., V, 3). If None,
        defaults to `self.gnm_np.template_vertex_positions`.
      world_to_camera: Optional OpenCV world-to-camera matrices of shape (...,
        4, 4). If None, defaults to `get_look_at_world_to_camera`.
      camera_to_image: Optional OpenCV camera-to-image matrices of shape (...,
        4, 4) or (..., 3, 3). If None, defaults to
        `get_fill_factor_camera_to_image`.
      image_size: Output image dimensions as (width, height).
      triangles: Optional triangle group expression or explicit face index array
        to update active mesh triangles before rendering.
      verbose: Optional progress-bar verbosity override. Defaults to
        `self.verbose`.

    Returns:
      A `MeshRenderOutput` containing the rendered uint8 RGB image of shape
      (..., height, width, 3) and float32 coverage mask in [0.0, 1.0] of shape
      (..., height, width, 1).
    """
    if triangles is not None:
      self.update_triangles(triangles)

    resolved_verbose = self.verbose if verbose is None else verbose

    if vertices is None:
      vertices_arr = np.asarray(
          self.gnm_np.template_vertex_positions, dtype=np.float32
      )
    else:
      vertices_arr = np.asarray(vertices, dtype=np.float32)

    if world_to_camera is None:
      world_to_camera_arr = np.asarray(
          get_look_at_world_to_camera(self.gnm_np, vertices_arr),
          dtype=np.float32,
      )
    else:
      world_to_camera_arr = np.asarray(world_to_camera, dtype=np.float32)

    if camera_to_image is None:
      camera_to_image_arr = np.asarray(
          get_fill_factor_camera_to_image(
              self.gnm_np, vertices_arr, image_size=image_size
          ),
          dtype=np.float32,
      )
    else:
      camera_to_image_arr = np.asarray(camera_to_image, dtype=np.float32)

    width, height = int(image_size[0]), int(image_size[1])
    resolved_image_size = (width, height)
    render_width = width * self.multisample_antialiasing
    render_height = height * self.multisample_antialiasing
    max_tex_dim = max(render_width, render_height)

    batch_dims = get_batch_dim(
        (vertices_arr, 2),
        (world_to_camera_arr, 2),
        (camera_to_image_arr, 2),
    )
    num_frames = int(np.prod(batch_dims)) if batch_dims else 1

    batched_vertices = np.broadcast_to(
        vertices_arr, (*batch_dims, *vertices_arr.shape[-2:])
    ).reshape(num_frames, *vertices_arr.shape[-2:])
    batched_normals = np.asarray(
        self.gnm_np.compute_vertex_normals(
            batched_vertices[:, np.newaxis, :, :]
        )[:, 0, :, :],
        dtype=np.float32,
    )
    batched_world_to_camera = np.broadcast_to(
        world_to_camera_arr, (*batch_dims, *world_to_camera_arr.shape[-2:])
    ).reshape(num_frames, *world_to_camera_arr.shape[-2:])
    batched_camera_to_image = np.broadcast_to(
        camera_to_image_arr, (*batch_dims, *camera_to_image_arr.shape[-2:])
    ).reshape(num_frames, *camera_to_image_arr.shape[-2:])

    foreground_frames: list[jt.Float[np.ndarray, 'H W 3']] = []
    alpha_frames: list[jt.Float[np.ndarray, 'H W 1']] = []
    frame_indices = tqdm.trange(
        num_frames,
        disable=not resolved_verbose,
        desc=self._progress_bar_desc(),
        leave=False,
    )

    for frame_idx in frame_indices:
      self.update_camera(
          world_to_camera=batched_world_to_camera[frame_idx],
          camera_to_image=batched_camera_to_image[frame_idx],
          image_size=resolved_image_size,
      )
      self.update_vertices(
          vertices=batched_vertices[frame_idx],
          vertex_normals=batched_normals[frame_idx],
          max_tex_dim=max_tex_dim,
      )
      frame_rgb, frame_alpha = self._render_single_frame(
          width=width, height=height
      )
      foreground_frames.append(frame_rgb)
      alpha_frames.append(frame_alpha)

    self._after_batch_render()

    stacked_fg = np.stack(foreground_frames, axis=0).reshape(
        *batch_dims, height, width, 3
    )
    stacked_alpha = np.stack(alpha_frames, axis=0).reshape(
        *batch_dims, height, width, 1
    )
    fg_uint8 = np.clip(stacked_fg * 255.0, 0.0, 255.0).astype(np.uint8)
    return MeshRenderOutput(image=fg_uint8, alpha=stacked_alpha)

  def close(self) -> None:
    """Releases any persistent backend rendering resources."""

  @abc.abstractmethod
  def _on_triangles_updated(self) -> None:
    """Synchronizes or invalidates backend state after `_triangle_dict` changes."""

  @abc.abstractmethod
  def _render_single_frame(
      self,
      width: int,
      height: int,
  ) -> tuple[jt.Float[np.ndarray, 'H W 3'], jt.Float[np.ndarray, 'H W 1']]:
    """Renders the current scene state for one frame at (height, width)."""

  def _after_batch_render(self) -> None:
    """Hook invoked once after completing a batch of frame renders."""

  def _progress_bar_desc(self) -> str:
    """Returns the progress bar description string for this backend."""
    return 'Rendering'

  def _compute_triangle_dict(
      self,
      triangles: TriangleGroups,
  ) -> dict[str, jt.Int[np.ndarray, 'F_part 3']]:
    """Resolves vertex group arithmetic expressions to per-component faces."""
    group_names = (
        (triangles,) if isinstance(triangles, str) else tuple(triangles)
    )
    all_triangle_indices = self.gnm_np.triangle_indices_for_group(*group_names)
    triangle_dict: dict[str, jt.Int[np.ndarray, 'F_part 3']] = {}
    for part_name in self.gnm_np.mesh_component_names:
      group_triangle_indices = self.gnm_np.triangle_indices_for_group(part_name)
      intersection = np.intersect1d(
          group_triangle_indices, all_triangle_indices
      )
      triangle_dict[part_name] = np.asarray(
          self.gnm_np.triangles[intersection], dtype=np.int32
      )
    return triangle_dict


def render_gnm_mesh(
    gnm_np: gnm_numpy.GNM,
    backend_render_fn: BackendRenderFn,
    *,
    vertices: FloatArray | None = None,
    world_to_camera: FloatArray | None = None,
    camera_to_image: FloatArray | None = None,
    convert_cameras_to_opengl: bool = False,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    triangles: str | npt.NDArray[np.integer] = '~eye_exteriors',
    texture: Texture = DEFAULT_TEXTURE,
    multisample_antialiasing: int = 2,
    background_color: ColorOrImage = DEFAULT_BACKGROUND_COLOR,
    alpha: float = 1.0,
    vertex_colors: npt.NDArray[np.floating] | None = None,
    multiple_gnms: bool = False,
    include_shading: bool = True,
    verbose: bool = False,
) -> FloatArray:
  """Shared rendering implementation for GNM mesh renderers.

  Prepares geometry, textures, camera transformations, and batching, then
  delegates the rasterization or raytracing to `backend_render_fn`.

  Args:
    gnm_np: The GNM model.
    backend_render_fn: The backend function that accepts batched parameters and
      returns rendered images of shape (N, H, W, 3).
    vertices: Optional posed GNM vertices shaped (..., [M], V, 3). If not
      provided, use template vertices.
    world_to_camera: Optional world-to-camera transformations in OpenCV
      convention, (..., 4, 4).
    camera_to_image: Optional camera-to-image transformations in OpenCV
      convention, (..., 4, 4) or (..., 3, 3).
    convert_cameras_to_opengl: Whether to convert OpenCV camera matrices to
      OpenGL conventions before passing to `backend_render_fn`.
    image_size: The width and height of the rendered image in pixels: (W, H).
    triangles: Determines which triangles to render. Either a vertex group name
      or an array of triangle indices.
    texture: Optional texture map(s) for GNM.
    multisample_antialiasing: Multiplier for MSAA.
    background_color: Color for the background: float, RGB tuple, or image.
    alpha: Float in [0, 1] for blending rasterized GNM meshes with background.
    vertex_colors: Per-vertex RGB(A) colors, (..., [M], V, 3 or 4).
    multiple_gnms: If True, vertices is expected to be (..., M, V, 3).
    include_shading: Whether to include shading/lighting in the render.
    verbose: Whether to display progress.

  Returns:
    Rendered image array of shape (..., H, W, 3).

  Raises:
    ValueError: If multiple_gnms=True but vertices is 2D, or if texture keys
      are not GNM part names, or if batch dimensions are incompatible.
  """
  width, height = image_size

  triangle_dict = {}
  all_triangle_indices = triangles
  if isinstance(triangles, str):
    all_triangle_indices = gnm_np.triangle_indices_for_group(triangles)

  for part_name in gnm_np.mesh_component_names:
    group_triangle_indices = gnm_np.triangle_indices_for_group(part_name)
    intersection = np.intersect1d(group_triangle_indices, all_triangle_indices)
    triangle_dict[part_name] = gnm_np.triangles[intersection]

  if vertices is None:
    vertices = gnm_np.template_vertex_positions

  if not multiple_gnms:
    vertices = vertices[..., None, :, :]  # Inject 'M' dimension.
  elif (vertices_dim := vertices.ndim) < 3:
    raise ValueError(
        f'Called with {multiple_gnms=}, but vertices is only {vertices_dim}D.'
    )

  if vertex_colors is None:
    vertex_colors = vertex_colors_module.get_vertex_colors(
        gnm_np, vertex_colors_module.DEFAULT_COLOR
    )
  elif vertex_colors.shape[-1] == 4:
    vertex_colors = vertex_colors[..., :3]
  elif vertex_colors.shape[-1] != 3:
    raise ValueError(
        'Vertex colors must have 3 or 4 channels, got'
        f' {vertex_colors.shape[-1]}.'
    )

  colors_has_m_dim = (0, 0, *vertex_colors.shape)[-3] == vertices.shape[-3]
  if not colors_has_m_dim:
    # 'M' dimension is not present but is required. Inject the dimension, and
    # expand to match the size of vertices' M dimension.
    vertex_colors = vertex_colors[..., None, :, :]
    vertex_colors_shape = list(vertex_colors.shape)
    vertex_colors_shape[-3] = vertices.shape[-3]
    vertex_colors = np.broadcast_to(vertex_colors, vertex_colors_shape)

  # Define default camera params based on the first GNM in the 'M' dimension.
  vertices_for_cameras = vertices[..., 0, :, :]

  if world_to_camera is None:
    world_to_camera = get_look_at_world_to_camera(
        gnm_np,
        vertices_for_cameras,
    )

  if camera_to_image is None:
    camera_to_image = get_fill_factor_camera_to_image(
        gnm_np, vertices_for_cameras, image_size=image_size
    )

  if convert_cameras_to_opengl:
    # Convert from OpenCV to OpenGL convention.
    world_to_camera = camera_conversions.opencv_extrinsics_to_opengl(
        world_to_camera
    )
    camera_to_image = (
        camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
            camera_to_image,
            width=image_size[0],
            height=image_size[1],
            near=DEFAULT_NEAR,
            far=DEFAULT_FAR,
        )
    )

  vertex_normals = gnm_np.compute_vertex_normals(vertices)

  # Broadcast a background color to an image.
  if not isinstance(background_color, np.ndarray) or background_color.ndim == 1:
    background_color = np.broadcast_to(background_color, (height, width, 3))

  # Convert background color to float [0-1].
  if background_color.dtype == np.uint8:
    background_color = background_color.astype(np.float32) / 255.0

  texture_dict = load_texture(gnm_np, texture)
  textures = list(texture_dict.values())
  texture_keys = texture_dict.keys()
  if not set(texture_keys).issubset(gnm_np.mesh_component_names):
    missing_parts = set(texture_keys) - set(gnm_np.mesh_component_names)
    raise ValueError(
        f'Texture keys {missing_parts} are not GNM part names'
        f' {gnm_np.mesh_component_names}.'
    )

  # Find the maximum batch dimension that satisfies all batch-able arguments.
  try:
    batch_dims = get_batch_dim(
        (vertices, 3),
        (vertex_colors, 3),
        (world_to_camera, 2),
        (camera_to_image, 2),
        (background_color, 3),
        *[(t, 3) for t in textures],
    )
  except ValueError as e:
    raise ValueError(
        f' Batch dimensions incompatible: vertices {vertices.shape},'
        f' vertex_colors {vertex_colors.shape}, world_to_camera'
        f' {world_to_camera.shape}, camera_to_image {camera_to_image.shape},'
        f' background_color {background_color.shape}, texture'
        f' {[t.shape for t in textures]}.'
    ) from e

  def batchify(arr, non_batch_dims):
    """Broadcast to batch dimensions, and flatten the batch dimensions."""
    arr = np.broadcast_to(arr, (*batch_dims, *arr.shape[-non_batch_dims:]))
    return arr.reshape(int(np.prod(batch_dims)), *arr.shape[-non_batch_dims:])

  vertices = batchify(vertices, 3)
  vertex_normals = batchify(vertex_normals, 3)
  vertex_colors = batchify(vertex_colors, 3)
  world_to_camera = batchify(world_to_camera, 2)
  camera_to_image = batchify(camera_to_image, 2)
  background_color = batchify(background_color, 3)
  batched_textures = {part: batchify(x, 3) for part, x in texture_dict.items()}

  renders = backend_render_fn(
      vertices=vertices,
      triangles=triangle_dict,
      world_to_camera=world_to_camera,
      camera_to_image=camera_to_image,
      image_size=image_size,
      texture=batched_textures,
      vertex_colors=vertex_colors,
      multisample_antialiasing=multisample_antialiasing,
      vertex_uvs=gnm_np.vertex_uvs,
      vertex_normals=vertex_normals,
      background_color=background_color,
      alpha=alpha,
      include_shading=include_shading,
      verbose=verbose,
  )

  color = renders.reshape(*batch_dims, height, width, 3)
  return color


def project_points_for_gnm(
    gnm_np: gnm_numpy.GNM,
    points_world: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    world_to_camera: FloatArray | None = None,
    camera_to_image: FloatArray | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    multiple_gnms: bool = False,
    **kwargs,
) -> np.ndarray:
  """Projects world points under the same conditions as a render_gnm call.

  Intended for identifying the per-frame positions of 3D points (e.g. GNM
  joints) in the same reference space used by render_gnm.

  For a description of the other arguments, please see the docstring of
  `render_gnm`. Any shading related arguments are ignored.

  Args:
    gnm_np: The GNM model.
    points_world: The world-space points to project, (..., P, 3). Defaults to
      the vertex positions of the template GNM.
    vertices: The GNM vertices in world space, (..., V, 3). If not provided,
      will use the template vertices. Used for default camera setup.
    world_to_camera: The world-to-camera transformation, (..., 4, 4).
    camera_to_image: The camera-to-image transformation, (..., 4, 4).
    image_size: The width and height of the rendered image in pixels: (W, H).
    multiple_gnms: If True, vertices is expected to be shape (..., M, V, 3), and
      we render M GNMs per image. Default cameras will be set relative to the
      first GNM in the sequence.
    **kwargs: Any additional arguments expected for render_gnm (ignored).

  Returns:
    The projected points in image space, (..., P, 2).
  """

  del kwargs

  if points_world is None:
    points_world = gnm_np.template_vertex_positions
  points_world = points_world.astype(np.float32)

  if vertices is None:
    vertices = gnm_np.template_vertex_positions

  if not multiple_gnms:
    vertices = vertices[..., None, :, :]  # Inject 'M' dimension.
  elif (vertices_dim := vertices.ndim) < 3:
    raise ValueError(
        f'Called with {multiple_gnms=}, but vertices is only {vertices_dim}D.'
    )

  # Define default camera params based on the first GNM in the 'M' dimension.
  vertices_for_cameras = vertices[..., 0, :, :]

  if world_to_camera is None:
    world_to_camera = get_look_at_world_to_camera(
        gnm_np,
        vertices_for_cameras,
    )

  if camera_to_image is None:
    camera_to_image = get_fill_factor_camera_to_image(
        gnm_np, vertices_for_cameras, image_size=image_size
    )

  # Convert from OpenCV to OpenGL convention.
  world_to_camera = camera_conversions.opencv_extrinsics_to_opengl(
      world_to_camera
  )
  camera_to_image = (
      camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
          camera_to_image,
          width=image_size[0],
          height=image_size[1],
          near=DEFAULT_NEAR,
          far=DEFAULT_FAR,
      )
  )

  # Find the maximum batch dimension that satisfies all batch-able arguments.
  try:
    batch_dims = get_batch_dim(
        (points_world, 2),
        (vertices, 3),
        (world_to_camera, 2),
        (camera_to_image, 2),
    )
  except ValueError as e:
    raise ValueError(
        f' Batch dimensions incompatible: points_world {points_world.shape},'
        f' vertices {vertices.shape}, world_to_camera {world_to_camera.shape},'
        f' camera_to_image {camera_to_image.shape}.'
    ) from e

  def batchify(arr, non_batch_dims):
    """Broadcast to batch dimensions, and flatten the batch dimensions."""
    arr = np.broadcast_to(arr, (*batch_dims, *arr.shape[-non_batch_dims:]))
    return arr.reshape(int(np.prod(batch_dims)), *arr.shape[-non_batch_dims:])

  points_world = batchify(points_world, 2)
  world_to_camera = batchify(world_to_camera, 2)
  camera_to_image = batchify(camera_to_image, 2)

  # Perform projection.
  view_projection_matrix = camera_to_image @ world_to_camera

  homogenous_ones = np.ones(
      (*points_world.shape[:-1], 1), dtype=points_world.dtype
  )
  points_homogeneous = np.concatenate([points_world, homogenous_ones], axis=-1)
  points_clip_space = (
      view_projection_matrix[:, None, :, :] @ points_homogeneous[..., None]
  )
  points_clip_space = points_clip_space[..., 0]

  # Perform perspective division to get Normalized Device Coordinates (NDC).
  points_ndc = points_clip_space[..., :3] / points_clip_space[..., [3]]

  # Convert NDC to image space.
  # NDC range is [-1, 1]. Image space range is [0, image_size].
  width, height = image_size
  points_image_space = (
      (points_ndc[..., :2] + 1.0) * 0.5 * np.array([width, height])
  )

  # Flip +Y up renders to +Y down.
  points_image_space[..., 1] = height - points_image_space[..., 1]

  return points_image_space.reshape(*batch_dims, *points_image_space.shape[-2:])


def get_look_at_world_to_camera(
    gnm_np: gnm_numpy.GNM,
    vertices_world: np.ndarray,
    azimuthal_angle: np.ndarray | float = 0.0,
    polar_angle: np.ndarray | float = 0.0,
    camera_distance: np.ndarray | float | None = DEFAULT_CAMERA_DISTANCE,
    share_camera: np.ndarray | bool = True,
    y_up: np.ndarray | bool = False,
    look_at_vertex_groups: Sequence[str] = ('hockey_mask',),
    left_vertex_groups: Sequence[str] = ('ears', '&left'),
    right_vertex_groups: Sequence[str] = ('ears', '&right'),
    forward_vertex_groups: Sequence[str] = ('nose_region',),
) -> np.ndarray:
  """Compute world-to-camera matrices for a 'look-at' transform.

  Returns matrices in OpenCV convention.

  Args:
    gnm_np: The GNM model.
    vertices_world: The GNM vertices in world space, (..., V, 3).
    azimuthal_angle: The azimuthal angle of the camera in degrees, (..., 1).
    polar_angle: The polar angle of the camera in degrees, (..., 1).
    camera_distance: The distance of the camera from the head, (..., 1).
    share_camera: Whether to use the first frame's vertices only for camera
      generation, (..., 1). It is assumed that the first dimension of vertices
      is the time dimension.
    y_up: Whether to use the Y-up convention for the world space, (..., 1).
    look_at_vertex_groups: The vertex groups to look at.
    left_vertex_groups: The vertex groups to use for the left axis.
    right_vertex_groups: The vertex groups to use for the right axis.
    forward_vertex_groups: The vertex groups to use for the forward axis.

  Returns:
    The world-to-camera matrices, (..., 4, 4).
  """

  batch_dims = vertices_world.shape[:-2]

  if not batch_dims:
    # If there is no batch dimension, we don't need to share the camera.
    share_camera = False

  if camera_distance is None:
    camera_distance = DEFAULT_CAMERA_DISTANCE

  azimuthal_angle = _adjust_scalar_shape(azimuthal_angle, batch_dims)
  polar_angle = _adjust_scalar_shape(polar_angle, batch_dims)
  camera_distance = _adjust_scalar_shape(camera_distance, batch_dims)
  share_camera = _adjust_scalar_shape(share_camera, batch_dims)
  y_up = _adjust_scalar_shape(y_up, batch_dims)

  first_frame_vertices = np.broadcast_to(
      vertices_world[:1], vertices_world.shape
  )

  vertices_for_camera = np.where(
      share_camera[..., None], first_frame_vertices, vertices_world
  )

  gnm_axes = _get_gnm_axes(
      vertices_for_camera,
      left_vertex_groups=left_vertex_groups,
      right_vertex_groups=right_vertex_groups,
      forward_vertex_groups=forward_vertex_groups,
      gnm_np=gnm_np,
  )
  camera_target = _vertex_group_mean(
      vertices_for_camera, look_at_vertex_groups, gnm_np
  )
  camera_location = camera_target + _get_camera_offset(
      gnm_axes,
      camera_distance,
      azimuthal_angle,
      polar_angle,
  )

  y_up_vector = np.array([0.0, 1.0, 0.0])
  up_vector = np.where(y_up, y_up_vector, gnm_axes[1])
  world_to_camera = right_handed_look_at(
      camera_location, camera_target, up_vector
  )

  world_to_camera = camera_conversions.opengl_extrinsics_to_opencv(
      world_to_camera
  )

  return world_to_camera


def get_fill_factor_camera_to_image(
    gnm_np: gnm_numpy.GNM,
    vertices: np.ndarray,
    target_fill_factor: np.ndarray | float = DEFAULT_TARGET_FILL_FACTOR,
    camera_distance: np.ndarray | float = DEFAULT_CAMERA_DISTANCE,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    vertex_group_name: str = 'hockey_mask',
) -> np.ndarray:
  """Compute the camera-to-image matrix for a 'fill factor' transform.

  The camera intrinsics are determined to fill the width of the given vertex
  group with the image. Returns matrices in OpenCV convention.

  Args:
    gnm_np: The GNM model.
    vertices: The GNM vertices in world space, (..., V, 3).
    target_fill_factor: The desired fill factor of the GNM mesh in the image,
      (..., 1).
    camera_distance: The distance of the camera from the head, (..., 1).
    image_size: The width and height of the rendered image in pixels: (W, H).
    vertex_group_name: The vertex group to use for the projection.

  Returns:
    The camera-to-image matrix, (..., 4, 4).
  """
  batch_dims = vertices.shape[:-2]
  target_fill_factor = _adjust_scalar_shape(target_fill_factor, batch_dims)
  camera_distance = _adjust_scalar_shape(camera_distance, batch_dims)

  width, height = image_size

  # Get the maximum width of the vertex group.
  vertex_group_indices = gnm_np.vertex_group_indices(vertex_group_name)
  vertex_group_points = vertices[..., vertex_group_indices, :]
  mask_width = np.ptp(vertex_group_points, axis=-2)[..., :1]
  dimension = mask_width / target_fill_factor

  vertical_field_of_view = np.atan2(dimension / 2.0, camera_distance) * 2
  aspect_ratio = _adjust_scalar_shape(width / height, batch_dims)

  near = _adjust_scalar_shape(DEFAULT_NEAR, batch_dims)
  far = _adjust_scalar_shape(DEFAULT_FAR, batch_dims)

  camera_to_image = _right_handed_perspective(
      vertical_field_of_view=vertical_field_of_view,
      aspect_ratio=aspect_ratio,
      near=near,
      far=far,
  )

  camera_to_image = camera_conversions.opengl_intrinsics_to_opencv_matrix(
      camera_to_image, width=width, height=height
  )

  return camera_to_image


def get_spin_world_to_camera(
    gnm_np: gnm_numpy.GNM,
    vertices: np.ndarray,
    has_time_dimension: bool = False,
    spin_period: int = 60,
    spin_azimuth_limit: float = 20.0,
    spin_polar_limit: float = 5.0,
    num_frames: int | None = None,
    **kwargs,
) -> np.ndarray:
  """Compute world-to-camera matrices for a 'spin' transform.

  Args:
    gnm_np: The GNM model.
    vertices: The GNM vertices in world space, (..., V, 3).
    has_time_dimension: Whether the first dimension of the vertices is time. If
      false, will broadcast vertices over the spin period.
    spin_period: The number of frames in a full spin.
    spin_azimuth_limit: The maximum azimuthal angle of the camera in degrees.
    spin_polar_limit: The maximum polar angle of the camera in degrees.
    num_frames: Optional number of frames to generate (defaults to spin_period).
    **kwargs: Additional arguments to pass to get_look_at_world_to_camera.

  Returns:
    The world-to-camera matrices, (spin_period, ..., 4, 4) if has_time_dimension
    is False, otherwise (..., 4, 4).
  """

  if num_frames is not None:
    if not has_time_dimension and (
        vertices.ndim < 3 or vertices.shape[0] != num_frames
    ):
      vertices = np.broadcast_to(vertices, (num_frames, *vertices.shape))
  elif has_time_dimension:
    num_frames = vertices.shape[0]
  else:
    num_frames = spin_period
    vertices = np.broadcast_to(vertices, (num_frames, *vertices.shape))

  batch_dims = vertices.shape[:-2]

  # Spin frequency is determined by spin_period.
  max_angle = np.pi * 2 * (num_frames / spin_period)
  angles = np.linspace(-0, max_angle, num_frames)

  azimuthal_angle = np.sin(angles)[..., None] * spin_azimuth_limit
  polar_angle = np.cos(angles)[..., None] * spin_polar_limit

  # Broadcast azimuthal and polar angles to the batch dimensions.
  azimuthal_angle = np.broadcast_to(azimuthal_angle, (*batch_dims, 1))
  polar_angle = np.broadcast_to(polar_angle, (*batch_dims, 1))

  return get_look_at_world_to_camera(
      gnm_np,
      vertices,
      azimuthal_angle=azimuthal_angle,
      polar_angle=polar_angle,
      **kwargs,
  )


def _get_gnm_axes(
    vertices: np.ndarray,
    left_vertex_groups: Sequence[str],
    right_vertex_groups: Sequence[str],
    forward_vertex_groups: Sequence[str],
    gnm_np: gnm_numpy.GNM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Determines the right, up, and forwards direction vectors of GNM.

  Args:
    vertices: The GNM vertices in world space, (..., V, 3).
    left_vertex_groups: The vertex groups to use for the left axis.
    right_vertex_groups: The vertex groups to use for the right axis.
    forward_vertex_groups: The vertex groups to use for the forward axis.
    gnm_np: The GNM model.

  Returns:
    A tuple of right, up, and forwards direction vectors, each (..., 3).
  """

  left = _vertex_group_mean(vertices, left_vertex_groups, gnm_np)
  right = _vertex_group_mean(vertices, right_vertex_groups, gnm_np)
  forward_point = _vertex_group_mean(vertices, forward_vertex_groups, gnm_np)
  back_point = (left + right) / 2.0
  right = right - left
  right = right / np.linalg.norm(right, axis=-1, keepdims=True)
  forwards = forward_point - back_point
  forwards = forwards / np.linalg.norm(forwards, axis=-1, keepdims=True)
  up = np.linalg.cross(right, forwards)
  return right, up, forwards


def right_handed_look_at(
    camera_position: np.ndarray, look_at: np.ndarray, up_vector: np.ndarray
) -> np.ndarray:
  """Builds a right handed look at view matrix.

  Args:
    camera_position: The position of the camera, (..., 3).
    look_at: The position of the target, (..., 3).
    up_vector: The up vector of the camera, (..., 3).

  Returns:
    The right handed look at view matrix, (..., 4, 4).
  """

  z_axis = look_at - camera_position
  z_axis /= np.linalg.norm(z_axis, axis=-1, keepdims=True)

  horizontal_axis = np.cross(z_axis, up_vector, axis=-1)
  horizontal_axis /= np.linalg.norm(horizontal_axis, axis=-1, keepdims=True)

  vertical_axis = np.cross(horizontal_axis, z_axis, axis=-1)

  def _dot_last_axis(arr1, arr2):
    return np.einsum('...i,...i->...', arr1, arr2)[..., None]

  batch_shape = horizontal_axis.shape[:-1]
  zeros = np.zeros((*batch_shape, 3), dtype=horizontal_axis.dtype)
  ones = np.ones((*batch_shape, 1), dtype=horizontal_axis.dtype)

  tx = -_dot_last_axis(horizontal_axis, camera_position)
  ty = -_dot_last_axis(vertical_axis, camera_position)
  tz = _dot_last_axis(z_axis, camera_position)

  row1 = np.concatenate([horizontal_axis, tx], axis=-1)
  row2 = np.concatenate([vertical_axis, ty], axis=-1)
  row3 = np.concatenate([-z_axis, tz], axis=-1)
  row4 = np.concatenate([zeros, ones], axis=-1)
  matrix = np.stack([row1, row2, row3, row4], axis=-2)
  return matrix


def _get_camera_offset(
    head_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    camera_distance: np.ndarray,
    azimuthal_angle: np.ndarray,
    polar_angle: np.ndarray,
) -> np.ndarray:
  """Determines how to place the camera so it points at the face.

  Args:
    head_axes: Three-tuple representing normalised right, up, and forwards face
      direction vectors, each (..., 3).
    camera_distance: Camera distance from the face (meters), (..., 1).
    azimuthal_angle: Places camera to the left or right, in degrees, (..., 1).
    polar_angle: Places camera above or below the head, in degrees, (..., 1).

  Returns:
    A batch of offset vectors to apply to camera targets (..., 3).
  """
  right, up, forwards = head_axes
  theta = np.deg2rad(90 + azimuthal_angle)
  phi = np.deg2rad(90 + polar_angle)
  x = camera_distance * np.sin(phi) * np.cos(theta) * right
  y = camera_distance * np.sin(phi) * np.sin(theta) * forwards
  z = camera_distance * np.cos(phi) * up
  return x + y + z


def _right_handed_perspective(
    vertical_field_of_view: np.ndarray,
    aspect_ratio: np.ndarray,
    near: np.ndarray,
    far: np.ndarray,
) -> np.ndarray:
  """Builds a right handed perspective projection matrix.

  Similar to tensorflow_graphics.rendering.camera.perspective.right_handed.

  Args:
    vertical_field_of_view: The vertical field of view in radians, (..., 1).
    aspect_ratio: The aspect ratio of the image, (..., 1).
    near: The near clipping plane distance, (..., 1).
    far: The far clipping plane distance, (..., 1).

  Returns:
    The perspective projection matrix, (..., 4, 4) in OpenGL convention.
  """

  itan_half_vertical_field_of_view = 1.0 / np.tan(vertical_field_of_view * 0.5)
  zero = np.zeros_like(itan_half_vertical_field_of_view)
  one = np.ones_like(itan_half_vertical_field_of_view)
  near_minus_far = near - far

  row1 = np.concatenate(
      (itan_half_vertical_field_of_view / aspect_ratio, zero, zero, zero),
      axis=-1,
  )
  row2 = np.concatenate(
      (zero, itan_half_vertical_field_of_view, zero, zero), axis=-1
  )
  row3 = np.concatenate(
      (
          zero,
          zero,
          (far + near) / near_minus_far,
          2.0 * far * near / near_minus_far,
      ),
      axis=-1,
  )
  row4 = np.concatenate((zero, zero, -one, zero), axis=-1)

  matrix = np.stack((row1, row2, row3, row4), axis=-2)
  return matrix


@functools.cache
def load_edgeflow_texture(texture_path: epath.Path | None) -> FloatArray | None:
  """Loads the edgeflow texture from a given path.

  Args:
    texture_path: The path to the edgeflow texture.

  Returns:
    The edgeflow texture as a float32 array, or None if texture_path is None.
  """
  if texture_path is None:
    return None
  with texture_path.open('rb') as f:
    image = imageio.imread(f).astype(np.float32)
  image = (image / np.iinfo(np.uint8).max) * 0.5 + 0.5
  image.flags.writeable = False
  return image


def _vertex_group_mean(
    vertices: np.ndarray,
    group_names: Sequence[str],
    gnm_np: gnm_numpy.GNM,
):
  """Gets the average point of GNM vertex groups.

  Args:
    vertices: The GNM vertices in world space, (..., V, 3).
    group_names: The names of the vertex groups.
    gnm_np: The GNM model.

  Returns:
    The average point of the vertex groups, (..., 3).
  """
  indices = gnm_np.vertex_group_indices(*group_names)
  return vertices[..., indices, :].mean(axis=-2)


def load_texture(
    gnm_np: gnm_numpy.GNM,
    texture: Texture = DEFAULT_TEXTURE,
) -> dict[str, npt.NDArray[np.uint8]]:
  """Loads the texture as a (potentially batched) image.

  Args:
    gnm_np: The GNM model.
    texture: The texture to load. If DEFAULT_TEXTURE, will load the edgeflow
      texture. If an ndarray, will use the given texture for skin. If a dict,
      will use the given texture for each part. If None, will use a white
      (plain) texture.

  Returns:
    The texture image, [0-255] uint8, (..., H, W, 3).
  """
  texture_dict = {}
  if texture is DEFAULT_TEXTURE:
    edgeflow_path = EDGEFLOW_TEXTURE_BY_BODY_PART.get(gnm_np.body_part)
    if edgeflow_path is not None:
      edgeflow = load_edgeflow_texture(edgeflow_path)
      if edgeflow is not None:
        texture_dict['skin'] = edgeflow[..., None]
  elif isinstance(texture, np.ndarray):
    texture_dict['skin'] = texture
  elif isinstance(texture, dict):
    texture_dict = texture

  # Fill remaining parts with white texture.
  for component in gnm_np.mesh_component_names:
    if component not in texture_dict:
      texture_dict[component] = np.ones((64, 64, 1)).astype(np.float32)

  def _to_3channel_uint8(texture_image: np.ndarray) -> npt.NDArray[np.uint8]:
    if texture_image.dtype != np.uint8:
      texture_image = (texture_image * 255.0).astype(np.uint8)
    if texture_image.shape[-1] == 1:
      texture_image = np.repeat(texture_image, 3, axis=-1)
    return texture_image

  texture_dict = {
      part: _to_3channel_uint8(texture_dict[part]) for part in texture_dict
  }

  return texture_dict


def _adjust_scalar_shape(
    array: np.ndarray | float, batch_dims: Sequence[int]
) -> np.ndarray:
  """Adjusts the shape of a scalar to match the batch dimensions.

  Args:
    array: The array to adjust.
    batch_dims: The batch dimensions to match.

  Returns:
    The adjusted array, (..., 1).
  """

  # If the array is a scalar or 1D array, add a final dimension of size 1.
  array = np.array(array)
  if array.ndim <= 1:
    array = array[..., None]

  return np.broadcast_to(array, (*batch_dims, 1))


def get_batch_dim(*arrays: tuple[np.ndarray | None, int]) -> tuple[int, ...]:
  """Finds the largest batch dimension that all arrays can be broadcast to.

  Requires that all arrays have batch dimensions that can be broadcast together.

  e.g. If given an array [B, C, 4] and an array [A, B, C, 3], return (A, B, C).

  Args:
    *arrays: The arrays to expand. Tuples of (array, non_batch_dims), where
      non_batch_dims is the number of rightmost dimensions of the array that are
      not batch dimensions. If None, will be ignored.

  Returns:
    The batch dimensions that all arrays can be safely broadcast to.
  Raises:
    ValueError: If the arrays cannot be broadcast together.
  """
  batch_dims: list[tuple[int, ...]] = []
  for array, non_batch_dims in arrays:
    if array is None:
      continue
    batch_dims.append(array.shape[:-non_batch_dims])

  if not batch_dims:
    return ()

  return np.broadcast_shapes(*batch_dims)
