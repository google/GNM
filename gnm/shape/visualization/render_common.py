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
from collections.abc import Mapping, Sequence
import dataclasses
import functools
from typing import Any, ClassVar, Protocol, Self, TypeAlias, overload
import warnings

from etils import epath
from gnm.shape import gnm_numpy
from gnm.shape import gnm_typing
from gnm.shape import gnm_udim
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import udim
from gnm.shape.visualization import vertex_colors as vertex_colors_module
import imageio
import immutabledict
import numpy as np
import numpy.typing as npt
import tqdm.autonotebook as tqdm

enpt = gnm_typing.enpt

FloatArray = npt.NDArray[np.floating]
UInt8Array = npt.NDArray[np.uint8]
ColorOrImage = UInt8Array | FloatArray | Sequence[float] | float

_pkg = __package__ or 'gnm.shape.visualization'
_TEXTURES_DIR = epath.resource_path(_pkg).parent / 'data' / 'textures'
EDGEFLOW_TEXTURE_BY_BODY_PART = immutabledict.immutabledict({
    gnm_numpy.GNMBodyPart.HEAD: _TEXTURES_DIR / 'edgeflow_bw_4k.png',
})

# The UDIM tile a texture given as a bare array (rather than a dict) applies to.
SKIN_UDIM_TILE = gnm_udim.UdimTile.SKIN

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
# Textures are keyed by UDIM tile (`gnm_udim.UdimTile` or its tile number).
# Keying them by GNM part name is deprecated; such a texture is applied to every
# tile of the part. One `Mapping` per key type, as `Mapping` is invariant in its
# key: a `dict[int, ...]` is not a `Mapping[int | str, ...]`.
Texture: TypeAlias = (
    FloatArray
    | UInt8Array
    | DefaultTexture
    | Mapping[gnm_udim.UdimTile, FloatArray | UInt8Array]
    | Mapping[int, FloatArray | UInt8Array]
    | Mapping[np.integer, FloatArray | UInt8Array]
    | Mapping[str, FloatArray | UInt8Array]
    | None
)
TriangleGroups: TypeAlias = (
    str
    | Sequence[str]
    | enpt.IntArray['F_sub']  # pyrefly: ignore[unknown-name]
)


class BackendRenderFn(Protocol):
  """Protocol for backend rendering functions.

  Backend render functions accept batched scene parameters and rasterize or
  raytrace the scene, returning rendered images of shape (N, H, W, 3).

  Texture coordinates are given per triangle corner in UDIM layout, so the
  integer part of a corner's coordinates identifies its texture in `texture`
  (see `udim.split_triangles_by_tile`). They are per corner rather than per
  vertex because a vertex shared by triangles of different tiles has different
  texture coordinates in each. Tiles absent from `texture` are rendered with
  the backend's default appearance (e.g. plain white).
  """

  def __call__(
      self,
      *,
      vertices: FloatArray,
      triangles: npt.NDArray[np.integer],
      world_to_camera: FloatArray,
      camera_to_image: FloatArray,
      vertex_normals: FloatArray,
      triangle_uvs: FloatArray,
      vertex_colors: FloatArray,
      image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
      texture: (
          Mapping[int, FloatArray | npt.NDArray[np.uint8]] | None
      ) = None,
      multisample_antialiasing: int = 1,
      background_color: FloatArray | None = None,
      alpha: float = 1.0,
      include_shading: bool = True,
      verbose: bool = False,
  ) -> FloatArray:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class MeshRenderOutput:
  """Output of the stateful GNM 3D mesh renderer.

  Attributes:
    image: Rendered RGB image as a uint8 NumPy array of shape (..., H, W, 3).
    alpha: Rendered coverage/opacity mask in [0.0, 1.0] as a float32 NumPy array
      of shape (..., H, W, 1) after multisample anti-aliasing.
  """

  image: enpt.ui8['... H W 3']
  alpha: enpt.FloatArray['... H W 1']

  def composite(
      self,
      background: ColorOrImage = DEFAULT_BACKGROUND_COLOR,
      *,
      alpha: float = 1.0,
      dtype: npt.DTypeLike = np.uint8,
  ) -> enpt.ui8['... H W 3'] | enpt.FloatArray['... H W 3']:
    """Composites the rendered mesh foreground over a background color or image.

    Args:
      background: Solid grayscale float, RGB float tuple, or background image
        array (float32 in [0.0, 1.0] or uint8 in [0, 255]) of shape (..., H, W,
        3).
      alpha: Global opacity multiplier in [0.0, 1.0] applied to `self.alpha`.
      dtype: Output NumPy dtype (`np.uint8` for [0, 255] or floating-point such
        as `np.float32` for [0.0, 1.0]). Defaults to `np.uint8`.

    Returns:
      Composited RGB image array of shape (..., H, W, 3) with the requested
      `dtype`.

    Raises:
      ValueError: If `background` does not have shape `()`, `(3,)`, or
        `(..., H, W, 3)`, if batch dimensions of `self.image` and
        `background` are incompatible, or if `dtype` is neither `np.uint8`
        nor floating-point.
    """
    dtype = np.dtype(dtype)
    if dtype != np.uint8 and not np.issubdtype(dtype, np.floating):
      raise ValueError(
          f'Unsupported output dtype {dtype}; expected np.uint8 or'
          ' floating-point.'
      )

    foreground = np.asarray(self.image, dtype=np.float32) / 255.0
    height, width = foreground.shape[-3:-1]
    if isinstance(background, np.ndarray) and background.dtype == np.uint8:
      background = background.astype(np.float32) / 255.0
    else:
      background = np.asarray(background, dtype=np.float32)

    if background.ndim == 0:
      background = np.broadcast_to(background, (height, width, 3))
    elif background.ndim == 1:
      if background.shape != (3,):
        raise ValueError(
            'Expected 1D background color of shape (3,), got'
            f' {background.shape}.'
        )
      background = np.broadcast_to(background, (height, width, 3))
    else:
      if background.ndim < 3 or background.shape[-3:] != (height, width, 3):
        raise ValueError(
            'Expected background image trailing dimensions'
            f' {(height, width, 3)}, got shape {background.shape}.'
        )

    try:
      batch_dims = get_batch_dim((foreground, 3), (background, 3))
    except ValueError as err:
      raise ValueError(
          f'Batch dimensions incompatible: image {foreground.shape},'
          f' background {background.shape}.'
      ) from err

    alpha_mask = np.broadcast_to(
        float(alpha) * np.asarray(self.alpha, dtype=np.float32),
        (*batch_dims, height, width, 1),
    )
    foreground = np.broadcast_to(foreground, (*batch_dims, height, width, 3))
    background = np.broadcast_to(background, (*batch_dims, height, width, 3))
    composited = foreground * alpha_mask + background * (1.0 - alpha_mask)
    if dtype == np.uint8:
      return np.clip(composited * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.asarray(composited, dtype=dtype)


def resolve_gnm_triangles(
    gnm_np: gnm_numpy.GNM,
    triangles: TriangleGroups = '~eye_exteriors',
) -> tuple[enpt.IntArray['F 3'], enpt.FloatArray['F 3 2']]:
  """Resolves GNM triangle groups or index arrays to faces and UDIM corner UVs.

  Args:
    gnm_np: NumPy GNM shape model.
    triangles: Vertex group expression string, sequence of group strings, or 1D
      face index array.

  Returns:
    Tuple of `(selected_triangles, selected_triangle_uvs)` where
    `selected_triangles` is an (F, 3) int32 face index array and
    `selected_triangle_uvs` is an (F, 3, 2) float32 UDIM corner UV array.
  """
  if isinstance(triangles, np.ndarray):
    triangle_indices = triangles
  else:
    group_names = (
        (triangles,) if isinstance(triangles, str) else tuple(triangles)
    )
    triangle_indices = gnm_np.triangle_indices_for_group(*group_names)

  return (
      np.asarray(gnm_np.triangles[triangle_indices], dtype=np.int32),
      np.asarray(gnm_np.triangle_uvs_udim[triangle_indices], dtype=np.float32),
  )


def resolve_gnm_camera_and_vertices(
    gnm_np: gnm_numpy.GNM,
    vertices: enpt.FloatArray['... V 3'] | None = None,
    world_to_camera: enpt.FloatArray['... 4 4'] | None = None,
    camera_to_image: (
        enpt.FloatArray['... 4 4'] | enpt.FloatArray['... 3 3'] | None
    ) = None,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    multiple_gnms: bool = False,
) -> tuple[
    enpt.FloatArray['... V 3'],
    enpt.FloatArray['... 4 4'],
    enpt.FloatArray['... 4 4'] | enpt.FloatArray['... 3 3'],
]:
  """Resolves default GNM template vertices and OpenCV camera matrices.

  Args:
    gnm_np: NumPy GNM shape model.
    vertices: Optional posed vertices of shape (..., [M], V, 3). Defaults to
      `gnm_np.template_vertex_positions`.
    world_to_camera: Optional OpenCV world-to-camera matrix of shape (..., 4,
      4). Defaults to `get_look_at_world_to_camera`.
    camera_to_image: Optional OpenCV camera-to-image matrix of shape (..., 4, 4)
      or (..., 3, 3). Defaults to `get_fill_factor_camera_to_image`.
    image_size: Output image dimensions as (width, height).
    multiple_gnms: Whether `vertices` includes a leading mesh axis M at
      dimension -3.

  Returns:
    Tuple of resolved `(vertices, world_to_camera, camera_to_image)`.

  Raises:
    ValueError: If `multiple_gnms=True` and `vertices` has fewer than 3
      dimensions.
  """
  if vertices is None:
    vertices = gnm_np.template_vertex_positions
  vertices = np.asarray(vertices, dtype=np.float32)

  if multiple_gnms:
    if (vertices_dim := vertices.ndim) < 3:
      raise ValueError(
          f'Called with {multiple_gnms=}, but vertices is only {vertices_dim}D.'
      )
    vertices_for_cameras = vertices[..., 0, :, :]
  else:
    vertices_for_cameras = vertices

  if world_to_camera is None:
    world_to_camera = get_look_at_world_to_camera(gnm_np, vertices_for_cameras)
  world_to_camera = np.asarray(world_to_camera, dtype=np.float32)

  if camera_to_image is None:
    camera_to_image = get_fill_factor_camera_to_image(
        gnm_np, vertices_for_cameras, image_size=image_size
    )
  camera_to_image = np.asarray(camera_to_image, dtype=np.float32)

  return vertices, world_to_camera, camera_to_image


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
      are not UDIM tile numbers, or if batch dimensions are incompatible.
  """
  width, height = image_size

  triangle_indices = (
      gnm_np.triangle_indices_for_group(triangles)
      if isinstance(triangles, str)
      else triangles
  )
  triangle_uvs = gnm_np.triangle_uvs_udim[triangle_indices]
  selected_triangles = gnm_np.triangles[triangle_indices]

  if vertices is None:
    vertices = gnm_np.template_vertex_positions

  if not multiple_gnms:
    vertices = vertices[..., np.newaxis, :, :]  # Inject 'M' dimension.
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
    vertex_colors = vertex_colors[..., np.newaxis, :, :]
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
  else:
    background_color = background_color.astype(np.float32, copy=False)

  texture_dict = load_texture(gnm_np, texture)
  textures = list(texture_dict.values())

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

  def batchify[ScalarT: np.generic](
      arr: npt.NDArray[ScalarT], non_batch_dims: int
  ) -> npt.NDArray[ScalarT]:
    """Broadcasts to batch dimensions, and flattens the batch dimensions."""
    arr = np.broadcast_to(arr, (*batch_dims, *arr.shape[-non_batch_dims:]))
    return arr.reshape(int(np.prod(batch_dims)), *arr.shape[-non_batch_dims:])

  vertices = batchify(vertices, 3)
  vertex_normals = batchify(vertex_normals, 3)
  vertex_colors = batchify(vertex_colors, 3)
  world_to_camera = batchify(world_to_camera, 2)
  camera_to_image = batchify(camera_to_image, 2)
  background_color = batchify(background_color, 3)
  batched_textures = {tile: batchify(x, 3) for tile, x in texture_dict.items()}

  renders = backend_render_fn(
      vertices=vertices,
      triangles=selected_triangles,
      world_to_camera=world_to_camera,
      camera_to_image=camera_to_image,
      image_size=image_size,
      texture=batched_textures,
      vertex_colors=vertex_colors,
      multisample_antialiasing=multisample_antialiasing,
      triangle_uvs=triangle_uvs,
      vertex_normals=vertex_normals,
      background_color=background_color,
      alpha=alpha,
      include_shading=include_shading,
      verbose=verbose,
  )

  color = renders.reshape(*batch_dims, height, width, 3)
  return color


@dataclasses.dataclass(kw_only=True)
class BaseGNMMeshRenderer(abc.ABC):
  """Abstract base class for stateful 3D mesh renderers.

  Caches static mesh topology, UDIM corner UV coordinates, vertex colors, and
  loaded UDIM tile textures across frames and repeated render calls.

  Attributes:
    triangles: Active triangle face index array of shape (F, 3).
    triangle_uvs: Per-corner UV texture coordinates in UDIM layout of shape (F,
      3, 2).
    vertex_colors: Per-vertex RGB colors of shape (..., V, 3).
    texture: Optional per-UDIM-tile texture mapping, single array for
      `SKIN_UDIM_TILE`, or None to use plain white textures.
    multisample_antialiasing: Supersampling factor for anti-aliasing. Defaults
      to 1 (native resolution).
    include_shading: Whether to include directional lighting/shading. Defaults
      to True.
    verbose: Default flag controlling whether to display a progress bar.
  """

  _progress_bar_desc: ClassVar[str] = 'Rendering'

  triangles: enpt.IntArray['F 3']
  triangle_uvs: enpt.FloatArray['F 3 2']
  vertex_colors: enpt.FloatArray['... V 3']
  texture: Texture = None
  multisample_antialiasing: int = 1
  include_shading: bool = True
  verbose: bool = False

  _tile_geometry: dict[int, udim.TileGeometry] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _active_vertex_colors: enpt.FloatArray['... V 3'] = dataclasses.field(
      default_factory=lambda: np.empty((0, 3), dtype=np.float32),
      init=False,
      repr=False,
      compare=False,
  )
  _raw_textures: dict[int, enpt.ui8['... H_texture W_texture 3']] = (
      dataclasses.field(
          default_factory=dict, init=False, repr=False, compare=False
      )
  )
  _active_textures: dict[int, enpt.ui8['H_texture W_texture 3']] = (
      dataclasses.field(
          default_factory=dict, init=False, repr=False, compare=False
      )
  )
  _cached_world_to_camera: enpt.FloatArray['4 4'] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _cached_camera_to_image: (
      enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'] | None
  ) = dataclasses.field(default=None, init=False, repr=False, compare=False)
  _cached_image_size: tuple[int, int] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )

  @classmethod
  def from_gnm(
      cls,
      gnm_np: gnm_numpy.GNM,
      *,
      triangles: TriangleGroups = '~eye_exteriors',
      texture: Texture = DEFAULT_TEXTURE,
      multisample_antialiasing: int = 1,
      vertex_colors: enpt.FloatArray['... V 3'] | None = None,
      include_shading: bool = True,
      verbose: bool = False,
  ) -> Self:
    """Instantiates a mesh renderer from a NumPy GNM shape model.

    Args:
      gnm_np: NumPy GNM shape model.
      triangles: Vertex group expression string, sequence of group strings, or
        1D face index array.
      texture: Optional per-UDIM-tile texture mapping, single array for
        `SKIN_UDIM_TILE`, `DEFAULT_TEXTURE` for the default edgeflow texture, or
        None for plain white textures.
      multisample_antialiasing: Supersampling factor for anti-aliasing. Defaults
        to 1 (native resolution).
      vertex_colors: Optional per-vertex RGB colors of shape (..., V, 3).
        Defaults to standard GNM per-component vertex colors.
      include_shading: Whether to include directional lighting/shading. Defaults
        to True.
      verbose: Default flag controlling whether to display a progress bar.

    Returns:
      A new renderer instance configured with the GNM model's mesh topology,
      UDIM corner UV coordinates, vertex colors, and textures.
    """
    selected_triangles, selected_triangle_uvs = resolve_gnm_triangles(
        gnm_np, triangles
    )
    if vertex_colors is None:
      vertex_colors = vertex_colors_module.get_vertex_colors(
          gnm_np, vertex_colors_module.DEFAULT_COLOR
      )
    return cls(
        triangles=selected_triangles,
        triangle_uvs=selected_triangle_uvs,
        vertex_colors=np.asarray(vertex_colors, dtype=np.float32),
        texture=load_texture(gnm_np, texture),
        multisample_antialiasing=multisample_antialiasing,
        include_shading=include_shading,
        verbose=verbose,
    )

  def __post_init__(self) -> None:
    triangles = np.asarray(self.triangles, dtype=np.int32)
    triangle_uvs = np.asarray(self.triangle_uvs, dtype=np.float32)
    self.triangles = triangles
    self.triangle_uvs = triangle_uvs

    tile_geometry = udim.build_tile_geometry(triangles, triangle_uvs)
    self._tile_geometry = tile_geometry

    normalized_colors = np.asarray(self.vertex_colors, dtype=np.float32)
    self.vertex_colors = normalized_colors
    self._active_vertex_colors = normalized_colors

    loaded_textures = _normalize_textures(
        self.texture, tuple(tile_geometry.keys())
    )
    self._raw_textures = loaded_textures
    self._active_textures = {
        tile: (
            raw_texture
            if raw_texture.ndim == 3
            else raw_texture.reshape(-1, *raw_texture.shape[-3:])[0]
        )
        for tile, raw_texture in loaded_textures.items()
    }

  def update_triangles(
      self,
      triangles: enpt.IntArray['F 3'],
      triangle_uvs: enpt.FloatArray['F 3 2'] | None = None,
  ) -> None:
    """Updates active mesh triangles and UVs and syncs backend state."""
    new_triangles = np.asarray(triangles, dtype=np.int32)
    new_triangle_uvs = (
        self.triangle_uvs
        if triangle_uvs is None
        else np.asarray(triangle_uvs, dtype=np.float32)
    )

    if np.array_equal(self.triangles, new_triangles) and np.array_equal(
        self.triangle_uvs, new_triangle_uvs
    ):
      return

    tile_geometry = udim.build_tile_geometry(new_triangles, new_triangle_uvs)
    self.triangles = new_triangles
    self.triangle_uvs = new_triangle_uvs
    self._tile_geometry = tile_geometry

    for tile in tile_geometry:
      if tile not in self._raw_textures:
        self._raw_textures[tile] = np.full((1, 1, 3), 255, dtype=np.uint8)
        self._active_textures[tile] = self._raw_textures[tile]

    self._on_triangles_updated()

  @abc.abstractmethod
  def update_vertices(
      self,
      vertices: enpt.FloatArray['... V 3'],
      vertex_normals: enpt.FloatArray['... V 3'] | None = None,
      *,
      max_texture_dim: int = 1024,
  ) -> None:
    """Updates scene mesh vertices and normals in-place."""

  @abc.abstractmethod
  def update_camera(
      self,
      world_to_camera: enpt.FloatArray['4 4'],
      camera_to_image: enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'],
      image_size: tuple[int, int],
  ) -> None:
    """Updates camera pose and projection matrices in OpenCV convention."""

  def render(
      self,
      vertices: enpt.FloatArray['... V 3'],
      world_to_camera: enpt.FloatArray['... 4 4'],
      camera_to_image: enpt.FloatArray['... 4 4'] | enpt.FloatArray['... 3 3'],
      *,
      image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
      triangles: enpt.IntArray['F 3'] | None = None,
      triangle_uvs: enpt.FloatArray['F 3 2'] | None = None,
      vertex_colors: enpt.FloatArray['... V 3'] | None = None,
      vertex_normals: enpt.FloatArray['... V 3'] | None = None,
      multiple_gnms: bool = False,
      verbose: bool = False,
  ) -> MeshRenderOutput:
    """Renders single-frame or batched 3D GNM meshes.

    Args:
      vertices: 3D mesh vertices of shape (..., [M], V, 3).
      world_to_camera: OpenCV world-to-camera matrices of shape (..., 4, 4).
      camera_to_image: OpenCV camera-to-image matrices of shape (..., 4, 4) or
        (..., 3, 3).
      image_size: Output image dimensions as (width, height).
      triangles: Optional active triangle face index array of shape (F, 3) to
        update `self.triangles` via `update_triangles` before rendering.
      triangle_uvs: Optional per-corner UV texture coordinates in UDIM layout of
        shape (F, 3, 2) to update `self.triangle_uvs` before rendering.
      vertex_colors: Optional per-vertex RGB colors override of shape (..., [M],
        V, 3).
      vertex_normals: Optional precomputed vertex normals of shape (..., [M], V,
        3). If None, computed via `gnm_numpy.compute_vertex_normals`.
      multiple_gnms: If True, `vertices` has shape (..., M, V, 3) with M meshes
        per frame.
      verbose: Whether to display a progress bar for this call, overriding the
        `self.verbose` attribute.

    Returns:
      A `MeshRenderOutput` containing the rendered uint8 RGB image of shape
      (..., height, width, 3) and float32 coverage mask in [0.0, 1.0] of shape
      (..., height, width, 1).

    Raises:
      ValueError: If `multiple_gnms=True` and `vertices` or `vertex_normals` is
        < 3D, or if batch dimensions are incompatible.
    """
    if triangles is not None or triangle_uvs is not None:
      self.update_triangles(
          self.triangles if triangles is None else triangles, triangle_uvs
      )

    verbose = verbose or self.verbose
    vertices = np.asarray(vertices, dtype=np.float32)

    # Ensure vertices always has an explicit mesh dimension M at axis -3:
    # shape (..., M, V, 3), where M=1 when multiple_gnms is False.
    if not multiple_gnms:
      vertices = vertices[..., np.newaxis, :, :]  # Inject 'M' dimension.
    elif (vertices_dim := vertices.ndim) < 3:
      raise ValueError(
          f'Called with {multiple_gnms=}, but vertices is only {vertices_dim}D.'
      )

    num_meshes = int(vertices.shape[-3])

    if vertex_colors is None:
      vertex_colors = self.vertex_colors
    else:
      vertex_colors = np.asarray(vertex_colors, dtype=np.float32)

    vertex_colors = _align_mesh_attribute(
        vertex_colors,
        num_meshes=num_meshes,
        multiple_gnms=multiple_gnms,
        allow_2d_with_multiple_gnms=True,
        attribute_name='vertex_colors',
        vertices_shape=vertices.shape,
    )

    if vertex_normals is not None:
      vertex_normals = np.asarray(vertex_normals, dtype=np.float32)
      vertex_normals = _align_mesh_attribute(
          vertex_normals,
          num_meshes=num_meshes,
          multiple_gnms=multiple_gnms,
          allow_2d_with_multiple_gnms=False,
          attribute_name='vertex_normals',
          vertices_shape=vertices.shape,
      )

    world_to_camera = np.asarray(world_to_camera, dtype=np.float32)
    camera_to_image = np.asarray(camera_to_image, dtype=np.float32)

    width, height = int(image_size[0]), int(image_size[1])
    image_size = (width, height)
    render_width = width * self.multisample_antialiasing
    render_height = height * self.multisample_antialiasing
    max_texture_dim = max(render_width, render_height)

    # Compute the common broadcast batch shape across vertices, colors,
    # normals, cameras, and any batched texture maps.
    textures_list = list(self._raw_textures.values())
    try:
      batch_dims = get_batch_dim(
          (vertices, 3),
          (vertex_colors, 3),
          (vertex_normals, 3),
          (world_to_camera, 2),
          (camera_to_image, 2),
          *[(texture_array, 3) for texture_array in textures_list],
      )
    except ValueError as err:
      raise ValueError(
          f' Batch dimensions incompatible: vertices {vertices.shape},'
          f' vertex_colors {vertex_colors.shape}, world_to_camera'
          f' {world_to_camera.shape}, camera_to_image'
          f' {camera_to_image.shape}, texture'
          f' {[texture_array.shape for texture_array in textures_list]}.'
      ) from err

    num_frames = int(np.prod(batch_dims)) if batch_dims else 1

    # Compute per-vertex normals when not explicitly provided.
    if vertex_normals is None:
      vertex_normals = gnm_numpy.compute_vertex_normals(
          vertices, self.triangles
      )

    def batchify[ScalarT: np.generic](
        arr: npt.NDArray[ScalarT], non_batch_dims: int
    ) -> npt.NDArray[ScalarT]:
      """Broadcasts to batch dimensions, and flattens the batch dimensions."""
      arr = np.broadcast_to(arr, (*batch_dims, *arr.shape[-non_batch_dims:]))
      return arr.reshape(num_frames, *arr.shape[-non_batch_dims:])

    vertices = batchify(vertices, 3)
    vertex_normals = batchify(vertex_normals, 3)
    vertex_colors = batchify(vertex_colors, 3)
    world_to_camera = batchify(world_to_camera, 2)
    camera_to_image = batchify(camera_to_image, 2)

    if num_meshes == 1:
      vertices = vertices[:, 0]
      vertex_normals = vertex_normals[:, 0]
      vertex_colors = vertex_colors[:, 0]

    # Only broadcast texture maps across frames when at least one part texture
    # varies across the batch.
    has_multi_frame_textures = any(
        texture_array.ndim > 3 and int(np.prod(texture_array.shape[:-3])) > 1
        for texture_array in textures_list
    )
    batched_textures: dict[int, enpt.ui8['N H_texture W_texture 3']] = {}
    if has_multi_frame_textures:
      batched_textures = {
          tile: batchify(texture_array, 3)
          for tile, texture_array in self._raw_textures.items()
      }

    foreground_frames: list[enpt.FloatArray['H W 3']] = []
    alpha_frames: list[enpt.FloatArray['H W 1']] = []
    frame_inputs = tqdm.tqdm(
        zip(
            vertices,
            vertex_normals,
            vertex_colors,
            world_to_camera,
            camera_to_image,
        ),
        total=num_frames,
        disable=not verbose,
        desc=self._progress_bar_desc,
        leave=False,
    )

    for frame_index, (
        frame_vertices,
        frame_normals,
        frame_colors,
        frame_world_to_camera,
        frame_camera_to_image,
    ) in enumerate(frame_inputs):
      # Sync per-frame vertex colors and textures to the backend only when they
      # differ from the currently active appearance state.
      should_sync_colors = not np.array_equal(
          frame_colors, self._active_vertex_colors
      )
      frame_texture_map: dict[int, enpt.ui8['H_texture W_texture 3']] | None = (
          None
      )
      if has_multi_frame_textures:
        changed_textures: dict[int, enpt.ui8['H_texture W_texture 3']] = {}
        for tile, texture_frames in batched_textures.items():
          texture_frame = texture_frames[frame_index]
          if not np.array_equal(texture_frame, self._active_textures[tile]):
            changed_textures[tile] = texture_frame
        if changed_textures:
          frame_texture_map = changed_textures

      if should_sync_colors or frame_texture_map is not None:
        self._update_frame_appearance(
            vertex_colors=frame_colors if should_sync_colors else None,
            frame_textures=frame_texture_map,
            max_texture_dim=max_texture_dim,
        )

      # Update camera pose/intrinsics and mesh geometry for the current frame.
      self.update_camera(
          world_to_camera=frame_world_to_camera,
          camera_to_image=frame_camera_to_image,
          image_size=image_size,
      )
      self.update_vertices(
          vertices=frame_vertices,
          vertex_normals=frame_normals,
          max_texture_dim=max_texture_dim,
      )
      frame_rgb, frame_alpha = self._render_single_frame(
          width=width, height=height
      )
      foreground_frames.append(frame_rgb)
      alpha_frames.append(frame_alpha)

    self._after_batch_render()

    # Reshape rendered frames back to the original leading batch dimensions.
    stacked_foreground = np.stack(foreground_frames, axis=0).reshape(
        *batch_dims, height, width, 3
    )
    stacked_alpha = np.stack(alpha_frames, axis=0).reshape(
        *batch_dims, height, width, 1
    )
    foreground_uint8 = np.clip(stacked_foreground * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
    return MeshRenderOutput(image=foreground_uint8, alpha=stacked_alpha)

  def close(self) -> None:
    """Releases any persistent backend rendering resources."""
    self._cached_world_to_camera = None
    self._cached_camera_to_image = None
    self._cached_image_size = None

  def _update_frame_appearance(
      self,
      *,
      vertex_colors: enpt.FloatArray['... V 3'] | None = None,
      frame_textures: (
          Mapping[int, enpt.ui8['H_texture W_texture 3']] | None
      ) = None,
      max_texture_dim: int = 1024,
  ) -> None:
    """Updates active per-frame vertex colors and textures."""
    del max_texture_dim
    if vertex_colors is not None:
      self._active_vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
    if frame_textures is not None:
      for tile, texture_frame in frame_textures.items():
        self._active_textures[tile] = texture_frame

  @abc.abstractmethod
  def _on_triangles_updated(self) -> None:
    """Syncs or invalidates backend state after `self.triangles` changes."""

  @abc.abstractmethod
  def _render_single_frame(
      self,
      width: int,
      height: int,
  ) -> tuple[enpt.FloatArray['H W 3'], enpt.FloatArray['H W 1']]:
    """Renders the current scene state for one frame at (height, width)."""

  def _after_batch_render(self) -> None:
    """Hook invoked once after completing a batch of frame renders."""

  def _prepare_vertices_and_normals(
      self,
      vertices: enpt.FloatArray['... V 3'],
      vertex_normals: enpt.FloatArray['... V 3'] | None = None,
  ) -> tuple[enpt.FloatArray['M V 3'], enpt.FloatArray['M V 3']]:
    """Normalizes vertices and vertex normals to shape (M, V, 3)."""
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim == 2:
      vertices = vertices[np.newaxis, :, :]
    if vertex_normals is None:
      vertex_normals = gnm_numpy.compute_vertex_normals(
          vertices, self.triangles
      )
    else:
      vertex_normals = np.asarray(vertex_normals, dtype=np.float32)
      if vertex_normals.ndim == 2:
        vertex_normals = vertex_normals[np.newaxis, :, :]
    return vertices, vertex_normals

  def _is_camera_cached(
      self,
      world_to_camera: enpt.FloatArray['4 4'],
      camera_to_image: enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'],
      image_size: tuple[int, int],
  ) -> bool:
    """Returns True if camera matrices and image_size match cached state."""
    return (
        self._cached_image_size == image_size
        and self._cached_world_to_camera is not None
        and self._cached_camera_to_image is not None
        and np.array_equal(self._cached_world_to_camera, world_to_camera)
        and np.array_equal(self._cached_camera_to_image, camera_to_image)
    )

  def _cache_camera(
      self,
      world_to_camera: enpt.FloatArray['4 4'],
      camera_to_image: enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'],
      image_size: tuple[int, int],
  ) -> None:
    """Caches copies of the camera matrices and image_size."""
    self._cached_world_to_camera = world_to_camera.copy()
    self._cached_camera_to_image = camera_to_image.copy()
    self._cached_image_size = image_size


def project_points_for_gnm(
    gnm_np: gnm_numpy.GNM,
    points_world: FloatArray | None = None,
    vertices: FloatArray | None = None,
    world_to_camera: FloatArray | None = None,
    camera_to_image: FloatArray | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    multiple_gnms: bool = False,
    **kwargs: Any,
) -> FloatArray:
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

  vertices, world_to_camera, camera_to_image = resolve_gnm_camera_and_vertices(
      gnm_np,
      vertices=vertices,
      world_to_camera=world_to_camera,
      camera_to_image=camera_to_image,
      image_size=image_size,
      multiple_gnms=multiple_gnms,
  )
  if not multiple_gnms:
    vertices = vertices[..., np.newaxis, :, :]

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

  def batchify(arr: FloatArray, non_batch_dims: int) -> FloatArray:
    """Broadcasts to batch dimensions, and flattens the batch dimensions."""
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
      view_projection_matrix[:, np.newaxis]
      @ points_homogeneous[..., np.newaxis]
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
    vertices_world: FloatArray,
    azimuthal_angle: FloatArray | float = 0.0,
    polar_angle: FloatArray | float = 0.0,
    camera_distance: FloatArray | float | None = DEFAULT_CAMERA_DISTANCE,
    share_camera: npt.NDArray[np.bool_] | bool = True,
    y_up: npt.NDArray[np.bool_] | bool = False,
    look_at_vertex_groups: Sequence[str] = ('hockey_mask',),
    left_vertex_groups: Sequence[str] = ('ears', '&left'),
    right_vertex_groups: Sequence[str] = ('ears', '&right'),
    forward_vertex_groups: Sequence[str] = ('nose_region',),
) -> FloatArray:
  """Computes world-to-camera matrices for a 'look-at' transform.

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
      share_camera[..., np.newaxis], first_frame_vertices, vertices_world
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
    vertices: FloatArray,
    target_fill_factor: FloatArray | float = DEFAULT_TARGET_FILL_FACTOR,
    camera_distance: FloatArray | float = DEFAULT_CAMERA_DISTANCE,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    vertex_group_name: str = 'hockey_mask',
) -> FloatArray:
  """Computes the camera-to-image matrix for a 'fill factor' transform.

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
    vertices: FloatArray,
    has_time_dimension: bool = False,
    spin_period: int = 60,
    spin_azimuth_limit: float = 20.0,
    spin_polar_limit: float = 5.0,
    num_frames: int | None = None,
    **kwargs: Any,
) -> FloatArray:
  """Computes world-to-camera matrices for a 'spin' transform.

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

  azimuthal_angle = np.sin(angles)[..., np.newaxis] * spin_azimuth_limit
  polar_angle = np.cos(angles)[..., np.newaxis] * spin_polar_limit

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
    vertices: FloatArray,
    left_vertex_groups: Sequence[str],
    right_vertex_groups: Sequence[str],
    forward_vertex_groups: Sequence[str],
    gnm_np: gnm_numpy.GNM,
) -> tuple[FloatArray, FloatArray, FloatArray]:
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
    camera_position: FloatArray, look_at: FloatArray, up_vector: FloatArray
) -> FloatArray:
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

  def _dot_last_axis(arr1: FloatArray, arr2: FloatArray) -> FloatArray:
    return np.einsum('...i,...i->...', arr1, arr2)[..., np.newaxis]

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
    head_axes: tuple[FloatArray, FloatArray, FloatArray],
    camera_distance: FloatArray,
    azimuthal_angle: FloatArray,
    polar_angle: FloatArray,
) -> FloatArray:
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
    vertical_field_of_view: FloatArray,
    aspect_ratio: FloatArray,
    near: FloatArray,
    far: FloatArray,
) -> FloatArray:
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
  image = imageio.imread(texture_path.read_bytes()).astype(np.float32)
  image = (image / np.iinfo(np.uint8).max) * 0.5 + 0.5
  image.flags.writeable = False
  return image


def _vertex_group_mean(
    vertices: FloatArray,
    group_names: Sequence[str],
    gnm_np: gnm_numpy.GNM,
) -> FloatArray:
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
) -> dict[int, UInt8Array]:
  """Loads the texture of each textured UDIM tile as a batchable image.

  Args:
    gnm_np: The GNM model.
    texture: The texture to load. If DEFAULT_TEXTURE, will load the edgeflow
      texture for the skin tile. If an ndarray, will use the given texture for
      the skin tile. If a mapping, will use the given texture for each UDIM tile
      (or, deprecated, for every tile of each GNM part name). If None, no tile
      is textured.

  Returns:
    The texture image of each textured UDIM tile, [0-255] uint8, (..., H, W, 3).
    Tiles without a texture are absent and left to the backend's default
    appearance.

  Raises:
    ValueError: If a texture key is not a UDIM tile or part name of the GNM,
      or if a texture image has invalid dimensions or channel count.
  """
  if isinstance(texture, DefaultTexture):
    edgeflow_path = EDGEFLOW_TEXTURE_BY_BODY_PART.get(gnm_np.body_part)
    edgeflow = (
        load_edgeflow_texture(edgeflow_path)
        if edgeflow_path is not None
        else None
    )
    resolved_texture = (
        {SKIN_UDIM_TILE: edgeflow[..., np.newaxis]}
        if edgeflow is not None
        else None
    )
  else:
    resolved_texture = texture
  normalized = _normalize_textures(resolved_texture)
  if isinstance(texture, Mapping):
    gnm_tiles = sorted(set(gnm_np.triangle_udim_tiles.tolist()))
    if unknown_tiles := set(normalized) - set(gnm_tiles):
      raise ValueError(
          f'Texture keys {unknown_tiles} are not UDIM tiles of the GNM'
          f' {gnm_tiles}.'
      )
  return normalized


def _udim_tiles_of_texture_key(key: int | np.integer | str) -> list[int]:
  """Returns the UDIM tiles a texture key denotes: a tile, or a GNM part."""
  if not isinstance(key, str):
    return [int(key)]  # Also accepts numpy integers.
  warnings.warn(
      'Keying textures by GNM part name is deprecated, key them by UDIM tile'
      ' instead (see gnm_udim.UdimTile).',
      DeprecationWarning,
      # Point at the caller of render_gnm(): load_texture, render_gnm_mesh and
      # render_gnm are the frames in between.
      stacklevel=5,
  )
  tiles = [
      int(tile)
      for tile, groups in gnm_udim.UDIM_TILE_REGIONS.items()
      if key in groups
  ]
  if not tiles:
    raise ValueError(f'Texture key {key!r} is not a GNM part name.')
  return tiles


@overload
def _adjust_scalar_shape(
    array: npt.NDArray[np.bool_] | bool, batch_dims: Sequence[int]
) -> npt.NDArray[np.bool_]:
  ...


@overload
def _adjust_scalar_shape(
    array: FloatArray | float, batch_dims: Sequence[int]
) -> FloatArray:
  ...


def _adjust_scalar_shape(
    array: npt.NDArray[np.generic] | float, batch_dims: Sequence[int]
) -> npt.NDArray[np.generic]:
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
    array = array[..., np.newaxis]

  return np.broadcast_to(array, (*batch_dims, 1))


def get_batch_dim(
    *arrays: tuple[npt.NDArray[np.generic] | None, int],
) -> tuple[int, ...]:
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


def _normalize_textures(
    texture: Texture,
    tiles: Sequence[int] = (),
) -> dict[int, UInt8Array]:
  """Normalizes a texture input into a per-tile 3-channel uint8 dictionary.

  Args:
    texture: Optional texture map, array for skin tile, or mapping.
    tiles: Sequence of required UDIM tiles to fill with white if missing.

  Returns:
    Dictionary mapping UDIM tile numbers to 3-channel uint8 texture arrays.

  Raises:
    ValueError: If `texture` is `DefaultTexture`, if a string texture key is
      not a valid GNM part name, or if a texture image has invalid dimensions
      or channel count.
  """
  if isinstance(texture, DefaultTexture):
    raise ValueError(
        'DEFAULT_TEXTURE requires a GNM model; use from_gnm() or'
        ' load_texture().'
    )
  texture_dict: dict[int, np.ndarray] = {}
  if isinstance(texture, np.ndarray):
    texture_dict[int(SKIN_UDIM_TILE)] = texture
  elif isinstance(texture, Mapping):
    for key, image in texture.items():
      for tile in _udim_tiles_of_texture_key(key):
        texture_dict[tile] = image

  for tile in tiles:
    if int(tile) not in texture_dict:
      texture_dict[int(tile)] = np.full((1, 1, 3), 255, dtype=np.uint8)

  return {
      tile: _to_3channel_uint8(image) for tile, image in texture_dict.items()
  }


def _to_3channel_uint8(
    texture_image: FloatArray | UInt8Array,
) -> UInt8Array:
  """Converts a floating-point or uint8 texture image to 3-channel uint8.

  Args:
    texture_image: Input texture image array.

  Returns:
    3-channel uint8 texture image array.

  Raises:
    ValueError: If `texture_image` has fewer than 2 dimensions or its
      channel count is not 1, 3, or 4.
  """
  if texture_image.ndim < 2:
    raise ValueError(
        'Expected texture_image with at least 2 dimensions, got shape'
        f' {texture_image.shape}.'
    )
  if texture_image.dtype != np.uint8:
    texture_image = np.clip(texture_image, 0.0, 1.0) * 255.0
  texture_image = texture_image.astype(np.uint8, copy=False)
  if texture_image.ndim == 2:
    texture_image = texture_image[..., np.newaxis]
  if texture_image.shape[-1] == 1:
    texture_image = np.repeat(texture_image, 3, axis=-1)
  elif texture_image.shape[-1] == 4:
    texture_image = texture_image[..., :3]
  elif texture_image.shape[-1] != 3:
    raise ValueError(
        'Expected texture_image with 1, 3, or 4 channels, got shape'
        f' {texture_image.shape}.'
    )
  return texture_image


def _align_mesh_attribute(
    attribute: enpt.FloatArray['... V 3'],
    *,
    num_meshes: int,
    multiple_gnms: bool,
    allow_2d_with_multiple_gnms: bool,
    attribute_name: str,
    vertices_shape: tuple[int, ...],
) -> enpt.FloatArray['... M V 3']:
  """Aligns a per-vertex attribute to shape (..., M, V, 3).

  Args:
    attribute: Per-vertex attribute array of shape (..., [M], V, 3).
    num_meshes: Number of meshes M in `vertices`.
    multiple_gnms: Whether `vertices` includes an explicit mesh axis M.
    allow_2d_with_multiple_gnms: Whether a 2D (V, 3) array may be broadcast
      across M when `multiple_gnms=True`.
    attribute_name: Attribute name for error messages ('vertex_colors' or
      'vertex_normals').
    vertices_shape: Shape of `vertices` for error messages.

  Returns:
    Attribute array aligned to shape (..., M, V, 3).

  Raises:
    ValueError: If `multiple_gnms=True` and `attribute` is < 3D (when
      `allow_2d_with_multiple_gnms=False`), or if axis -3 is incompatible with
      `num_meshes`.
  """
  if not multiple_gnms or (allow_2d_with_multiple_gnms and attribute.ndim == 2):
    attribute = attribute[..., np.newaxis, :, :]
  elif (attribute_dim := attribute.ndim) < 3:
    raise ValueError(
        f'Called with {multiple_gnms=}, but {attribute_name} is only'
        f' {attribute_dim}D.'
    )
  if attribute.shape[-3] not in (1, num_meshes):
    raise ValueError(
        f'Batch dimensions incompatible: vertices {vertices_shape},'
        f' {attribute_name} {attribute.shape}.'
    )
  if attribute.shape[-3] != num_meshes:
    target_shape = list(attribute.shape)
    target_shape[-3] = num_meshes
    attribute = np.broadcast_to(attribute, target_shape)
  return attribute
