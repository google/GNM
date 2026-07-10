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

from collections.abc import Sequence
import functools

from etils import epath
from gnm.shape import gnm_numpy
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import gnm_pyrender
from gnm.shape.visualization import vertex_colors as vertex_colors_module
import imageio
import immutabledict
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]

ColorOrImage = npt.NDArray[np.uint8] | FloatArray | Sequence[float] | float

_pkg = __package__ or 'gnm.shape.visualization'
_TEXTURES_DIR = epath.resource_path(_pkg).parent / 'data' / 'textures'
_EDGEFLOW_TEXTURE_BY_BODY_PART = immutabledict.immutabledict({
    gnm_numpy.GNMBodyPart.HEAD: str(_TEXTURES_DIR / 'edgeflow_bw_4k.png'),
})

# Default parameters for scene.
_DEFAULT_IMAGE_SIZE = (240, 320)
_DEFAULT_CAMERA_DISTANCE = 2.0
_DEFAULT_NEAR = 0.01
_DEFAULT_FAR = 100.0
_DEFAULT_TARGET_FILL_FACTOR = 0.4
_DEFAULT_BACKGROUND_COLOR = (0.95, 0.95, 0.95)


# Placeholder for lazy loading the default texture.
class _DefaultTexture:
  """Placeholder for lazy loading the default texture."""


DEFAULT_TEXTURE = _DefaultTexture()
Texture = FloatArray | _DefaultTexture | dict[str, FloatArray] | None


def render_gnm(
    gnm_np: gnm_numpy.GNM,
    vertices: FloatArray | None = None,
    world_to_camera: FloatArray | None = None,
    camera_to_image: FloatArray | None = None,
    image_size: tuple[int, int] = _DEFAULT_IMAGE_SIZE,
    triangles: str | npt.NDArray[np.integer] = 'all_but_eyeball_exterior',
    texture: Texture = DEFAULT_TEXTURE,
    multisample_antialiasing: int = 2,
    background_color: ColorOrImage = _DEFAULT_BACKGROUND_COLOR,
    alpha: float = 1.0,
    vertex_colors: npt.NDArray[np.floating] | None = None,
    multiple_gnms: bool = False,
    include_shading: bool = True,
    verbose: bool = False,
) -> FloatArray:
  """Render GNM meshes.

  Uses a pyrender backend to render GNM meshes.

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
  and Z towards the scene. The camera-to-image
  should project points in the camera coordinate system to normalized
  coordinates in [-1, 1].

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
      will be used for the skin only. If given as a dictionary, then the keys
      should be GNM part names, and the values should be arrays. All arrays
      should be float32 [0-1], (..., H, W, 3).
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

  Returns:
    A rendered image of GNM, (..., H, W, 3).

  Raises:
    ValueError: If multiple_gnms=True, but vertices is only 2D.
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

  colors_has_m_dim = (0, 0, *vertex_colors.shape)[-3] == vertices.shape[-3]
  if not colors_has_m_dim:
    # 'M' dimensions is not present but is required. Inject the dimension, and
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

  # Convert from OpenCV to OpenGL convention.
  world_to_camera = camera_conversions.opencv_extrinsics_to_opengl(
      world_to_camera
  )
  camera_to_image = (
      camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
          camera_to_image,
          width=image_size[0],
          height=image_size[1],
          near=_DEFAULT_NEAR,
          far=_DEFAULT_FAR,
      )
  )

  vertex_normals = gnm_np.compute_vertex_normals(vertices)

  # Broadcast a background color to an image.
  if not isinstance(background_color, np.ndarray) or background_color.ndim == 1:
    background_color = np.broadcast_to(background_color, (height, width, 3))

  # Convert background color to float [0-1].
  if background_color.dtype == np.uint8:
    background_color = background_color.astype(np.float32) / 255.0

  texture = _load_texture(gnm_np, texture)
  textures = list(texture.values())
  if not set(texture.keys()).issubset(gnm_np.mesh_component_names):
    missing_parts = set(texture.keys()) - set(gnm_np.mesh_component_names)
    raise ValueError(
        f'Texture keys {missing_parts} are not GNM part names'
        f' {gnm_np.mesh_component_names}.'
    )

  # Find the maximum batch dimension that satisfies all batch-able arguments.
  try:
    batch_dims = _get_batch_dim(
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
  texture = {part: batchify(texture[part], 3) for part in texture}

  renders = gnm_pyrender.render(
      vertices=vertices,
      triangles=triangle_dict,
      world_to_camera=world_to_camera,
      camera_to_image=camera_to_image,
      image_size=image_size,
      texture=texture,
      vertex_colors=vertex_colors,
      multisample_antialiasing=multisample_antialiasing,
      vertex_uvs=gnm_np.vertex_uvs,
      vertex_normals=vertex_normals,
      background_color=background_color,
      alpha=alpha,
      include_shading=include_shading,
      verbose=verbose,
  )

  width, height = image_size
  color = renders.reshape(*batch_dims, height, width, 3)

  return color


def project_points_for_gnm(
    gnm_np: gnm_numpy.GNM,
    points_world: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    world_to_camera: FloatArray | None = None,
    camera_to_image: FloatArray | None = None,
    image_size: tuple[int, int] = _DEFAULT_IMAGE_SIZE,
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
          near=_DEFAULT_NEAR,
          far=_DEFAULT_FAR,
      )
  )

  # Find the maximum batch dimension that satisfies all batch-able arguments.
  try:
    batch_dims = _get_batch_dim(
        (points_world, 2),
        (vertices, 3),
        (world_to_camera, 2),
        (camera_to_image, 2),
    )
  except ValueError as e:
    raise ValueError(
        f' Batch dimensions incompatible: points_world {points_world.shape},'
        f' vertices {vertices.shape}, world_to_camera{{world_to_camera.shape}},'
        ' camera_to_image {camera_to_image.shape}.'
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
    camera_distance: np.ndarray | float | None = _DEFAULT_CAMERA_DISTANCE,
    share_camera: np.ndarray | bool = True,
    y_up: np.ndarray | bool = False,
    look_at_vertex_group: str = 'hockey_mask',
    left_vertex_group: str = 'left_ear',
    right_vertex_group: str = 'right_ear',
    forward_vertex_group: str = 'nose_region',
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
    look_at_vertex_group: The vertex group to look at.
    left_vertex_group: The vertex group to use for the left axis.
    right_vertex_group: The vertex group to use for the right axis.
    forward_vertex_group: The vertex group to use for the forward axis.

  Returns:
    The world-to-camera matrices, (..., 4, 4).
  """

  batch_dims = vertices_world.shape[:-2]

  if not batch_dims:
    # If there is no batch dimension, we don't need to share the camera.
    share_camera = False

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
      left_vertex_group=left_vertex_group,
      right_vertex_group=right_vertex_group,
      forward_vertex_group=forward_vertex_group,
      gnm_np=gnm_np,
  )
  camera_target = _vertex_group_mean(
      vertices_for_camera, look_at_vertex_group, gnm_np
  )
  camera_location = camera_target + _get_camera_offset(
      gnm_axes,
      camera_distance,
      azimuthal_angle,
      polar_angle,
  )

  y_up_vector = np.array([0.0, 1.0, 0.0])
  up_vector = np.where(y_up, y_up_vector, gnm_axes[1])
  world_to_camera = _right_handed_look_at(
      camera_location, camera_target, up_vector
  )

  world_to_camera = camera_conversions.opengl_extrinsics_to_opencv(
      world_to_camera
  )

  return world_to_camera


def get_fill_factor_camera_to_image(
    gnm_np: gnm_numpy.GNM,
    vertices: np.ndarray,
    target_fill_factor: np.ndarray | float = _DEFAULT_TARGET_FILL_FACTOR,
    camera_distance: np.ndarray | float = _DEFAULT_CAMERA_DISTANCE,
    image_size: tuple[int, int] = _DEFAULT_IMAGE_SIZE,
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

  near = _adjust_scalar_shape(_DEFAULT_NEAR, batch_dims)
  far = _adjust_scalar_shape(_DEFAULT_FAR, batch_dims)

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
    **kwargs: Additional arguments to pass to get_look_at_world_to_camera.

  Returns:
    The world-to-camera matrices, (spin_period, ..., 4, 4) if has_time_dimension
    is False, otherwise (..., 4, 4).
  """

  if has_time_dimension:
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
    left_vertex_group: str,
    right_vertex_group: str,
    forward_vertex_group: str,
    gnm_np: gnm_numpy.GNM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Determines the right, up, and forwards direction vectors of GNM."""

  left = _vertex_group_mean(vertices, left_vertex_group, gnm_np)
  right = _vertex_group_mean(vertices, right_vertex_group, gnm_np)
  forward_point = _vertex_group_mean(vertices, forward_vertex_group, gnm_np)
  back_point = (left + right) / 2.0
  right = right - left
  right = right / np.linalg.norm(right, axis=-1, keepdims=True)
  forwards = forward_point - back_point
  forwards = forwards / np.linalg.norm(forwards, axis=-1, keepdims=True)
  up = np.linalg.cross(right, forwards)
  return right, up, forwards


def _right_handed_look_at(
    camera_position: np.ndarray, look_at: np.ndarray, up_vector: np.ndarray
):
  """Builds a right handed look at view matrix."""
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
def _load_edgeflow_texture(texture_path: str | None) -> FloatArray | None:
  if texture_path is None:
    return None
  with epath.Path(texture_path).open('rb') as f:
    image = imageio.imread(f).astype(np.float32)
  image = (image / np.iinfo(np.uint8).max) * 0.5 + 0.5
  image.flags.writeable = False
  return image


def _vertex_group_mean(
    vertices: np.ndarray,
    group_name: str,
    gnm_np: gnm_numpy.GNM,
):
  """Gets the average point of a GNM vertex group."""
  indices = gnm_np.vertex_group_indices(group_name)
  return vertices[..., indices, :].mean(axis=-2)


def _load_texture(
    gnm_np: gnm_numpy.GNM,
    texture: Texture = DEFAULT_TEXTURE,
) -> dict[str, npt.NDArray[np.uint8]]:
  """Loads the texture as a (potentially batched) image.

  Args:
    gnm_np: The GNM model.
    texture: The texture to load. If _DEFAULT_TEXTURE, will load the edgeflow
      texture. If an ndarray, will use the given texture for skin. If a dict,
      will use the given texture for each part. If None, will use a white
      (plain) texture.

  Returns:
    The texture image, [0-255] uint8, (..., H, W, 3).
  """
  texture_dict = {}
  if texture is DEFAULT_TEXTURE:
    edgeflow_path = _EDGEFLOW_TEXTURE_BY_BODY_PART.get(gnm_np.body_part)
    if edgeflow_path is not None:
      texture_dict['skin'] = _load_edgeflow_texture(edgeflow_path)[..., None]
  elif isinstance(texture, np.ndarray):
    texture_dict['skin'] = texture
  elif isinstance(texture, dict):
    texture_dict = texture

  # Fill remaining parts with white texture.
  for component in gnm_np.mesh_component_names:
    if component not in texture_dict:
      texture_dict[component] = np.ones((64, 64, 1)).astype(np.float32)

  def _to_3channel_uint8(texture_image: np.ndarray) -> npt.NDArray[np.uint8]:
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
  """Adjusts the shape of a scalar to match the batch dimensions."""

  # If the array is a scalar or 1D array, add a final dimension of size 1.
  array = np.array(array)
  if array.ndim <= 1:
    array = array[..., None]

  return np.broadcast_to(array, (*batch_dims, 1))


def _get_batch_dim(*arrays: tuple[np.ndarray | None, int]) -> tuple[int, ...]:
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
  batch_dims = []
  for array, non_batch_dims in arrays:
    if array is None:
      continue
    batch_dims.append(array.shape[:-non_batch_dims])

  max_batch_dims = max(batch_dims, key=len)

  # Verify that all arrays can be broadcast together.
  for batch_dim in batch_dims:
    np.broadcast_shapes(batch_dim, max_batch_dims)

  return max_batch_dims
