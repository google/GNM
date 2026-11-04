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

"""Render a GNM mesh with Mitsuba.

This is the rendering backend of `render_gnm_mitsuba`, which prepares and
batches its arguments; call `render_gnm_mitsuba.render_gnm` rather than `render`
directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import threading
from typing import Any

import cv2
import drjit as dr
from gnm.shape.visualization import integrators
from gnm.shape.visualization import udim
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import tqdm

MITSUBA_USE_LEGACY_MESH_API = tuple(
    int(x)
    for x in (
        str(mi.__version__).split('.')  # pyrefly: ignore[missing-attribute]
    )
    if x.isdigit()
) < (3, 9, 1)


def _create_mitsuba_mesh_legacy(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    *,
    texture_coordinates: npt.NDArray[np.floating] | None = None,
    vertex_normals: npt.NDArray[np.floating] | None = None,
    properties: mi.Properties | None = None,
    flip_texture_coordinates: bool = False,
    vertex_colors: npt.NDArray[np.floating] | None = None,
) -> mi.Mesh:
  """Instantiates a Mitsuba mesh using the legacy API (<= 3.9.1)."""
  if properties is None:
    properties = mi.Properties()

  has_vertex_normals = (vertex_normals is not None) and not properties.get(
      'face_normals', False
  )

  mi_mesh = mi.Mesh(  # pyrefly: ignore[missing-argument]
      name='',
      vertex_count=vertices.shape[0],  # pyrefly: ignore[unexpected-keyword]
      face_count=faces.shape[0],  # pyrefly: ignore[unexpected-keyword]
      props=properties,  # pyrefly: ignore[unexpected-keyword]
      has_vertex_normals=(  # pyrefly: ignore[unexpected-keyword]
          has_vertex_normals
      ),
      has_vertex_texcoords=(  # pyrefly: ignore[unexpected-keyword]
          texture_coordinates is not None
      ),
  )
  if vertex_colors is not None:
    if vertex_colors.ndim == 2 and vertex_colors.shape[1] == 4:
      vertex_colors = vertex_colors[:, :3]
    if vertex_colors.dtype != np.float32:
      raise ValueError(
          f'Vertex colors must be of type float32, got {vertex_colors.dtype}'
      )
    mi_mesh.add_attribute(
        'vertex_colors',
        vertex_colors.shape[1],
        vertex_colors.ravel().tolist(),  # pyrefly: ignore[bad-argument-count]
    )

  params = mi.traverse(mi_mesh)
  params['vertex_positions'] = np.ravel(vertices)
  if texture_coordinates is not None:
    if texture_coordinates.dtype != np.float32:
      raise ValueError(
          'Texture coordinates must be of type float32, got'
          f' {texture_coordinates.dtype}'
      )

    if (
        flip_texture_coordinates
        and texture_coordinates.ndim == 2
        and texture_coordinates.shape[1] == 2
    ):
      texture_coordinates = np.array(texture_coordinates)
      texture_coordinates[:, 1] = 1 - texture_coordinates[:, 1]
    params['vertex_texcoords'] = np.ravel(texture_coordinates)

  params['faces'] = np.ravel(faces)

  if vertex_colors is not None:
    params['vertex_colors'] = np.ravel(vertex_colors)

  if vertex_normals is not None:
    params['vertex_normals'] = np.ravel(vertex_normals)

  params.update()

  return mi_mesh


def create_mitsuba_mesh(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    *,
    texture_coordinates: npt.NDArray[np.floating] | None = None,
    vertex_normals: npt.NDArray[np.floating] | None = None,
    properties: mi.Properties | None = None,
    flip_texture_coordinates: bool = False,
    vertex_colors: npt.NDArray[np.floating] | None = None,
) -> mi.Mesh:
  """Instantiates a Mitsuba mesh from vertices, faces, UVs, and normals.

  Args:
    vertices: A NumPy array of vertex positions (shape: (V, 3)) of floating
      point type.
    faces: A NumPy array of triangle indices (shape: (F, 3)) of type int32 or
      uint32.
    texture_coordinates: Optional NumPy array of UV coordinates (shape: (V, 2))
      of type float32.
    vertex_normals: Optional NumPy array of vertex normals (shape: (V, 3)) of
      type float32.
    properties: Optional properties object to be passed to the mi.Mesh
      constructor. The primary purpose of this is to assign a BSDF.
    flip_texture_coordinates: Whether the texture coordinates should be flipped
      vertically. Mitsuba places the UV origin at the top left image corner,
      while some other tools (e.g., Blender) set it as the bottom left.
    vertex_colors: Optional NumPy array of vertex colors (shape: (V, 3) or (V,
      4)) of type float32.

  Returns:
    A new mi.Mesh instance.

  Raises:
    ValueError: If the input data buffers are not as expected.
  """
  if not mi.variant():
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

  with dr.suspend_grad():
    if MITSUBA_USE_LEGACY_MESH_API:
      return _create_mitsuba_mesh_legacy(
          vertices=vertices,
          faces=faces,
          texture_coordinates=texture_coordinates,
          vertex_normals=vertex_normals,
          properties=properties,
          flip_texture_coordinates=flip_texture_coordinates,
          vertex_colors=vertex_colors,
      )

    if properties is None:
      properties = mi.Properties()

    if texture_coordinates is not None:
      if texture_coordinates.dtype != np.float32:
        raise ValueError(
            'Texture coordinates must be of type float32, got'
            f' {texture_coordinates.dtype}'
        )

      if (
          flip_texture_coordinates
          and texture_coordinates.ndim == 2
          and texture_coordinates.shape[1] == 2
      ):
        texture_coordinates = np.array(texture_coordinates)
        texture_coordinates[:, 1] = 1 - texture_coordinates[:, 1]

    normals = mi.TensorXf32()
    if vertex_normals is not None and not properties.get('face_normals', False):
      normals = mi.TensorXf32(vertex_normals.astype(np.float32))

    texcoords = (
        mi.TensorXf32(texture_coordinates.astype(np.float32))
        if texture_coordinates is not None
        else mi.TensorXf32()
    )

    mi_mesh = mi.Mesh(properties)
    mi_mesh.from_fields(
        faces=mi.TensorXu32(faces.astype(np.uint32)),
        positions=mi.TensorXf32(vertices.astype(np.float32)),
        normals=normals,
        texcoords=texcoords,
    )

    if vertex_colors is not None:
      if vertex_colors.ndim == 2 and vertex_colors.shape[1] == 4:
        vertex_colors = vertex_colors[:, :3]
      if vertex_colors.dtype != np.float32:
        raise ValueError(
            f'Vertex colors must be of type float32, got {vertex_colors.dtype}'
        )
      mi_mesh.add_attribute(
          'vertex_colors',
          vertex_colors.astype(np.float32),
      )

    return mi_mesh


def _compute_camera_to_world_and_fov(
    world_to_camera_cv: npt.NDArray[np.float32],
    camera_to_image_cv: npt.NDArray[np.float32],
    *,
    width: int,
    height: int,
) -> tuple[npt.NDArray[np.float32], float, float, float]:
  """Computes Mitsuba to_world transform, FOV, and principal point offsets.

  Args:
    world_to_camera_cv: The world-to-camera transform (OpenCV convention), (4,
      4).
    camera_to_image_cv: The camera-to-image transform (OpenCV convention), (4,
      4) or (3, 3).
    width: The width of the image in pixels.
    height: The height of the image in pixels.

  Returns:
    A tuple of (to_world_mitsuba, fov_x, offset_x, offset_y).
  """
  fx = camera_to_image_cv[0, 0]
  fy = camera_to_image_cv[1, 1]
  skew = camera_to_image_cv[0, 1]
  cx = camera_to_image_cv[0, 2]
  cy = camera_to_image_cv[1, 2]

  f_max = max(abs(fx), abs(fy), 1e-6)
  if abs(fx - fy) / f_max > 1e-4 or abs(skew) / f_max > 1e-4:
    raise NotImplementedError(
        'Mitsuba sensor only supports square pixels (fx == fy) and zero skew, '
        f'got fx={fx}, fy={fy}, skew={skew}.'
    )

  focal_length = (fx + fy) / 2.0
  fov_x = np.rad2deg(2.0 * np.arctan(width / (2.0 * focal_length)))
  offset_x = 0.5 - cx / width
  offset_y = 0.5 - cy / height

  if world_to_camera_cv.shape == (3, 4):
    row = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=world_to_camera_cv.dtype)
    world_to_camera_cv = np.concatenate([world_to_camera_cv, row], axis=0)

  # OpenCV camera-to-world: inv(world_to_camera)
  c2w_cv = np.linalg.inv(world_to_camera_cv)

  # Mitsuba camera coordinate frame transform: diag(-1, -1, 1, 1)
  coord_transform = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
  to_world_mitsuba = (c2w_cv @ coord_transform).astype(np.float32)

  return to_world_mitsuba, float(fov_x), float(offset_x), float(offset_y)


def _compute_world_light_direction(
    world_to_camera_cv: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
  """Computes unit vector in world space pointing TO the light source.

  Args:
    world_to_camera_cv: The world-to-camera transform (OpenCV convention), (4,
      4).

  Returns:
    The unit vector in world space pointing to the light source, (3,).
  """
  # In OpenGL camera space, L_gl = [1, 1, 1] / sqrt(3).
  # In OpenCV camera space, L_cv = [1, -1, -1] / sqrt(3).
  l_cv = np.array([1.0, -1.0, -1.0], dtype=np.float32)
  l_cv /= np.linalg.norm(l_cv)

  r_cv = world_to_camera_cv[:3, :3]
  l_world = r_cv.T @ l_cv
  l_world /= np.linalg.norm(l_world)
  return l_world.astype(np.float32)


def _prepare_tile_textures(
    *,
    texture: (
        Mapping[int, npt.NDArray[np.floating] | npt.NDArray[np.uint8]] | None
    ),
    tiles: Sequence[int],
    num_frames: int,
    max_texture_dimension: int,
) -> dict[int, list[npt.NDArray[np.float32]]]:
  """Prepares contiguous float32 textures per UDIM tile and frame.

  Args:
    texture: Optional mapping from UDIM tile number to per-frame texture arrays
      of shape (N, H, W, C).
    tiles: Ordered sequence of UDIM tile numbers.
    num_frames: Number of frames to render.
    max_texture_dimension: Maximum allowed texture width or height before
      downsampling.

  Returns:
    A dictionary mapping each UDIM tile number to a list of prepared float32
    texture arrays of length `num_frames`. Adjacent identical frames reuse the
    same array instance.
  """
  # Tiles without a texture are rendered plain white.
  default_texture = np.ones((num_frames, 1, 1, 3), dtype=np.float32)
  texture_dict = texture or {}

  prepared_textures: dict[int, list[npt.NDArray[np.float32]]] = {}
  for tile in tiles:
    tile_textures = texture_dict.get(tile, default_texture)
    frame_textures: list[npt.NDArray[np.float32]] = []
    for frame in range(num_frames):
      if frame > 0 and np.array_equal(
          tile_textures[frame], tile_textures[frame - 1]
      ):
        frame_textures.append(frame_textures[-1])
        continue

      texture_float = tile_textures[frame].astype(np.float32, copy=False)
      if tile_textures.dtype == np.uint8:
        texture_float = texture_float / 255.0
      texture_float = np.clip(texture_float, 0.0, 1.0)
      texture_height, texture_width = texture_float.shape[:2]
      if (
          texture_height > max_texture_dimension
          or texture_width > max_texture_dimension
      ):
        scale = max_texture_dimension / max(texture_height, texture_width)
        resized_width = max(1, int(round(texture_width * scale)))
        resized_height = max(1, int(round(texture_height * scale)))
        texture_float = cv2.resize(
            texture_float,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
      if texture_float.ndim == 2:
        texture_float = texture_float[..., np.newaxis]
      elif texture_float.ndim == 3 and texture_float.shape[2] == 4:
        texture_float = texture_float[..., :3]
      texture_height, texture_width = texture_float.shape[:2]
      if texture_height < 2 or texture_width < 2:
        texture_float = np.pad(
            texture_float,
            (
                (0, max(0, 2 - texture_height)),
                (0, max(0, 2 - texture_width)),
                (0, 0),
            ),
            mode='edge',
        )
      frame_textures.append(
          np.ascontiguousarray(texture_float, dtype=np.float32)
      )
    prepared_textures[tile] = frame_textures
  return prepared_textures


def _create_mitsuba_sensor(
    *,
    world_to_camera_cv: npt.NDArray[np.float32],
    camera_to_image_cv: npt.NDArray[np.float32],
    width: int,
    height: int,
    render_width: int,
    render_height: int,
) -> mi.Sensor:
  """Instantiates the Mitsuba perspective sensor for the initial frame.

  Args:
    world_to_camera_cv: The world-to-camera transform (OpenCV convention), (4,
      4).
    camera_to_image_cv: The camera-to-image transform (OpenCV convention), (4,
      4) or (3, 3).
    width: The target output image width in pixels.
    height: The target output image height in pixels.
    render_width: The supersampled film width in pixels.
    render_height: The supersampled film height in pixels.

  Returns:
    A configured `mi.Sensor` instance.
  """
  to_world_mitsuba, fov_x, offset_x, offset_y = (
      _compute_camera_to_world_and_fov(
          world_to_camera_cv=world_to_camera_cv,
          camera_to_image_cv=camera_to_image_cv,
          width=width,
          height=height,
      )
  )
  sensor_dict = {
      'type': 'perspective',
      'fov': fov_x,
      'principal_point_offset_x': offset_x,
      'principal_point_offset_y': offset_y,
      'to_world': mi.ScalarTransform4f(to_world_mitsuba),
      'film': {
          'type': 'hdrfilm',
          'width': render_width,
          'height': render_height,
          'pixel_format': 'rgba',
          'rfilter': {'type': 'box'},
      },
      'sampler': {
          'type': 'independent',
          'sample_count': 4,
      },
  }
  return mi.load_dict(sensor_dict)


# Mitsuba sensors by variant and film size, see `_get_mitsuba_sensor()`.
_SENSORS: dict[tuple[str | None, int, int], mi.Sensor] = {}
# Guards `_SENSORS` and its sensors, whose cameras `render()` updates per frame.
_SENSORS_LOCK = threading.Lock()


def _get_mitsuba_sensor(
    *,
    world_to_camera_cv: npt.NDArray[np.float32],
    camera_to_image_cv: npt.NDArray[np.float32],
    width: int,
    height: int,
    render_width: int,
    render_height: int,
) -> tuple[mi.Sensor, mi.SceneParameters]:
  """Returns the Mitsuba sensor of the film size, set up for the initial frame.

  The sensors are kept for the lifetime of the process instead of being created
  for every call. A film keeps its crop window in a small device buffer (see
  `Film::launch_params()`), and the buffer of a new film may be allocated at the
  address of a freed one. On A100 GPUs (driver 535), OptiX rendering kernels
  then read the crop window of the freed film in some threads, which shears or
  blanks the renders. Each sensor also keeps the image storage of its film.

  The caller must hold `_SENSORS_LOCK` while it uses the sensor.

  Args:
    world_to_camera_cv: The world-to-camera transform (OpenCV convention), (4,
      4).
    camera_to_image_cv: The camera-to-image transform (OpenCV convention), (4,
      4) or (3, 3).
    width: The target output image width in pixels.
    height: The target output image height in pixels.
    render_width: The supersampled film width in pixels.
    render_height: The supersampled film height in pixels.

  Returns:
    The sensor and its traversable parameters.
  """
  key = (mi.variant(), render_width, render_height)
  if key not in _SENSORS:
    _SENSORS[key] = _create_mitsuba_sensor(
        world_to_camera_cv=world_to_camera_cv,
        camera_to_image_cv=camera_to_image_cv,
        width=width,
        height=height,
        render_width=render_width,
        render_height=render_height,
    )
    return _SENSORS[key], mi.traverse(_SENSORS[key])
  sensor_parameters = mi.traverse(_SENSORS[key])
  _update_mitsuba_sensor(
      sensor_parameters=sensor_parameters,
      world_to_camera_cv=world_to_camera_cv,
      camera_to_image_cv=camera_to_image_cv,
      width=width,
      height=height,
  )
  return _SENSORS[key], sensor_parameters


def _index_optional(
    array: npt.NDArray[np.float32] | None,
    index: int | slice | npt.NDArray[np.int32] | tuple[int | slice, ...],
) -> npt.NDArray[np.float32] | None:
  """Returns `array[index]` when `array` is not `None`, otherwise `None`."""
  return array[index] if array is not None else None


def _create_mitsuba_scene(
    *,
    vertices_frame: npt.NDArray[np.float32],
    tile_corners: Mapping[int, npt.NDArray[np.int32]],
    tile_uvs: Mapping[int, npt.NDArray[np.float32]],
    tiles: Sequence[int],
    vertex_normals_frame: npt.NDArray[np.float32] | None,
    vertex_colors_frame: npt.NDArray[np.float32] | None,
    prepared_textures: Mapping[int, Sequence[npt.NDArray[np.float32]]],
    world_to_camera_cv: npt.NDArray[np.float32],
    include_shading: bool,
) -> mi.Scene:
  """Instantiates the Mitsuba scene and tile meshes for the initial frame.

  Args:
    vertices_frame: Vertex positions for the initial frame, (M, V, 3).
    tile_corners: Mapping from UDIM tile number to unwelded corner vertex
      indices.
    tile_uvs: Mapping from UDIM tile number to unwelded corner UV coordinates.
    tiles: Ordered sequence of UDIM tile numbers.
    vertex_normals_frame: Optional vertex normals for the initial frame, (M, V,
      3).
    vertex_colors_frame: Optional vertex colors for the initial frame, (M, V,
      3).
    prepared_textures: Prepared per-tile textures for each frame.
    world_to_camera_cv: Initial frame world-to-camera transform, (4, 4).
    include_shading: Whether half-Lambert shading is enabled in the integrator.

  Returns:
    A configured `mi.Scene` instance.
  """
  light_direction_world = _compute_world_light_direction(world_to_camera_cv)
  scene_dict: dict[str, Any] = {
      'type': 'scene',
      'integrator': {
          'type': 'gnm_half_lambert',
          'light_dir': mi.ScalarVector3f(
              light_direction_world[0],
              light_direction_world[1],
              light_direction_world[2],
          ),
          'include_shading': include_shading,
      },
  }

  num_meshes = vertices_frame.shape[0]
  mesh_counter = 0
  for mesh_index in range(num_meshes):
    mesh_normals = _index_optional(vertex_normals_frame, mesh_index)
    mesh_colors = _index_optional(vertex_colors_frame, mesh_index)
    for tile in tiles:
      corners = tile_corners[tile]
      if corners.size == 0:
        continue
      faces = np.arange(corners.size, dtype=np.int32).reshape(-1, 3)

      properties = mi.Properties()
      properties['bsdf'] = mi.load_dict({
          'type': 'diffuse',
          'reflectance': {
              'type': 'bitmap',
              'bitmap': mi.Bitmap(prepared_textures[tile][0]),
              'raw': True,
              'filter_type': 'bilinear',
          },
      })

      mesh_part = create_mitsuba_mesh(
          vertices=vertices_frame[mesh_index][corners],
          faces=faces,
          texture_coordinates=tile_uvs[tile],
          vertex_normals=_index_optional(mesh_normals, corners),
          vertex_colors=_index_optional(mesh_colors, corners),
          properties=properties,
          flip_texture_coordinates=True,
      )
      scene_dict[f'mesh_{mesh_counter}'] = mesh_part
      mesh_counter += 1

  return mi.load_dict(scene_dict)


def _update_mitsuba_sensor(
    *,
    sensor_parameters: mi.SceneParameters,
    world_to_camera_cv: npt.NDArray[np.float32],
    camera_to_image_cv: npt.NDArray[np.float32],
    width: int,
    height: int,
) -> None:
  """Updates the Mitsuba perspective sensor parameters for the current frame.

  Args:
    sensor_parameters: Traversable parameters of the Mitsuba sensor.
    world_to_camera_cv: Current frame world-to-camera transform, (4, 4).
    camera_to_image_cv: Current frame camera-to-image transform, (4, 4) or (3,
      3).
    width: Target output image width in pixels.
    height: Target output image height in pixels.
  """
  to_world_mitsuba, fov_x, offset_x, offset_y = (
      _compute_camera_to_world_and_fov(
          world_to_camera_cv=world_to_camera_cv,
          camera_to_image_cv=camera_to_image_cv,
          width=width,
          height=height,
      )
  )
  sensor_parameters['x_fov'] = fov_x
  sensor_parameters['principal_point_offset_x'] = offset_x
  sensor_parameters['principal_point_offset_y'] = offset_y
  sensor_parameters['to_world'] = mi.ScalarTransform4f(to_world_mitsuba)
  sensor_parameters.update()


def _update_mitsuba_scene(
    *,
    scene_parameters: mi.SceneParameters,
    vertices: npt.NDArray[np.float32],
    vertex_normals: npt.NDArray[np.float32] | None,
    vertex_colors: npt.NDArray[np.float32] | None,
    tile_corners: Mapping[int, npt.NDArray[np.int32]],
    tiles: Sequence[int],
    prepared_textures: Mapping[int, Sequence[npt.NDArray[np.float32]]],
    world_to_camera: npt.NDArray[np.float32],
    frame: int,
) -> None:
  """Updates Mitsuba scene light direction, mesh attributes, and textures.

  Args:
    scene_parameters: Traversable parameters of the Mitsuba scene.
    vertices: All frames' vertex positions, (N, M, V, 3).
    vertex_normals: Optional per-frame vertex normals, (N, M, V, 3).
    vertex_colors: Optional per-frame vertex colors, (N, M, V, 3).
    tile_corners: Mapping from UDIM tile number to unwelded corner vertex
      indices.
    tiles: Ordered sequence of UDIM tile numbers.
    prepared_textures: Prepared per-tile textures for each frame.
    world_to_camera: All frames' world-to-camera transforms, (N, 4, 4).
    frame: Index of the current frame (`frame > 0`).
  """
  scene_updated = False
  if not np.array_equal(world_to_camera[frame], world_to_camera[frame - 1]):
    light_direction_world = _compute_world_light_direction(
        world_to_camera[frame]
    )
    scene_parameters['integrator.light_dir'] = mi.ScalarVector3f(
        light_direction_world[0],
        light_direction_world[1],
        light_direction_world[2],
    )
    scene_updated = True

  num_meshes = vertices.shape[1]
  mesh_counter = 0
  for mesh_index in range(num_meshes):
    vertices_changed = not np.array_equal(
        vertices[frame, mesh_index], vertices[frame - 1, mesh_index]
    )
    normals_changed = vertex_normals is not None and not np.array_equal(
        vertex_normals[frame, mesh_index],
        vertex_normals[frame - 1, mesh_index],
    )
    colors_changed = vertex_colors is not None and not np.array_equal(
        vertex_colors[frame, mesh_index],
        vertex_colors[frame - 1, mesh_index],
    )
    update_geometry = vertices_changed or normals_changed
    frame_positions = vertices[frame, mesh_index] if update_geometry else None
    frame_normals = (
        _index_optional(vertex_normals, (frame, mesh_index))
        if update_geometry
        else None
    )
    frame_colors = (
        _index_optional(vertex_colors, (frame, mesh_index))
        if colors_changed
        else None
    )
    for tile in tiles:
      corners = tile_corners[tile]
      if corners.size == 0:
        continue

      if frame_positions is not None:
        tile_positions = frame_positions[corners]
        if MITSUBA_USE_LEGACY_MESH_API:
          scene_parameters[f'mesh_{mesh_counter}.vertex_positions'] = np.ravel(
              tile_positions
          )
        else:
          scene_parameters[f'mesh_{mesh_counter}.positions'] = mi.TensorXf32(
              tile_positions
          )
        scene_updated = True

      if frame_normals is not None:
        tile_normals = frame_normals[corners]
        if MITSUBA_USE_LEGACY_MESH_API:
          scene_parameters[f'mesh_{mesh_counter}.vertex_normals'] = np.ravel(
              tile_normals
          )
        else:
          scene_parameters[f'mesh_{mesh_counter}.normals'] = mi.TensorXf32(
              tile_normals
          )
        scene_updated = True

      if frame_colors is not None:
        tile_colors = frame_colors[corners]
        if MITSUBA_USE_LEGACY_MESH_API:
          scene_parameters[f'mesh_{mesh_counter}.vertex_colors'] = np.ravel(
              tile_colors
          )
        else:
          scene_parameters[f'mesh_{mesh_counter}.vertex_colors'] = (
              mi.TensorXf32(tile_colors)
          )
        scene_updated = True

      if (
          prepared_textures[tile][frame]
          is not prepared_textures[tile][frame - 1]
      ):
        scene_parameters[f'mesh_{mesh_counter}.bsdf.reflectance.data'] = (
            mi.TensorXf(prepared_textures[tile][frame])
        )
        scene_updated = True

      mesh_counter += 1

  if scene_updated:
    scene_parameters.update()


def render(
    *,
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
    world_to_camera: npt.NDArray[np.floating],
    camera_to_image: npt.NDArray[np.floating],
    vertex_normals: npt.NDArray[np.floating],
    triangle_uvs: npt.NDArray[np.floating],
    vertex_colors: npt.NDArray[np.floating],
    image_size: tuple[int, int] = (240, 320),
    texture: (
        Mapping[int, npt.NDArray[np.floating] | npt.NDArray[np.uint8]] | None
    ) = None,
    multisample_antialiasing: int = 1,
    background_color: npt.NDArray[np.floating] | None = None,
    alpha: float = 1.0,
    include_shading: bool = True,
    verbose: bool = False,
) -> npt.NDArray[np.float32]:
  """Renders GNM meshes using Mitsuba.

  N frames, M meshes. See `render_common.BackendRenderFn`.

  Args:
    vertices: The GNM vertices in world space, (N, M, V, 3).
    triangles: The GNM triangles to render, (F, 3).
    world_to_camera: The world-to-camera transform (OpenCV convention), (N, 4,
      4).
    camera_to_image: The camera-to-image transform (OpenCV convention), (N, 4,
      4) or (N, 3, 3).
    vertex_normals: The per-vertex normals to use for rendering, (N, M, V, 3).
    triangle_uvs: The UV coordinates of the triangle corners in UDIM layout, (F,
      3, 2).
    vertex_colors: The per-vertex colors to use for rendering, (N, M, V, 3).
    image_size: The width and height of the rendered image in pixels: (W, H).
    texture: The per-tile texture to use for rendering in linear space,
      {udim_tile: (N, H, W, 3)}.
    multisample_antialiasing: Render with e.g., double resolution, and then
      downsample for anti-aliasing.
    background_color: The background color, float32 [0-1] (N, H, W, 3).
    alpha: A float [0-1] to be multiplied by the render-alpha to determine
      blending with the background color.
    include_shading: If False, disable shading in render.
    verbose: Whether to print progress bars.

  Returns:
    The rendered color image, float32 [0-1] (N, H, W, 3).
  """
  if not mi.variant():
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  integrators.register()

  num_frames = vertices.shape[0]
  width, height = image_size
  if num_frames == 0:
    return np.empty((0, height, width, 3), dtype=np.float32)

  vertices = vertices.astype(np.float32, copy=False)
  triangles_int32 = triangles.astype(np.int32, copy=False)
  world_to_camera = world_to_camera.astype(np.float32, copy=False)
  camera_to_image = camera_to_image.astype(np.float32, copy=False)
  if vertex_normals is not None:
    vertex_normals = vertex_normals.astype(np.float32, copy=False)
  if vertex_colors is not None:
    vertex_colors = vertex_colors[..., :3].astype(np.float32, copy=False)
  if background_color is not None:
    background_color = background_color.astype(np.float32, copy=False)

  render_width = width * multisample_antialiasing
  render_height = height * multisample_antialiasing

  tile_split = udim.split_triangles_by_tile(triangle_uvs)
  tiles = list(tile_split)
  # Unweld each tile's triangles once so that each corner has its own UVs.
  tile_corners = {
      tile: triangles_int32[tile_triangles].ravel()
      for tile, (tile_triangles, _) in tile_split.items()
  }
  tile_uvs = {
      tile: uvs.reshape(-1, 2).astype(np.float32, copy=False)
      for tile, (_, uvs) in tile_split.items()
  }

  prepared_textures = _prepare_tile_textures(
      texture=texture,
      tiles=tiles,
      num_frames=num_frames,
      max_texture_dimension=max(render_width, render_height),
  )

  with _SENSORS_LOCK, dr.suspend_grad():
    sensor, sensor_parameters = _get_mitsuba_sensor(
        world_to_camera_cv=world_to_camera[0],
        camera_to_image_cv=camera_to_image[0],
        width=width,
        height=height,
        render_width=render_width,
        render_height=render_height,
    )

    scene = _create_mitsuba_scene(
        vertices_frame=vertices[0],
        tile_corners=tile_corners,
        tile_uvs=tile_uvs,
        tiles=tiles,
        vertex_normals_frame=_index_optional(vertex_normals, 0),
        vertex_colors_frame=_index_optional(vertex_colors, 0),
        prepared_textures=prepared_textures,
        world_to_camera_cv=world_to_camera[0],
        include_shading=include_shading,
    )
    scene_parameters = mi.traverse(scene)

    composited_frames = []
    tqdm_kwargs = {
        'disable': not verbose,
        'desc': 'Rendering (Mitsuba)',
        'leave': False,
    }

    for frame in tqdm.trange(num_frames, **tqdm_kwargs):
      if frame > 0:
        if not (
            np.array_equal(world_to_camera[frame], world_to_camera[frame - 1])
            and np.array_equal(
                camera_to_image[frame], camera_to_image[frame - 1]
            )
        ):
          _update_mitsuba_sensor(
              sensor_parameters=sensor_parameters,
              world_to_camera_cv=world_to_camera[frame],
              camera_to_image_cv=camera_to_image[frame],
              width=width,
              height=height,
          )
        _update_mitsuba_scene(
            scene_parameters=scene_parameters,
            vertices=vertices,
            vertex_normals=vertex_normals,
            vertex_colors=vertex_colors,
            tile_corners=tile_corners,
            tiles=tiles,
            prepared_textures=prepared_textures,
            world_to_camera=world_to_camera,
            frame=frame,
        )

      image_film = mi.render(scene, sensor=sensor, spp=4)

      # The background is a per-frame, per-pixel image, so it cannot be
      # expressed as a Mitsuba emitter/envmap; compositing here also keeps the
      # alpha blending as close as 1-to-1 with the render_gnm rasterization
      # pipelines.
      foreground_rgb = dr.clip(image_film[..., :3], 0.0, 1.0)
      alpha_coverage = image_film[..., 3:4]
      alpha_mask = alpha * dr.select(alpha_coverage > 0.0, 1.0, 0.0)

      if background_color is not None:
        background_rgb = mi.TensorXf(background_color[frame])
        if multisample_antialiasing > 1:
          background_rgb = dr.resample(
              background_rgb,
              shape=(render_height, render_width, 3),
              filter='box',
          )
      else:
        background_rgb = dr.zeros(mi.TensorXf, foreground_rgb.shape)

      composited_rgb = foreground_rgb * alpha_mask + background_rgb * (
          1.0 - alpha_mask
      )

      if multisample_antialiasing > 1:
        composited_rgb = dr.resample(
            composited_rgb, shape=(height, width, 3), filter='box'
        )

      composited_frames.append(np.array(composited_rgb))
      del image_film, composited_rgb

  return np.array(composited_frames, dtype=np.float32)
