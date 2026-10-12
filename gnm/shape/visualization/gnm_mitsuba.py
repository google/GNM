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

from collections.abc import Mapping
import dataclasses
from typing import Any

import cv2
import drjit as dr
from gnm.shape.visualization import integrators
from gnm.shape.visualization import render_common
import jaxtyping as jt
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
    world_to_camera_cv: npt.NDArray[np.floating],
    camera_to_image_cv: npt.NDArray[np.floating],
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
    world_to_camera_cv: npt.NDArray[np.floating],
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


def render(
    *,
    vertices: npt.NDArray[np.floating],
    triangles: Mapping[str, npt.NDArray[np.integer]],
    world_to_camera: npt.NDArray[np.floating],
    camera_to_image: npt.NDArray[np.floating],
    vertex_normals: npt.NDArray[np.floating],
    vertex_uvs: npt.NDArray[np.floating],
    vertex_colors: npt.NDArray[np.floating],
    image_size: tuple[int, int] = (240, 320),
    texture: (
        Mapping[str, npt.NDArray[np.floating] | npt.NDArray[np.uint8]] | None
    ) = None,
    multisample_antialiasing: int = 1,
    background_color: npt.NDArray[np.floating] | None = None,
    alpha: float = 1.0,
    include_shading: bool = True,
    verbose: bool = False,
) -> npt.NDArray[np.float32]:
  """Render GNM meshes using Mitsuba.

  N frames, M meshes.

  Args:
    vertices: The GNM vertices in world space, (N, M, V, 3).
    triangles: A dictionary of part name to GNM triangles, (F_part, 3).
    world_to_camera: The world-to-camera transform (OpenCV convention), (N, 4,
      4).
    camera_to_image: The camera-to-image transform (OpenCV convention), (N, 4,
      4) or (N, 3, 3).
    vertex_normals: The per-vertex normals to use for rendering, (N, M, V, 3).
    vertex_uvs: The per-vertex UV coordinates to use for rendering, (V, 2).
    vertex_colors: The per-vertex colors to use for rendering, (N, M, V, 3).
    image_size: The width and height of the rendered image in pixels: (W, H).
    texture: The per-part texture to use for rendering in linear space,
      {part_name: (N, H, W, 3)}.
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

  num_frames, num_meshes = vertices.shape[:2]

  width, height = image_size
  render_width = width * multisample_antialiasing
  render_height = height * multisample_antialiasing

  part_names = list(triangles.keys())
  if texture is None:
    texture_dict = {
        part: np.ones((num_frames, 1, 1, 3), dtype=np.float32)
        for part in part_names
    }
  else:
    texture_dict = texture

  texture_bitmaps: dict[str, list[mi.Bitmap]] = {}
  max_tex_dim = max(render_width, render_height)
  default_texture = np.ones((num_frames, 1, 1, 3), dtype=np.float32)
  for part in part_names:
    part_textures = texture_dict.get(part, default_texture)
    bitmaps = []
    for frame in range(num_frames):
      tex_f = part_textures[frame].astype(np.float32)
      if part_textures.dtype == np.uint8 or tex_f.max() > 1.0:
        tex_f = tex_f / 255.0
      tex_f = np.clip(tex_f, 0.0, 1.0)
      h, w = tex_f.shape[:2]
      if h > max_tex_dim or w > max_tex_dim:
        scale = max_tex_dim / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(
            1, int(round(h * scale))
        )
        tex_f = cv2.resize(tex_f, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if tex_f.ndim == 2:
          tex_f = tex_f[..., None]
      bitmaps.append(mi.Bitmap(tex_f))
    texture_bitmaps[part] = bitmaps

  composited_frames = []
  tqdm_kwargs = dict(
      disable=not verbose, desc='Rendering (Mitsuba)', leave=False
  )

  for frame in tqdm.trange(num_frames, **tqdm_kwargs):
    to_world_mitsuba, fov_x, offset_x, offset_y = (
        _compute_camera_to_world_and_fov(
            world_to_camera[frame],
            camera_to_image[frame],
            width=width,
            height=height,
        )
    )
    l_world = _compute_world_light_direction(world_to_camera[frame])

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
    sensor = mi.load_dict(sensor_dict)

    scene_dict: dict[str, Any] = {
        'type': 'scene',
        'integrator': {
            'type': 'gnm_half_lambert',
            'light_dir': mi.ScalarVector3f(l_world[0], l_world[1], l_world[2]),
            'include_shading': include_shading,
        },
    }

    mesh_counter = 0
    for mesh_index in range(num_meshes):
      for part in part_names:
        faces = triangles[part]
        if len(faces) == 0:
          continue

        props = mi.Properties()
        props['bsdf'] = mi.load_dict({
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'bitmap': texture_bitmaps[part][frame],
                'raw': True,
                'filter_type': 'bilinear',
            },
        })

        mesh_part = create_mitsuba_mesh(
            vertices=vertices[frame, mesh_index].astype(np.float32),
            faces=faces.astype(np.int32),
            texture_coordinates=vertex_uvs.astype(np.float32)
            if vertex_uvs is not None
            else None,
            vertex_normals=vertex_normals[frame, mesh_index].astype(np.float32)
            if vertex_normals is not None
            else None,
            vertex_colors=vertex_colors[frame, mesh_index].astype(np.float32)
            if vertex_colors is not None
            else None,
            properties=props,
            flip_texture_coordinates=True,
        )
        scene_dict[f'mesh_{mesh_counter}'] = mesh_part
        mesh_counter += 1

    scene = mi.load_dict(scene_dict)

    image_film = mi.render(scene, sensor=sensor, spp=4)

    # The background is a per-frame, per-pixel image, so it cannot be expressed
    # as a Mitsuba emitter/envmap; compositing here also keeps the alpha
    # blending as close as 1-to-1 with the render_gnm rasterization
    # pipelines.
    foreground_rgb = dr.clip(image_film[..., :3], 0.0, 1.0)
    alpha_coverage = image_film[..., 3:4]
    alpha_mask = alpha * dr.select(alpha_coverage > 0.0, 1.0, 0.0)

    if background_color is not None:
      background_rgb = mi.TensorXf(background_color[frame].astype(np.float32))
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

    # Clean up Dr.Jit / Mitsuba allocations.
    del scene, sensor, image_film, scene_dict
    dr.flush_malloc_cache()

  dr.flush_malloc_cache()
  return np.array(composited_frames, dtype=np.float32)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GNMMeshRenderer(render_common.BaseGNMMeshRenderer):
  """Stateful Mitsuba 3D mesh renderer for `gnm_numpy.GNM` models.

  Attributes:
    spp: Raytracing samples per pixel. Defaults to 4.
  """

  spp: int = 4

  _cached_bitmaps: dict[tuple[str, int], mi.Bitmap] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _scene: mi.Scene | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _scene_params: mi.SceneParameters | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _mesh_keys: tuple[str, ...] = dataclasses.field(
      default=(), init=False, repr=False, compare=False
  )
  _active_part_names: tuple[str, ...] = dataclasses.field(
      default=(), init=False, repr=False, compare=False
  )
  _cached_normals: jt.Float[np.ndarray, 'V 3'] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _sensor: mi.Sensor | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _cached_world_to_camera: jt.Float[np.ndarray, '4 4'] | None = (
      dataclasses.field(default=None, init=False, repr=False, compare=False)
  )
  _cached_camera_to_image: (
      jt.Float[np.ndarray, '4 4'] | jt.Float[np.ndarray, '3 3'] | None
  ) = dataclasses.field(default=None, init=False, repr=False, compare=False)
  _cached_image_size: tuple[int, int] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _cached_light_dir: jt.Float[np.ndarray, '3'] = dataclasses.field(
      default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32),
      init=False,
      repr=False,
      compare=False,
  )

  def update_vertices(
      self,
      vertices: jt.Float[jt.ArrayLike, 'V 3'],
      vertex_normals: jt.Float[jt.ArrayLike, 'V 3'] | None = None,
      *,
      max_tex_dim: int = 1024,
  ) -> None:
    """Updates the Mitsuba scene mesh vertices and normals in-place."""
    _ensure_mitsuba_initialized()
    vertices_f32 = np.asarray(vertices, dtype=np.float32)
    if vertex_normals is None:
      normals_f32 = np.asarray(
          self.gnm_np.compute_vertex_normals(vertices_f32[np.newaxis, ...])[0],
          dtype=np.float32,
      )
    else:
      normals_f32 = np.asarray(vertex_normals, dtype=np.float32)

    object.__setattr__(self, '_cached_normals', normals_f32)

    if self._scene is None or self._scene_params is None:
      self._initialize_scene(vertices_f32, normals_f32, max_tex_dim=max_tex_dim)
      return

    if MITSUBA_USE_LEGACY_MESH_API:
      flat_positions = np.ravel(vertices_f32)
      flat_normals = np.ravel(normals_f32)
      for mesh_key in self._mesh_keys:
        self._scene_params[f'{mesh_key}.vertex_positions'] = flat_positions
        self._scene_params[f'{mesh_key}.vertex_normals'] = flat_normals
    else:
      positions_tensor = mi.TensorXf32(vertices_f32)
      normals_tensor = mi.TensorXf32(normals_f32)
      for mesh_key in self._mesh_keys:
        self._scene_params[f'{mesh_key}.positions'] = positions_tensor
        self._scene_params[f'{mesh_key}.normals'] = normals_tensor
    self._scene_params.update()

  def update_camera(
      self,
      world_to_camera: jt.Float[np.ndarray, '4 4'],
      camera_to_image: (
          jt.Float[np.ndarray, '4 4'] | jt.Float[np.ndarray, '3 3']
      ),
      image_size: tuple[int, int],
  ) -> None:
    """Updates the Mitsuba sensor and integrator light direction if changed."""
    _ensure_mitsuba_initialized()
    world_to_camera_f32 = np.asarray(world_to_camera, dtype=np.float32)
    camera_to_image_f32 = np.asarray(camera_to_image, dtype=np.float32)

    if (
        self._sensor is not None
        and self._cached_image_size == image_size
        and self._cached_world_to_camera is not None
        and self._cached_camera_to_image is not None
        and np.array_equal(self._cached_world_to_camera, world_to_camera_f32)
        and np.array_equal(self._cached_camera_to_image, camera_to_image_f32)
    ):
      return

    width, height = image_size
    render_width = width * self.multisample_antialiasing
    render_height = height * self.multisample_antialiasing

    to_world_mitsuba, fov_x, offset_x, offset_y = (
        _compute_camera_to_world_and_fov(
            world_to_camera_f32,
            camera_to_image_f32,
            width=width,
            height=height,
        )
    )
    light_dir = _compute_world_light_direction(world_to_camera_f32)

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
            'sample_count': self.spp,
        },
    }
    sensor = mi.load_dict(sensor_dict)
    object.__setattr__(self, '_sensor', sensor)
    object.__setattr__(
        self, '_cached_world_to_camera', world_to_camera_f32.copy()
    )
    object.__setattr__(
        self, '_cached_camera_to_image', camera_to_image_f32.copy()
    )
    object.__setattr__(self, '_cached_image_size', image_size)
    object.__setattr__(self, '_cached_light_dir', light_dir)

    if self._scene is not None:
      integrator = self._scene.integrator()
      if integrator is not None:
        setattr(
            integrator,
            'light_dir',
            mi.ScalarVector3f(
                float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
            ),
        )

  def close(self) -> None:
    """Releases cached Mitsuba scene, sensor, and Dr.Jit allocations."""
    object.__setattr__(self, '_scene', None)
    object.__setattr__(self, '_scene_params', None)
    object.__setattr__(self, '_sensor', None)
    object.__setattr__(self, '_mesh_keys', ())
    object.__setattr__(self, '_active_part_names', ())
    dr.flush_malloc_cache()

  def _on_triangles_updated(self) -> None:
    """Syncs or invalidates the Mitsuba scene after `_triangle_dict` changes."""
    _ensure_mitsuba_initialized()
    if self._scene is None or self._scene_params is None:
      return

    new_active_parts = tuple(
        part_name
        for part_name, faces in self._triangle_dict.items()
        if len(faces) > 0
    )
    if (
        not MITSUBA_USE_LEGACY_MESH_API
        and new_active_parts == self._active_part_names
    ):
      for mesh_key, part_name in zip(
          self._mesh_keys, self._active_part_names, strict=True
      ):
        part_faces = self._triangle_dict[part_name]
        self._scene_params[f'{mesh_key}.faces'] = mi.TensorXu32(
            part_faces.astype(np.uint32)
        )
        if self._cached_normals is not None:
          self._scene_params[f'{mesh_key}.normals'] = mi.TensorXf32(
              self._cached_normals
          )
      self._scene_params.update()
    else:
      object.__setattr__(self, '_scene', None)
      object.__setattr__(self, '_scene_params', None)
      object.__setattr__(self, '_mesh_keys', ())
      object.__setattr__(self, '_active_part_names', ())

  def _render_single_frame(
      self,
      width: int,
      height: int,
  ) -> tuple[jt.Float[np.ndarray, 'H W 3'], jt.Float[np.ndarray, 'H W 1']]:
    """Renders the current Mitsuba scene and sensor for one frame.

    Args:
      width: Target output width in pixels.
      height: Target output height in pixels.

    Returns:
      A tuple of `(frame_rgb, frame_alpha)` as float32 arrays of shapes
      `(height, width, 3)` and `(height, width, 1)`.

    Raises:
      RuntimeError: If the Mitsuba scene or sensor is not initialized.
    """
    if self._scene is None or self._sensor is None:
      raise RuntimeError(
          'Mitsuba scene and sensor must be initialized before rendering.'
      )
    image_film = mi.render(self._scene, sensor=self._sensor, spp=self.spp)

    foreground_rgb = dr.clip(image_film[..., :3], 0.0, 1.0)
    raw_alpha = image_film[..., 3:4]
    subpixel_alpha = dr.select(
        raw_alpha > 0.0, dr.ones_like(raw_alpha), dr.zeros_like(raw_alpha)
    )
    premultiplied_rgb = foreground_rgb * subpixel_alpha

    if self.multisample_antialiasing > 1:
      premultiplied_rgb = dr.resample(
          premultiplied_rgb, shape=(height, width, 3), filter='box'
      )
      msaa_alpha = dr.resample(
          subpixel_alpha, shape=(height, width, 1), filter='box'
      )
    else:
      msaa_alpha = subpixel_alpha

    safe_alpha = dr.maximum(msaa_alpha, 1e-6)
    clipped_rgb = dr.clip(premultiplied_rgb / safe_alpha, 0.0, 1.0)
    unpremultiplied_rgb = dr.select(
        msaa_alpha > 0.0,
        clipped_rgb,
        dr.zeros_like(clipped_rgb),
    )
    frame_rgb = np.asarray(unpremultiplied_rgb, dtype=np.float32)
    frame_alpha = np.asarray(msaa_alpha, dtype=np.float32)
    del image_film
    return frame_rgb, frame_alpha

  def _after_batch_render(self) -> None:
    dr.flush_malloc_cache()

  def _progress_bar_desc(self) -> str:
    return 'Rendering (Mitsuba)'

  def _get_part_bitmap(self, part_name: str, max_tex_dim: int) -> mi.Bitmap:
    """Returns a cached `mi.Bitmap` for `part_name` at `max_tex_dim`."""
    cache_key = (part_name, max_tex_dim)
    cached_bitmap = self._cached_bitmaps.get(cache_key)
    if cached_bitmap is not None:
      return cached_bitmap

    raw_tex = self._raw_textures.get(part_name)
    if raw_tex is None:
      tex_frame = np.ones((1, 1, 3), dtype=np.float32)
      is_uint8 = False
    else:
      tex_frame = (
          raw_tex
          if raw_tex.ndim == 3
          else raw_tex.reshape(-1, *raw_tex.shape[-3:])[0]
      )
      is_uint8 = raw_tex.dtype == np.uint8

    bitmap = _prepare_texture_bitmap(
        tex_frame=tex_frame,
        is_uint8=is_uint8,
        max_tex_dim=max_tex_dim,
    )
    self._cached_bitmaps[cache_key] = bitmap
    return bitmap

  def _initialize_scene(
      self,
      vertices: jt.Float[np.ndarray, 'V 3'],
      vertex_normals: jt.Float[np.ndarray, 'V 3'],
      *,
      max_tex_dim: int,
  ) -> None:
    """Constructs the initial `mi.Scene` and caches `mi.traverse(scene)`."""
    light_dir = self._cached_light_dir
    scene_dict: dict[
        str, str | dict[str, str | mi.ScalarVector3f] | mi.Mesh
    ] = {
        'type': 'scene',
        'integrator': {
            'type': 'gnm_half_lambert',
            'light_dir': mi.ScalarVector3f(
                float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
            ),
        },
    }

    mesh_keys: list[str] = []
    active_part_names: list[str] = []
    mesh_counter = 0
    for part_name, faces in self._triangle_dict.items():
      if len(faces) == 0:
        continue

      props = mi.Properties()
      props['bsdf'] = mi.load_dict({
          'type': 'diffuse',
          'reflectance': {
              'type': 'bitmap',
              'bitmap': self._get_part_bitmap(part_name, max_tex_dim),
              'raw': True,
              'filter_type': 'bilinear',
          },
      })

      mesh_part = create_mitsuba_mesh(
          vertices=vertices,
          faces=faces,
          texture_coordinates=self._vertex_uvs,
          vertex_normals=vertex_normals,
          vertex_colors=self._vertex_colors,
          properties=props,
          flip_texture_coordinates=True,
      )
      mesh_key = f'mesh_{mesh_counter}'
      scene_dict[mesh_key] = mesh_part
      mesh_keys.append(mesh_key)
      active_part_names.append(part_name)
      mesh_counter += 1

    scene = mi.load_dict(scene_dict)
    scene_params = mi.traverse(scene)
    object.__setattr__(self, '_scene', scene)
    object.__setattr__(self, '_scene_params', scene_params)
    object.__setattr__(self, '_mesh_keys', tuple(mesh_keys))
    object.__setattr__(self, '_active_part_names', tuple(active_part_names))


def _ensure_mitsuba_initialized() -> None:
  """Initializes the Mitsuba variant and registers custom GNM integrators."""
  if not mi.variant():
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  integrators.register()


def _prepare_texture_bitmap(
    tex_frame: (
        jt.Float[np.ndarray, 'H_tex W_tex 3']
        | jt.UInt8[np.ndarray, 'H_tex W_tex 3']
    ),
    is_uint8: bool,
    max_tex_dim: int,
) -> mi.Bitmap:
  """Normalizes, rescales, and converts a 2D texture array into `mi.Bitmap`."""
  tex_f32 = tex_frame.astype(np.float32)
  if is_uint8 or tex_f32.max() > 1.0:
    tex_f32 = tex_f32 / 255.0
  tex_f32 = np.clip(tex_f32, 0.0, 1.0)
  tex_height, tex_width = tex_f32.shape[:2]
  if tex_height > max_tex_dim or tex_width > max_tex_dim:
    scale_factor = max_tex_dim / max(tex_height, tex_width)
    scaled_width = max(1, int(round(tex_width * scale_factor)))
    scaled_height = max(1, int(round(tex_height * scale_factor)))
    tex_f32 = cv2.resize(
        tex_f32, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
    )
    if tex_f32.ndim == 2:
      tex_f32 = tex_f32[..., np.newaxis]
  return mi.Bitmap(tex_f32)
