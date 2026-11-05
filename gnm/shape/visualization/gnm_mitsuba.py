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
batches its arguments; call `render_gnm_mitsuba.render_gnm` or `GNMMeshRenderer`
rather than `render` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, ClassVar, Self

import cv2
import drjit as dr
from gnm.shape import gnm_numpy
from gnm.shape import gnm_typing
from gnm.shape.visualization import integrators
from gnm.shape.visualization import render_common
import mitsuba as mi
import numpy as np

enpt = gnm_typing.enpt

MITSUBA_USE_LEGACY_MESH_API = tuple(
    int(x)
    for x in (
        str(mi.__version__).split('.')  # pyrefly: ignore[missing-attribute]
    )
    if x.isdigit()
) < (3, 9, 1)


def _create_mitsuba_mesh_legacy(
    vertices: enpt.FloatArray['V 3'],
    faces: enpt.IntArray['F 3'],
    *,
    texture_coordinates: enpt.FloatArray['V 2'] | None = None,
    vertex_normals: enpt.FloatArray['V 3'] | None = None,
    properties: mi.Properties | None = None,
    flip_texture_coordinates: bool = False,
    vertex_colors: (
        enpt.FloatArray['V 3'] | enpt.FloatArray['V 4'] | None
    ) = None,
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
    vertices: enpt.FloatArray['V 3'],
    faces: enpt.IntArray['F 3'],
    *,
    texture_coordinates: enpt.FloatArray['V 2'] | None = None,
    vertex_normals: enpt.FloatArray['V 3'] | None = None,
    properties: mi.Properties | None = None,
    flip_texture_coordinates: bool = False,
    vertex_colors: (
        enpt.FloatArray['V 3'] | enpt.FloatArray['V 4'] | None
    ) = None,
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


def _create_sensor_dict(
    world_to_camera: enpt.FloatArray['4 4'],
    camera_to_image: enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'],
    *,
    width: int,
    height: int,
    render_width: int,
    render_height: int,
    spp: int,
) -> dict[str, Any]:
  """Constructs the dictionary spec for a Mitsuba perspective sensor."""
  fov_x, offset_x, offset_y, to_world_mitsuba = _compute_sensor_parameters(
      world_to_camera=world_to_camera,
      camera_to_image=camera_to_image,
      width=width,
      height=height,
  )
  return {
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
          'sample_count': spp,
      },
  }


def _compute_world_light_direction(
    world_to_camera_cv: enpt.FloatArray['4 4'],
) -> enpt.FloatArray['3']:
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


@dataclasses.dataclass(kw_only=True)
class GNMMeshRenderer(render_common.BaseGNMMeshRenderer):
  """Stateful Mitsuba 3D mesh renderer for `gnm_numpy.GNM` models.

  Attributes:
    ambient_light: Ambient light floor in [0, 1]. Defaults to 0.0.
    backface_culling: Whether to cull back-facing triangles. Defaults to True.
    spp: Raytracing samples per pixel. Defaults to 4.
  """

  _progress_bar_desc: ClassVar[str] = 'Rendering (Mitsuba)'

  ambient_light: float = 0.0
  backface_culling: bool = True
  spp: int = 4

  def __post_init__(self) -> None:
    super().__post_init__()
    _ensure_mitsuba_initialized()

  @classmethod
  def from_gnm(
      cls,
      gnm_np: gnm_numpy.GNM,
      *,
      triangles: render_common.TriangleGroups = '~eye_exteriors',
      texture: render_common.Texture = render_common.DEFAULT_TEXTURE,
      multisample_antialiasing: int = 1,
      vertex_colors: enpt.FloatArray['... V 3'] | None = None,
      include_shading: bool = True,
      ambient_light: float = 0.0,
      backface_culling: bool = True,
      verbose: bool = False,
      spp: int = 4,
  ) -> Self:
    """Instantiates a Mitsuba mesh renderer from a NumPy GNM shape model.

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
      ambient_light: Ambient light floor in [0, 1]. Defaults to 0.0.
      backface_culling: Whether to cull back-facing triangles. Defaults to True.
      verbose: Default flag controlling whether to display a progress bar.
      spp: Raytracing samples per pixel. Defaults to 4.

    Returns:
      A new `GNMMeshRenderer` instance configured for `gnm_np`.
    """
    renderer = super().from_gnm(
        gnm_np,
        triangles=triangles,
        texture=texture,
        multisample_antialiasing=multisample_antialiasing,
        vertex_colors=vertex_colors,
        include_shading=include_shading,
        verbose=verbose,
    )
    renderer.ambient_light = ambient_light
    renderer.backface_culling = backface_culling
    renderer.spp = spp
    return renderer

  _cached_bitmaps: dict[tuple[int, int], mi.Bitmap] = dataclasses.field(
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
  _mesh_indices: tuple[int, ...] = dataclasses.field(
      default=(), init=False, repr=False, compare=False
  )
  _active_tiles: tuple[int, ...] = dataclasses.field(
      default=(), init=False, repr=False, compare=False
  )
  _num_meshes: int = dataclasses.field(
      default=1, init=False, repr=False, compare=False
  )
  _sensor: mi.Sensor | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _sensor_params: mi.SceneParameters | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _light_direction: enpt.FloatArray['3'] | None = dataclasses.field(
      default=None,
      init=False,
      repr=False,
      compare=False,
  )

  def update_vertices(
      self,
      vertices: enpt.FloatArray['... V 3'],
      vertex_normals: enpt.FloatArray['... V 3'] | None = None,
      *,
      max_texture_dim: int = 1024,
  ) -> None:
    """Updates the Mitsuba scene mesh vertices and normals in-place."""
    vertices, vertex_normals = self._prepare_vertices_and_normals(
        vertices, vertex_normals
    )

    num_meshes = int(vertices.shape[0])

    with dr.suspend_grad():
      if (
          self._scene is None
          or self._scene_params is None
          or self._num_meshes != num_meshes
      ):
        self._initialize_scene(
            vertices, vertex_normals, max_texture_dim=max_texture_dim
        )
        return

      if MITSUBA_USE_LEGACY_MESH_API:
        for mesh_key, mesh_index, tile in zip(
            self._mesh_keys, self._mesh_indices, self._active_tiles, strict=True
        ):
          corners = self._tile_geometry[tile][0]
          self._scene_params[f'{mesh_key}.vertex_positions'] = np.ravel(
              vertices[mesh_index][corners]
          )
          self._scene_params[f'{mesh_key}.vertex_normals'] = np.ravel(
              vertex_normals[mesh_index][corners]
          )
      else:
        for mesh_key, mesh_index, tile in zip(
            self._mesh_keys, self._mesh_indices, self._active_tiles, strict=True
        ):
          corners = self._tile_geometry[tile][0]
          self._scene_params[f'{mesh_key}.positions'] = mi.TensorXf32(
              vertices[mesh_index][corners]
          )
          self._scene_params[f'{mesh_key}.normals'] = mi.TensorXf32(
              vertex_normals[mesh_index][corners]
          )
      self._scene_params.update()

  def update_camera(
      self,
      world_to_camera: enpt.FloatArray['4 4'],
      camera_to_image: enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'],
      image_size: tuple[int, int],
  ) -> None:
    """Updates camera pose and projection matrices in OpenCV convention."""
    world_to_camera = np.asarray(world_to_camera, dtype=np.float32)
    camera_to_image = np.asarray(camera_to_image, dtype=np.float32)

    if self._sensor is not None and self._is_camera_cached(
        world_to_camera, camera_to_image, image_size
    ):
      return

    width, height = image_size
    render_width = width * self.multisample_antialiasing
    render_height = height * self.multisample_antialiasing

    light_direction = _compute_world_light_direction(world_to_camera)

    if self._sensor is None or self._cached_image_size != image_size:
      sensor_dict = _create_sensor_dict(
          world_to_camera=world_to_camera,
          camera_to_image=camera_to_image,
          width=width,
          height=height,
          render_width=render_width,
          render_height=render_height,
          spp=self.spp,
      )
      with dr.suspend_grad():
        sensor = mi.load_dict(sensor_dict)
      self._sensor = sensor
      self._sensor_params = None
    else:
      fov_x, offset_x, offset_y, to_world_mitsuba = _compute_sensor_parameters(
          world_to_camera=world_to_camera,
          camera_to_image=camera_to_image,
          width=width,
          height=height,
      )
      with dr.suspend_grad():
        sensor_params = self._sensor_params
        if sensor_params is None:
          sensor_params = mi.traverse(self._sensor)
          self._sensor_params = sensor_params
        sensor_params['x_fov'] = fov_x
        sensor_params['principal_point_offset_x'] = offset_x
        sensor_params['principal_point_offset_y'] = offset_y
        sensor_params['to_world'] = mi.ScalarTransform4f(to_world_mitsuba)
        sensor_params.update()

    self._cache_camera(world_to_camera, camera_to_image, image_size)
    self._light_direction = light_direction

    if self._scene is not None:
      integrator = self._scene.integrator()
      if integrator is not None:
        setattr(
            integrator,
            'light_dir',
            mi.ScalarVector3f(
                float(light_direction[0]),
                float(light_direction[1]),
                float(light_direction[2]),
            ),
        )

  def close(self) -> None:
    """Releases cached Mitsuba scene, sensor, and bitmap state."""
    super().close()
    self._cached_bitmaps.clear()
    self._scene = None
    self._scene_params = None
    self._sensor = None
    self._sensor_params = None
    self._light_direction = None
    self._mesh_keys = ()
    self._mesh_indices = ()
    self._active_tiles = ()
    dr.flush_malloc_cache()

  def _update_frame_appearance(
      self,
      *,
      vertex_colors: enpt.FloatArray['... V 3'] | None = None,
      frame_textures: (
          Mapping[int, enpt.ui8['H_texture W_texture 3']] | None
      ) = None,
      max_texture_dim: int = 1024,
  ) -> None:
    """Updates active per-frame vertex colors and Mitsuba textures in-place."""
    super()._update_frame_appearance(
        vertex_colors=vertex_colors,
        frame_textures=frame_textures,
        max_texture_dim=max_texture_dim,
    )
    if frame_textures is not None:
      for tile in frame_textures:
        self._cached_bitmaps.pop((tile, max_texture_dim), None)
    if self._scene_params is not None and (
        vertex_colors is not None or frame_textures is not None
    ):
      colors_per_mesh: enpt.FloatArray['M V 3'] | None = None
      if vertex_colors is not None:
        colors = self._active_vertex_colors
        if colors.ndim == 2:
          colors = np.broadcast_to(
              colors[np.newaxis, :, :],
              (self._num_meshes, *colors.shape),
          )
        colors_per_mesh = colors
      with dr.suspend_grad():
        for mesh_key, mesh_index, tile in zip(
            self._mesh_keys, self._mesh_indices, self._active_tiles, strict=True
        ):
          if colors_per_mesh is not None:
            corners = self._tile_geometry[tile][0]
            tile_colors = np.asarray(
                colors_per_mesh[mesh_index][corners], dtype=np.float32
            )
            if MITSUBA_USE_LEGACY_MESH_API:
              self._scene_params[f'{mesh_key}.vertex_colors'] = np.ravel(
                  tile_colors
              )
            else:
              self._scene_params[f'{mesh_key}.vertex_colors'] = mi.TensorXf32(
                  tile_colors
              )
          if frame_textures is not None and tile in frame_textures:
            bitmap = self._get_tile_bitmap(tile, max_texture_dim)
            self._scene_params[f'{mesh_key}.bsdf.reflectance.data'] = (
                mi.TensorXf(bitmap)
            )
        self._scene_params.update()

  def _on_triangles_updated(self) -> None:
    """Invalidates the Mitsuba scene after `self.triangles` changes."""
    if self._scene is not None:
      self._scene = None
      self._scene_params = None
      self._mesh_keys = ()
      self._mesh_indices = ()
      self._active_tiles = ()

  def _render_single_frame(
      self,
      width: int,
      height: int,
  ) -> tuple[enpt.FloatArray['H W 3'], enpt.FloatArray['H W 1']]:
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
    with dr.suspend_grad():
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

  def _get_tile_bitmap(self, tile: int, max_texture_dim: int) -> mi.Bitmap:
    """Returns a cached `mi.Bitmap` for `tile` at `max_texture_dim`."""
    cache_key = (tile, max_texture_dim)
    cached_bitmap = self._cached_bitmaps.get(cache_key)
    if cached_bitmap is not None:
      return cached_bitmap

    texture_frame = self._active_textures[tile]
    bitmap = _prepare_texture_bitmap(
        texture_frame=texture_frame,
        max_texture_dim=max_texture_dim,
    )
    self._cached_bitmaps[cache_key] = bitmap
    return bitmap

  def _initialize_scene(
      self,
      vertices: enpt.FloatArray['M V 3'],
      vertex_normals: enpt.FloatArray['M V 3'],
      *,
      max_texture_dim: int,
  ) -> None:
    """Constructs the initial `mi.Scene` and caches `mi.traverse(scene)`."""
    light_direction = self._light_direction
    if light_direction is None:
      light_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    num_meshes = int(vertices.shape[0])
    active_colors = self._active_vertex_colors
    if active_colors.size == 0:
      colors_per_mesh = None
    elif active_colors.ndim == 2:
      colors_per_mesh = np.broadcast_to(
          active_colors[np.newaxis, ...], (num_meshes, *active_colors.shape)
      )
    else:
      colors_per_mesh = active_colors

    tile_bsdfs: dict[int, mi.Properties] = {}
    for tile, (corners, _, _) in self._tile_geometry.items():
      if corners.size == 0:
        continue
      properties = mi.Properties()
      properties['bsdf'] = mi.load_dict({
          'type': 'diffuse',
          'reflectance': {
              'type': 'bitmap',
              'bitmap': self._get_tile_bitmap(tile, max_texture_dim),
              'raw': True,
              'filter_type': 'bilinear',
          },
      })
      tile_bsdfs[tile] = properties

    meshes = {}
    mesh_keys: list[str] = []
    mesh_indices: list[int] = []
    active_tiles: list[int] = []
    mesh_counter = 0

    for mesh_index in range(num_meshes):
      mesh_colors = (
          colors_per_mesh[mesh_index] if colors_per_mesh is not None else None
      )
      for tile, (corners, faces, tile_uvs) in self._tile_geometry.items():
        if corners.size == 0:
          continue
        mesh_part = create_mitsuba_mesh(
            vertices=vertices[mesh_index][corners],
            faces=faces,
            texture_coordinates=tile_uvs,
            vertex_normals=vertex_normals[mesh_index][corners],
            vertex_colors=(
                mesh_colors[corners] if mesh_colors is not None else None
            ),
            properties=tile_bsdfs[tile],
            flip_texture_coordinates=True,
        )
        mesh_key = f'mesh_{mesh_counter}'
        meshes[mesh_key] = mesh_part
        mesh_keys.append(mesh_key)
        mesh_indices.append(mesh_index)
        active_tiles.append(tile)
        mesh_counter += 1

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'gnm_half_lambert',
            'light_dir': mi.ScalarVector3f(
                float(light_direction[0]),
                float(light_direction[1]),
                float(light_direction[2]),
            ),
            'include_shading': self.include_shading,
            'ambient_light': self.ambient_light,
            'backface_culling': self.backface_culling,
        },
        **meshes,
    }
    scene = mi.load_dict(scene_dict)
    scene_params = mi.traverse(scene)
    self._scene = scene
    self._scene_params = scene_params
    self._mesh_keys = tuple(mesh_keys)
    self._mesh_indices = tuple(mesh_indices)
    self._active_tiles = tuple(active_tiles)
    self._num_meshes = num_meshes


def render(
    *,
    vertices: enpt.FloatArray['N M V 3'],
    triangles: enpt.IntArray['F 3'],
    world_to_camera: enpt.FloatArray['N 4 4'],
    camera_to_image: enpt.FloatArray['N 4 4'] | enpt.FloatArray['N 3 3'],
    vertex_normals: enpt.FloatArray['N M V 3'],
    triangle_uvs: enpt.FloatArray['F 3 2'],
    vertex_colors: enpt.FloatArray['N M V 3'],
    image_size: tuple[int, int] = (240, 320),
    texture: (
        Mapping[
            int,
            enpt.FloatArray['N H_texture W_texture 3']
            | enpt.ui8['N H_texture W_texture 3'],
        ]
        | None
    ) = None,
    multisample_antialiasing: int = 1,
    background_color: enpt.FloatArray['N H W 3'] | None = None,
    alpha: float = 1.0,
    include_shading: bool = True,
    ambient_light: float = 0.0,
    backface_culling: bool = True,
    verbose: bool = False,
) -> enpt.FloatArray['N H W 3']:
  """Renders GNM meshes using Mitsuba.

  N frames, M meshes.

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
    ambient_light: Ambient light floor in [0, 1].
    backface_culling: Whether to cull back-facing triangles.
    verbose: Whether to print progress bars.

  Returns:
    The rendered color image, float32 [0-1] (N, H, W, 3).
  """
  width, height = image_size
  if vertices.shape[0] == 0:
    return np.empty((0, height, width, 3), dtype=np.float32)
  # pylint: disable=unexpected-keyword-arg
  renderer = GNMMeshRenderer(
      triangles=triangles,
      triangle_uvs=triangle_uvs,
      vertex_colors=vertex_colors,
      texture=texture,
      multisample_antialiasing=multisample_antialiasing,
      include_shading=include_shading,
      ambient_light=ambient_light,
      backface_culling=backface_culling,
      verbose=verbose,
  )
  # pylint: enable=unexpected-keyword-arg
  try:
    output = renderer.render(
        vertices=vertices,
        world_to_camera=world_to_camera,
        camera_to_image=camera_to_image,
        image_size=image_size,
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals,
        multiple_gnms=vertices.ndim >= 4,
        verbose=verbose,
    )
  finally:
    renderer.close()
  return output.composite(
      background=0.0 if background_color is None else background_color,
      alpha=alpha,
      dtype=np.float32,
  )


def _ensure_mitsuba_initialized() -> None:
  """Initializes the Mitsuba variant and registers custom GNM integrators."""
  if not mi.variant():
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  integrators.register()


def _prepare_texture_bitmap(
    texture_frame: (
        enpt.ui8['H_texture W_texture 3']
        | enpt.FloatArray['H_texture W_texture 3']
    ),
    max_texture_dim: int,
) -> enpt.FloatArray['H_out W_out 3']:
  """Prepares a flipped, downsampled float32 texture bitmap in [0.0, 1.0]."""
  is_uint8 = texture_frame.dtype == np.uint8
  texture_frame = texture_frame.astype(np.float32)
  if is_uint8:
    texture_frame = texture_frame / 255.0
  texture_frame = np.clip(texture_frame, 0.0, 1.0)
  texture_height, texture_width = texture_frame.shape[:2]
  if texture_height > max_texture_dim or texture_width > max_texture_dim:
    scale_factor = max_texture_dim / max(texture_height, texture_width)
    scaled_width = max(1, int(round(texture_width * scale_factor)))
    scaled_height = max(1, int(round(texture_height * scale_factor)))
    texture_frame = cv2.resize(
        texture_frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA,
    )
    if texture_frame.ndim == 2:
      texture_frame = texture_frame[..., np.newaxis]
  texture_height, texture_width = texture_frame.shape[:2]
  if texture_height < 2 or texture_width < 2:
    texture_frame = np.pad(
        texture_frame,
        (
            (0, max(0, 2 - texture_height)),
            (0, max(0, 2 - texture_width)),
            (0, 0),
        ),
        mode='edge',
    )
  return mi.Bitmap(np.ascontiguousarray(texture_frame, dtype=np.float32))


def _compute_sensor_parameters(
    world_to_camera: enpt.FloatArray['4 4'],
    camera_to_image: enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'],
    width: int,
    height: int,
) -> tuple[float, float, float, enpt.FloatArray['4 4']]:
  """Computes horizontal FOV, principal point offsets, and to_world matrix."""
  camera_to_image_cv = np.asarray(camera_to_image, dtype=np.float32)
  world_to_camera_cv = np.asarray(world_to_camera, dtype=np.float32)

  fx = float(camera_to_image_cv[0, 0])
  fy = float(camera_to_image_cv[1, 1])
  skew = float(camera_to_image_cv[0, 1])
  cx = float(camera_to_image_cv[0, 2])
  cy = float(camera_to_image_cv[1, 2])

  f_max = max(abs(fx), abs(fy), 1e-6)
  if abs(fx - fy) / f_max > 1e-2 or abs(skew) / f_max > 1e-4:
    raise NotImplementedError(
        'Mitsuba sensor only supports square pixels (fx == fy) and zero skew, '
        f'got fx={fx}, fy={fy}, skew={skew}.'
    )

  focal_length = (fx + fy) / 2.0
  fov_x = float(np.rad2deg(2.0 * np.arctan(width / (2.0 * focal_length))))
  offset_x = float(0.5 - cx / width)
  offset_y = float(0.5 - cy / height)

  if world_to_camera_cv.shape == (3, 4):
    row = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=world_to_camera_cv.dtype)
    world_to_camera_cv = np.concatenate([world_to_camera_cv, row], axis=0)

  c2w_cv = np.linalg.inv(world_to_camera_cv)
  coord_transform = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
  to_world_mitsuba = np.asarray(c2w_cv @ coord_transform, dtype=np.float32)

  return fov_x, offset_x, offset_y, to_world_mitsuba
