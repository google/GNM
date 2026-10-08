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
      faces=mi.TensorXu32(
          faces.astype(np.uint32)
      ),
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
  """Render GNM meshes using Mitsuba.

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

  num_frames, num_meshes = vertices.shape[:2]

  width, height = image_size
  render_width = width * multisample_antialiasing
  render_height = height * multisample_antialiasing

  tile_split = udim.split_triangles_by_tile(triangle_uvs)
  tiles = list(tile_split)
  # Unweld each tile's triangles once so that each corner has its own UVs.
  tile_geometry = {}
  for tile, (tile_triangles, tile_uvs) in tile_split.items():
    corners = triangles[tile_triangles].ravel()
    faces = np.arange(corners.size, dtype=np.int32).reshape(-1, 3)
    tile_geometry[tile] = (
        corners,
        faces,
        tile_uvs.reshape(-1, 2).astype(np.float32),
    )
  texture_dict = texture or {}

  texture_bitmaps: dict[int, list[mi.Bitmap]] = {}
  max_tex_dim = max(render_width, render_height)
  # Tiles without a texture are rendered plain white.
  default_texture = np.ones((num_frames, 1, 1, 3), dtype=np.float32)
  for tile in tiles:
    tile_textures = texture_dict.get(tile, default_texture)
    bitmaps = []
    for frame in range(num_frames):
      tex_f = tile_textures[frame].astype(np.float32)
      if tile_textures.dtype == np.uint8 or tex_f.max() > 1.0:
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
    texture_bitmaps[tile] = bitmaps

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
      for tile in tiles:
        corners, faces, tile_uvs = tile_geometry[tile]

        props = mi.Properties()
        props['bsdf'] = mi.load_dict({
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'bitmap': texture_bitmaps[tile][frame],
                'raw': True,
                'filter_type': 'bilinear',
            },
        })

        mesh_part = create_mitsuba_mesh(
            vertices=vertices[frame, mesh_index][corners].astype(np.float32),
            faces=faces,
            texture_coordinates=tile_uvs,
            vertex_normals=vertex_normals[frame, mesh_index][corners].astype(
                np.float32
            ),
            vertex_colors=vertex_colors[frame, mesh_index][corners].astype(
                np.float32
            ),
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
