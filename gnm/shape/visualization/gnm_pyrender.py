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

"""Render a GNM mesh with pyrender.

This is the rendering backend of `render_gnm`, which prepares and batches its
arguments; call `render_gnm.render_gnm` rather than `render` directly.
"""

import collections.abc
import dataclasses
import functools
import os
import tempfile
from typing import Any
import cv2
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import render_common
import jaxtyping as jt
import numpy as np
import tqdm

if 'PYOPENGL_PLATFORM' not in os.environ:
  os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender  # pylint: disable=wrong-import-position,ungrouped-imports

# Light direction for shading, in camera-space.

_LIGHT_DIRECTION = np.ones([3], dtype=np.float32)
_LIGHT_INTENSITY = 3.34  # Empirically chosen.

# Transparent black background for correct alpha blending.
_BLACK = (0.0, 0.0, 0.0, 0.0)

DEFAULT_RENDERER_FACTORY = pyrender.OffscreenRenderer


class ProjectionMatrixCamera(pyrender.Camera):
  """A camera defined by a full projection matrix."""

  def __init__(self, projection_matrix: np.ndarray):
    super().__init__()
    self._projection_matrix = projection_matrix

  def get_projection_matrix(
      self, width: int | None = None, height: int | None = None
  ) -> np.ndarray:
    del width, height
    return self._projection_matrix

  def set_projection_matrix(self, projection_matrix: np.ndarray):
    self._projection_matrix = projection_matrix


def _changed_flag(array: np.ndarray) -> np.ndarray:
  """Returns a flag indicating whether the array has changed between frames.

  Args:
    array: The data of N frames to check for change, (N, ...).

  Returns:
    A flag indicating whether the array has changed between frames, (N,). Always
    1 for the first frame.
  """

  if array.strides[0] == 0:
    # 'N' dimension is just a broadcast.
    arr = np.zeros(array.shape[0], dtype=np.bool_)
    arr[0] = True
    return arr

  frame_flags = np.any(np.diff(array, axis=0), axis=tuple(range(1, array.ndim)))
  return np.concatenate([np.ones(1, dtype=np.bool_), frame_flags])


def render(
    vertices: np.ndarray,
    triangles: dict[str, np.ndarray],
    world_to_camera: np.ndarray,
    camera_to_image: np.ndarray,
    vertex_normals: np.ndarray,
    vertex_uvs: np.ndarray,
    vertex_colors: np.ndarray,
    image_size: tuple[int, int] = (240, 320),
    texture: dict[str, np.ndarray] | None = None,
    multisample_antialiasing: int = 1,
    background_color: np.ndarray | None = None,
    alpha: float = 1.0,
    include_shading: bool = True,
    verbose: bool = False,
    renderer_factory: collections.abc.Callable[
        [int, int], pyrender.OffscreenRenderer
    ] = DEFAULT_RENDERER_FACTORY,
) -> np.ndarray:
  """Render GNM meshes.

  N frames, M meshes.

  Args:
    vertices: The GNM vertices in world space, (N, M, V, 3).
    triangles: A dictionary of part name to GNM triangles, (F_part, 3).
    world_to_camera: The world-to-camera transform, (N, 4, 4).
    camera_to_image: The camera-to-image transform, (N, 4, 4).
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
    renderer_factory: Optional factory function returning a
      pyrender.OffscreenRenderer.

  Returns:
    The rendered color image, float32 [0-1] (H, W, 3).
  """

  num_frames, num_meshes = vertices.shape[:2]

  # Use multi-sample anti-aliasing (MSAA) to reduce jagged edges.
  width, height = image_size
  render_width = width * multisample_antialiasing
  render_height = height * multisample_antialiasing

  part_names = list(triangles.keys())
  if texture is None:
    texture = {
        part: np.ones((num_frames, 1, 1, 3), dtype=np.float32)
        for part in part_names
    }

  scene = pyrender.Scene(bg_color=_BLACK)

  def _create_texture(frame: int, part: str) -> pyrender.Texture:
    return pyrender.Texture(source=texture[part][frame], source_channels='RGB')

  def _create_mesh(frame: int, gnm_index: int) -> pyrender.Mesh:
    """Set-up GNM mesh and texture."""
    primitives = []
    for part in part_names:
      material = pyrender.MetallicRoughnessMaterial(
          metallicFactor=0.0,
          roughnessFactor=1.0,
      )
      primitives.append(
          pyrender.Primitive(
              positions=vertices[frame, gnm_index],
              indices=triangles[part],
              normals=vertex_normals[frame, gnm_index],
              texcoord_0=vertex_uvs,
              color_0=vertex_colors[frame, gnm_index],
              material=material,
          )
      )
    return pyrender.Mesh(primitives=primitives)

  def _apply_texture(
      mesh_node: pyrender.Node,
      texture: pyrender.Texture | None,
      part_index: int,
  ):
    material = mesh_node.mesh.primitives[part_index].material
    material.baseColorTexture = texture

  # Create a mesh per GNM.
  mesh_nodes = [None] * num_meshes

  # Set-up camera.
  camera = ProjectionMatrixCamera(camera_to_image[0].copy())
  camera_to_world = np.linalg.inv(world_to_camera[0])
  camera_node = scene.add(camera, pose=camera_to_world)

  # Set-up light.
  light_direction_camera = _LIGHT_DIRECTION / np.linalg.norm(_LIGHT_DIRECTION)
  light_pose_camera = _align_vectors(
      np.array([0.0, 0.0, 1.0]), light_direction_camera
  )
  light = pyrender.DirectionalLight(
      color=np.ones(3), intensity=_LIGHT_INTENSITY
  )
  scene.add(light, pose=light_pose_camera, parent_node=camera_node)

  renderer = renderer_factory(render_width, render_height)
  flags = pyrender.constants.RenderFlags.NONE

  if include_shading:
    _override_shader_cache(renderer)
  else:
    flags |= pyrender.constants.RenderFlags.FLAT

  vertices_changed = _changed_flag(vertices)
  vertex_colors_changed = _changed_flag(vertex_colors)
  texture_changed = {part: _changed_flag(texture[part]) for part in part_names}
  world_to_camera_changed = _changed_flag(world_to_camera)
  camera_to_image_changed = _changed_flag(camera_to_image)

  texture_objects = {}
  colors = []
  tqdm_kwargs = dict(disable=not verbose, desc='Rendering', leave=False)
  for f in tqdm.trange(0, num_frames, **tqdm_kwargs):
    # Update anything that's changed.
    if vertices_changed[f] or vertex_colors_changed[f]:
      for m in range(num_meshes):
        # b/487992965: Figure out a way to do this without re-creating the mesh.
        mesh = _create_mesh(f, m)
        if mesh_nodes[m] is not None:
          scene.remove_node(mesh_nodes[m])
        mesh_nodes[m] = scene.add(mesh)
        for index in range(len(part_names)):
          _apply_texture(mesh_nodes[m], texture_objects.get(index, None), index)

    for index, part in enumerate(part_names):
      if texture_changed[part][f]:
        texture_objects[index] = _create_texture(f, part)
        for m in range(num_meshes):
          mesh_node = mesh_nodes[m]
          if mesh_node is not None:
            _apply_texture(mesh_node, texture_objects[index], index)

    if world_to_camera_changed[f]:
      camera_to_world = np.linalg.inv(world_to_camera[f])
      scene.set_pose(camera_node, camera_to_world)

    if camera_to_image_changed[f]:
      camera.set_projection_matrix(camera_to_image[f])

    # Render.
    color, depth = renderer.render(scene, flags=flags)
    alpha_im = alpha * (depth > 0.0).astype(np.float32)

    # Upsample background color for MSAA.
    background = np.zeros_like(color)
    if background_color is not None:
      background = background_color[f]
      if multisample_antialiasing > 1:
        target_size = (render_width, render_height)
        background = cv2.resize(
            background, target_size, interpolation=cv2.INTER_AREA
        )

    # sRGB to linear.
    color = np.clip(color.astype(np.float32) / 255.0, 0.0, 1.0)
    color = np.power(color, 2.2)

    # Alpha blending with background color.
    color = color * alpha_im[..., None] + background * (
        1.0 - alpha_im[..., None]
    )

    # Downsample for anti-aliasing.
    if multisample_antialiasing > 1:
      color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)

    colors.append(color)

  renderer.delete()
  return np.array(colors)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GNMMeshRenderer(render_common.BaseGNMMeshRenderer):
  """Stateful PyRender 3D mesh renderer for `gnm_numpy.GNM` models.

  Supports both OSMesa (`DEFAULT_RENDERER_FACTORY`) and SwiftShader
  (`gnm_pyrender_swiftshader.create_swiftshader_renderer`) backends.

  Attributes:
    renderer_factory: Factory callable `(viewport_width, viewport_height) ->
      pyrender.OffscreenRenderer`.
  """

  renderer_factory: collections.abc.Callable[
      [int, int], pyrender.OffscreenRenderer
  ] = DEFAULT_RENDERER_FACTORY

  _scene: pyrender.Scene | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _mesh_node: pyrender.Node | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _camera: ProjectionMatrixCamera | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _camera_node: pyrender.Node | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _renderer: pyrender.OffscreenRenderer | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _textures: dict[str, pyrender.Texture] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _cached_vertices: jt.Float[np.ndarray, 'V 3'] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _cached_normals: jt.Float[np.ndarray, 'V 3'] | None = dataclasses.field(
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

  def update_vertices(
      self,
      vertices: jt.Float[jt.ArrayLike, 'V 3'],
      vertex_normals: jt.Float[jt.ArrayLike, 'V 3'] | None = None,
      *,
      max_tex_dim: int = 1024,
  ) -> None:
    """Updates the PyRender mesh node with new vertices and normals."""
    del max_tex_dim
    vertices_f32 = np.asarray(vertices, dtype=np.float32)
    if vertex_normals is None:
      normals_f32 = np.asarray(
          self.gnm_np.compute_vertex_normals(vertices_f32[np.newaxis, ...])[0],
          dtype=np.float32,
      )
    else:
      normals_f32 = np.asarray(vertex_normals, dtype=np.float32)

    object.__setattr__(self, '_cached_vertices', vertices_f32)
    object.__setattr__(self, '_cached_normals', normals_f32)

    self._ensure_scene_initialized()
    self._rebuild_mesh_node(vertices_f32, normals_f32)

  def update_camera(
      self,
      world_to_camera: jt.Float[np.ndarray, '4 4'],
      camera_to_image: (
          jt.Float[np.ndarray, '4 4'] | jt.Float[np.ndarray, '3 3']
      ),
      image_size: tuple[int, int],
  ) -> None:
    """Updates the PyRender camera pose, projection matrix, and offscreen renderer."""
    world_to_camera_f32 = np.asarray(world_to_camera, dtype=np.float32)
    camera_to_image_f32 = np.asarray(camera_to_image, dtype=np.float32)

    if (
        self._renderer is not None
        and self._camera is not None
        and self._camera_node is not None
        and self._cached_image_size == image_size
        and self._cached_world_to_camera is not None
        and self._cached_camera_to_image is not None
        and np.array_equal(self._cached_world_to_camera, world_to_camera_f32)
        and np.array_equal(self._cached_camera_to_image, camera_to_image_f32)
    ):
      return

    self._ensure_scene_initialized()
    width, height = image_size
    render_width = width * self.multisample_antialiasing
    render_height = height * self.multisample_antialiasing

    opengl_world_to_camera = camera_conversions.opencv_extrinsics_to_opengl(
        world_to_camera_f32
    )
    opengl_camera_to_image = (
        camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
            camera_to_image_f32,
            width=width,
            height=height,
            near=render_common.DEFAULT_NEAR,
            far=render_common.DEFAULT_FAR,
        )
    )
    camera_to_world = np.linalg.inv(opengl_world_to_camera)

    if (
        self._camera is not None
        and self._camera_node is not None
        and self._scene is not None
    ):
      self._camera.set_projection_matrix(opengl_camera_to_image)
      self._scene.set_pose(self._camera_node, camera_to_world)

    if self._renderer is None or self._cached_image_size != image_size:
      if self._renderer is not None:
        self._renderer.delete()
      renderer = self.renderer_factory(render_width, render_height)
      _override_shader_cache(renderer)
      object.__setattr__(self, '_renderer', renderer)

    object.__setattr__(
        self, '_cached_world_to_camera', world_to_camera_f32.copy()
    )
    object.__setattr__(
        self, '_cached_camera_to_image', camera_to_image_f32.copy()
    )
    object.__setattr__(self, '_cached_image_size', image_size)

  def close(self) -> None:
    """Deletes the offscreen renderer and clears cached PyRender state."""
    if self._renderer is not None:
      self._renderer.delete()
      object.__setattr__(self, '_renderer', None)
    object.__setattr__(self, '_scene', None)
    object.__setattr__(self, '_mesh_node', None)
    object.__setattr__(self, '_camera', None)
    object.__setattr__(self, '_camera_node', None)
    self._textures.clear()

  def _on_triangles_updated(self) -> None:
    """Rebuilds the PyRender mesh node after `_triangle_dict` changes."""
    if (
        self._scene is not None
        and self._cached_vertices is not None
        and self._cached_normals is not None
    ):
      self._rebuild_mesh_node(self._cached_vertices, self._cached_normals)

  def _render_single_frame(
      self,
      width: int,
      height: int,
  ) -> tuple[jt.Float[np.ndarray, 'H W 3'], jt.Float[np.ndarray, 'H W 1']]:
    """Renders the current PyRender scene for one frame.

    Args:
      width: Target output width in pixels.
      height: Target output height in pixels.

    Returns:
      A tuple of `(frame_rgb, frame_alpha)` as float32 arrays of shapes
      `(height, width, 3)` and `(height, width, 1)`.

    Raises:
      RuntimeError: If the PyRender scene, mesh node, or offscreen renderer is
        not initialized.
    """
    if self._scene is None or self._renderer is None or self._mesh_node is None:
      raise RuntimeError(
          'PyRender scene, mesh node, and offscreen renderer must be'
          ' initialized before rendering.'
      )
    color_srgb, depth = self._renderer.render(
        self._scene, flags=pyrender.constants.RenderFlags.NONE
    )
    foreground_rgb = np.clip(color_srgb.astype(np.float32) / 255.0, 0.0, 1.0)
    foreground_rgb = np.power(foreground_rgb, 2.2)
    subpixel_alpha = (depth > 0.0).astype(np.float32)[..., np.newaxis]
    premultiplied_rgb = foreground_rgb * subpixel_alpha

    if self.multisample_antialiasing > 1:
      premultiplied_rgb = cv2.resize(
          premultiplied_rgb, (width, height), interpolation=cv2.INTER_AREA
      )
      msaa_alpha = cv2.resize(
          subpixel_alpha, (width, height), interpolation=cv2.INTER_AREA
      )
      if msaa_alpha.ndim == 2:
        msaa_alpha = msaa_alpha[..., np.newaxis]
    else:
      msaa_alpha = subpixel_alpha

    safe_alpha = np.maximum(msaa_alpha, 1e-6)
    clipped_rgb = np.clip(premultiplied_rgb / safe_alpha, 0.0, 1.0)
    unpremultiplied_rgb = np.where(
        msaa_alpha > 0.0,
        clipped_rgb,
        np.zeros_like(clipped_rgb),
    )
    return (
        np.asarray(unpremultiplied_rgb, dtype=np.float32),
        np.asarray(msaa_alpha, dtype=np.float32),
    )

  def _progress_bar_desc(self) -> str:
    return 'Rendering (PyRender)'

  def _ensure_scene_initialized(self) -> None:
    """Initializes the PyRender scene, camera node, and directional light if needed."""
    if self._scene is not None:
      return
    scene = pyrender.Scene(bg_color=_BLACK)
    camera = ProjectionMatrixCamera(np.eye(4, dtype=np.float32))
    camera_node = scene.add(camera, pose=np.eye(4, dtype=np.float32))
    light_direction_camera = _LIGHT_DIRECTION / np.linalg.norm(_LIGHT_DIRECTION)
    light_pose_camera = _align_vectors(
        np.array([0.0, 0.0, 1.0]), light_direction_camera
    )
    light = pyrender.DirectionalLight(
        color=np.ones(3), intensity=_LIGHT_INTENSITY
    )
    scene.add(light, pose=light_pose_camera, parent_node=camera_node)
    object.__setattr__(self, '_scene', scene)
    object.__setattr__(self, '_camera', camera)
    object.__setattr__(self, '_camera_node', camera_node)

  def _get_part_texture(self, part_name: str) -> pyrender.Texture:
    """Returns a cached `pyrender.Texture` for `part_name`."""
    cached_tex = self._textures.get(part_name)
    if cached_tex is not None:
      return cached_tex

    raw_tex = self._raw_textures.get(part_name)
    if raw_tex is None:
      tex_frame = np.full((1, 1, 3), 255, dtype=np.uint8)
    else:
      tex_frame = (
          raw_tex
          if raw_tex.ndim == 3
          else raw_tex.reshape(-1, *raw_tex.shape[-3:])[0]
      )
    pyrender_tex = pyrender.Texture(source=tex_frame, source_channels='RGB')
    self._textures[part_name] = pyrender_tex
    return pyrender_tex

  def _rebuild_mesh_node(
      self,
      vertices: jt.Float[np.ndarray, 'V 3'],
      vertex_normals: jt.Float[np.ndarray, 'V 3'],
  ) -> None:
    """Rebuilds the PyRender mesh node from `vertices`, `vertex_normals`, and `_triangle_dict`."""
    if self._scene is None:
      return
    primitives: list[pyrender.Primitive] = []
    for part_name, faces in self._triangle_dict.items():
      if len(faces) == 0:
        continue
      material = pyrender.MetallicRoughnessMaterial(
          metallicFactor=0.0,
          roughnessFactor=1.0,
          baseColorTexture=self._get_part_texture(part_name),
      )
      primitives.append(
          pyrender.Primitive(
              positions=vertices,
              indices=faces,
              normals=vertex_normals,
              texcoord_0=self._vertex_uvs,
              color_0=self._vertex_colors,
              material=material,
          )
      )
    mesh = pyrender.Mesh(primitives=primitives)
    if self._mesh_node is not None:
      self._scene.remove_node(self._mesh_node)
    mesh_node = self._scene.add(mesh)
    object.__setattr__(self, '_mesh_node', mesh_node)


@functools.cache
def _custom_fragment_file(fragment_shader: str) -> str:
  """Returns a custom fragment shader file with modified diffuse lighting."""
  filepath = pyrender.shader_program.get_shader_path(fragment_shader)
  with open(filepath, 'r', encoding='utf-8') as f:
    mesh_frag_shader = f.readlines()

  # Replace line containing float nl = clamp(...)
  for i, line in enumerate(mesh_frag_shader):
    if line.strip().startswith('float nl = clamp'):
      mesh_frag_shader[i] = line.replace(
          'clamp(dot(n, l), 0.001, 1.0)',
          'clamp((dot(n, l) + 0.8) / 1.8, 0.0, 1.0)',
      )
      break
  else:
    raise ValueError('Could not find line containing float nl = clamp(...)')

  with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write(''.join(mesh_frag_shader))

  return f.name


class CustomShaderCache(pyrender.shader_program.ShaderProgramCache):
  """Custom shader cache that overrides the fragment shader."""

  def __init__(
      self, original_cache: pyrender.shader_program.ShaderProgramCache
  ):
    super().__init__()
    self.original_cache = original_cache
    self.shader_program = None

  # pylint: disable=keyword-arg-before-vararg
  def get_program(
      self,
      vertex_shader: str,
      fragment_shader: str,
      defines: dict[Any, Any] | None = None,
      *args,
      **kwargs,
  ) -> pyrender.shader_program.ShaderProgram:

    # Remove sRGB correction from textures - our textures are in linear space.
    defines = defines or {}
    defines['SRGB_CORRECTED'] = 0

    # We know shader program is always identical, so we can simplify
    # the shader program cache - and avoid issues existing within pyrender's
    # shader program cache.
    if self.shader_program is not None:
      return self.shader_program

    original_program = self.original_cache.get_program(
        vertex_shader, fragment_shader, defines=defines, *args, **kwargs
    )

    # Create new program with custom fragment shader
    program = pyrender.shader_program.ShaderProgram(
        vertex_shader=original_program.vertex_shader,
        fragment_shader=_custom_fragment_file(fragment_shader),
        geometry_shader=original_program.geometry_shader,
        defines=original_program.defines,
    )
    self.shader_program = program
    return program


def _override_shader_cache(renderer: pyrender.OffscreenRenderer):
  """Override the shader cache with a custom cache."""

  # pylint: disable=protected-access
  if r := renderer._renderer:
    cache = r._program_cache
    r._program_cache = CustomShaderCache(cache)
  # pylint: enable=protected-access


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Returns a 4x4 transform that rotates vector `a` to vector `b`."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  u_a = np.linalg.svd(np.reshape(a, (-1, 1)))[0]
  u_b = np.linalg.svd(np.reshape(b, (-1, 1)))[0]
  u_a[:, -1] *= np.linalg.det(u_a)
  u_b[:, -1] *= np.linalg.det(u_b)
  pose = np.eye(4, dtype=np.float32)
  pose[:3, :3] = u_b @ u_a.T
  return pose
