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

"""Render a GNM mesh with pyrender."""

import functools
import os
import tempfile
from typing import Any
import cv2
import numpy as np
import tqdm
import trimesh

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender  # pylint: disable=wrong-import-position

# Light direction for shading, in camera-space.

_LIGHT_DIRECTION = np.ones([3], dtype=np.float32)
_LIGHT_INTENSITY = 3.34  # Empirically chosen.


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

  scene = pyrender.Scene()

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
  light_pose_camera = trimesh.geometry.align_vectors(
      [0, 0, 1], light_direction_camera
  )
  light = pyrender.DirectionalLight(
      color=np.ones(3), intensity=_LIGHT_INTENSITY
  )
  scene.add(light, pose=light_pose_camera, parent_node=camera_node)

  renderer = pyrender.OffscreenRenderer(render_width, render_height)
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
