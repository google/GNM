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
arguments; call `render_gnm.render_gnm` or `GNMMeshRenderer` rather than
`render` directly.
"""

from collections.abc import Callable, Mapping
import dataclasses
import functools
import os
import tempfile
import types
from typing import Any, ClassVar, Self

import cv2
from gnm.shape import gnm_numpy
from gnm.shape import gnm_typing
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import render_common
import numpy as np
import numpy.typing as npt

if 'PYOPENGL_PLATFORM' not in os.environ:
  os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender  # pylint: disable=wrong-import-position,ungrouped-imports

enpt = gnm_typing.enpt

# Light direction for shading, in camera-space.

_LIGHT_DIRECTION = np.ones([3], dtype=np.float32)
_LIGHT_INTENSITY = 3.34  # Empirically chosen.

# Transparent black background for correct alpha blending.
_BLACK = (0.0, 0.0, 0.0, 0.0)

DEFAULT_RENDERER_FACTORY = pyrender.OffscreenRenderer


class ProjectionMatrixCamera(pyrender.Camera):
  """A camera defined by a full projection matrix."""

  def __init__(self, projection_matrix: npt.NDArray[np.floating]):
    """Initializes the camera with the given 4x4 projection matrix."""
    super().__init__()
    self._projection_matrix = projection_matrix

  def get_projection_matrix(
      self, width: int | None = None, height: int | None = None
  ) -> npt.NDArray[np.floating]:
    """Returns the 4x4 projection matrix."""
    del width, height
    return self._projection_matrix

  def set_projection_matrix(
      self, projection_matrix: npt.NDArray[np.floating]
  ) -> None:
    """Updates the 4x4 projection matrix."""
    self._projection_matrix = projection_matrix


@dataclasses.dataclass(kw_only=True)
class GNMMeshRenderer(render_common.BaseGNMMeshRenderer):
  """Stateful PyRender 3D mesh renderer for `gnm_numpy.GNM` models.

  Supports both OSMesa (`DEFAULT_RENDERER_FACTORY`) and SwiftShader
  (`gnm_pyrender_swiftshader.create_swiftshader_renderer`) backends.

  Attributes:
    ambient_light: Ambient light floor in [0, 1]. Defaults to 0.0.
    backface_culling: Whether to cull back-facing triangles. Defaults to True.
    renderer_factory: Factory callable `(viewport_width, viewport_height) ->
      pyrender.OffscreenRenderer`.
  """

  _progress_bar_desc: ClassVar[str] = 'Rendering (PyRender)'

  ambient_light: float = 0.0
  backface_culling: bool = True
  renderer_factory: Callable[[int, int], pyrender.OffscreenRenderer] = (
      DEFAULT_RENDERER_FACTORY
  )
  _cameras_in_opengl: bool = dataclasses.field(
      default=False, repr=False, compare=False
  )

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
      renderer_factory: Callable[
          [int, int], pyrender.OffscreenRenderer
      ] = DEFAULT_RENDERER_FACTORY,
  ) -> Self:
    """Instantiates a PyRender mesh renderer from a NumPy GNM shape model.

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
      renderer_factory: Factory callable `(viewport_width, viewport_height) ->
        pyrender.OffscreenRenderer`.

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
    renderer.renderer_factory = renderer_factory
    return renderer

  _scene: pyrender.Scene | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _mesh_nodes: tuple[pyrender.Node, ...] = dataclasses.field(
      default=(), init=False, repr=False, compare=False
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
  _textures: dict[int, pyrender.Texture] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _cached_vertices: enpt.FloatArray['... V 3'] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _cached_normals: enpt.FloatArray['... V 3'] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )

  def update_vertices(
      self,
      vertices: enpt.FloatArray['... V 3'],
      vertex_normals: enpt.FloatArray['... V 3'] | None = None,
      *,
      max_texture_dim: int = 1024,
  ) -> None:
    """Updates scene mesh vertices and normals in-place."""
    del max_texture_dim
    vertices, vertex_normals = self._prepare_vertices_and_normals(
        vertices, vertex_normals
    )
    if (
        self._scene is not None
        and self._mesh_nodes
        and self._cached_vertices is not None
        and self._cached_normals is not None
        and np.array_equal(self._cached_vertices, vertices)
        and np.array_equal(self._cached_normals, vertex_normals)
    ):
      return

    self._cached_vertices = vertices
    self._cached_normals = vertex_normals

    self._ensure_scene_initialized()
    self._rebuild_mesh_nodes(vertices, vertex_normals)

  def update_camera(
      self,
      world_to_camera: enpt.FloatArray['4 4'],
      camera_to_image: enpt.FloatArray['4 4'] | enpt.FloatArray['3 3'],
      image_size: tuple[int, int],
  ) -> None:
    """Updates the PyRender camera pose, projection matrix, and renderer."""
    world_to_camera = np.asarray(world_to_camera, dtype=np.float32)
    camera_to_image = np.asarray(camera_to_image, dtype=np.float32)

    if (
        self._renderer is not None
        and self._camera is not None
        and self._camera_node is not None
        and self._is_camera_cached(world_to_camera, camera_to_image, image_size)
    ):
      return

    self._ensure_scene_initialized()
    width, height = image_size
    render_width = width * self.multisample_antialiasing
    render_height = height * self.multisample_antialiasing

    if self._cameras_in_opengl:
      opengl_world_to_camera = world_to_camera
      opengl_camera_to_image = camera_to_image
    else:
      opengl_world_to_camera = camera_conversions.opencv_extrinsics_to_opengl(
          world_to_camera
      )
      opengl_camera_to_image = (
          camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
              camera_to_image,
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
      if self.include_shading:
        _override_shader_cache(renderer)
      self._renderer = renderer

    self._cache_camera(world_to_camera, camera_to_image, image_size)

  def close(self) -> None:
    """Releases persistent OpenGL context, scene, and cached state."""
    super().close()
    if self._renderer is not None:
      self._renderer.delete()
      self._renderer = None
    self._scene = None
    self._mesh_nodes = ()
    self._camera = None
    self._camera_node = None
    self._cached_vertices = None
    self._cached_normals = None
    self._textures.clear()

  def __enter__(self) -> Self:
    """Enters the runtime context related to this renderer."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    """Exits the runtime context and releases persistent OpenGL state."""
    del exc_type, exc_val, exc_tb
    self.close()

  def _update_frame_appearance(
      self,
      *,
      vertex_colors: enpt.FloatArray['... V 3'] | None = None,
      frame_textures: (
          Mapping[int, enpt.ui8['H_texture W_texture 3']] | None
      ) = None,
      max_texture_dim: int = 1024,
  ) -> None:
    """Updates active per-frame vertex colors and PyRender textures."""
    super()._update_frame_appearance(
        vertex_colors=vertex_colors,
        frame_textures=frame_textures,
        max_texture_dim=max_texture_dim,
    )
    if frame_textures is not None:
      for tile, texture_frame in frame_textures.items():
        self._textures[tile] = pyrender.Texture(
            source=texture_frame, source_channels='RGB'
        )
    if vertex_colors is not None or frame_textures is not None:
      self._cached_vertices = None
      self._cached_normals = None

  def _on_triangles_updated(self) -> None:
    """Rebuilds the PyRender mesh nodes after `self.triangles` changes."""
    if (
        self._scene is not None
        and self._cached_vertices is not None
        and self._cached_normals is not None
    ):
      self._rebuild_mesh_nodes(self._cached_vertices, self._cached_normals)

  def _render_single_frame(
      self,
      width: int,
      height: int,
  ) -> tuple[enpt.FloatArray['H W 3'], enpt.FloatArray['H W 1']]:
    """Renders the current PyRender scene for one frame.

    Args:
      width: Target output width in pixels.
      height: Target output height in pixels.

    Returns:
      A tuple of `(frame_rgb, frame_alpha)` as float32 arrays of shapes
      `(height, width, 3)` and `(height, width, 1)`.

    Raises:
      RuntimeError: If the PyRender scene, mesh nodes, or offscreen renderer is
        not initialized.
    """
    if self._scene is None or self._renderer is None or not self._mesh_nodes:
      raise RuntimeError(
          'PyRender scene, mesh nodes, and offscreen renderer must be'
          ' initialized before rendering.'
      )
    flags = (
        pyrender.constants.RenderFlags.NONE
        if self.include_shading
        else pyrender.constants.RenderFlags.FLAT
    )
    if not self.backface_culling:
      flags |= pyrender.constants.RenderFlags.SKIP_CULL_FACES
    color_srgb, depth = self._renderer.render(self._scene, flags=flags)
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

  def _ensure_scene_initialized(self) -> None:
    """Initializes PyRender scene, camera node, and light if needed."""
    if self._scene is not None:
      return
    scene = pyrender.Scene(
        bg_color=_BLACK, ambient_light=np.full(3, self.ambient_light)
    )
    camera = ProjectionMatrixCamera(np.eye(4, dtype=np.float32))
    camera_node = scene.add(camera, pose=np.eye(4, dtype=np.float32))
    light_direction_camera = _LIGHT_DIRECTION / np.linalg.norm(_LIGHT_DIRECTION)
    light_pose_camera = _align_vectors(
        np.array([0.0, 0.0, 1.0]), light_direction_camera
    )
    light = pyrender.DirectionalLight(
        color=np.ones(3),
        intensity=_LIGHT_INTENSITY * (1.0 - self.ambient_light),
    )
    scene.add(light, pose=light_pose_camera, parent_node=camera_node)
    self._scene = scene
    self._camera = camera
    self._camera_node = camera_node

  def _get_tile_texture(self, tile: int) -> pyrender.Texture:
    """Returns a cached `pyrender.Texture` for `tile`."""
    cached_texture = self._textures.get(tile)
    if cached_texture is not None:
      return cached_texture
    texture_frame = self._active_textures[tile]
    pyrender_texture = pyrender.Texture(
        source=texture_frame, source_channels='RGB'
    )
    self._textures[tile] = pyrender_texture
    return pyrender_texture

  def _rebuild_mesh_nodes(
      self,
      vertices: enpt.FloatArray['M V 3'],
      vertex_normals: enpt.FloatArray['M V 3'],
  ) -> None:
    """Rebuilds the PyRender mesh nodes from vertices and tile geometry."""
    if self._scene is None:
      return
    for existing_node in self._mesh_nodes:
      self._scene.remove_node(existing_node)

    num_meshes = int(vertices.shape[0])
    active_colors = self._active_vertex_colors
    if active_colors.ndim == 2:
      colors_per_mesh = np.broadcast_to(
          active_colors[np.newaxis, ...], (num_meshes, *active_colors.shape)
      )
    else:
      colors_per_mesh = active_colors

    new_nodes: list[pyrender.Node] = []
    for mesh_index in range(num_meshes):
      primitives: list[pyrender.Primitive] = []
      for tile, (corners, _, tile_uvs) in self._tile_geometry.items():
        if corners.size == 0:
          continue
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=1.0,
            baseColorTexture=self._get_tile_texture(tile),
        )
        primitives.append(
            pyrender.Primitive(
                positions=vertices[mesh_index][corners],
                normals=vertex_normals[mesh_index][corners],
                texcoord_0=tile_uvs,
                color_0=colors_per_mesh[mesh_index][corners],
                material=material,
            )
        )
      mesh = pyrender.Mesh(primitives=primitives)
      new_nodes.append(self._scene.add(mesh))
    self._mesh_nodes = tuple(new_nodes)


def render(
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
    ambient_light: float = 0.0,
    backface_culling: bool = True,
    verbose: bool = False,
    renderer_factory: Callable[
        [int, int], pyrender.OffscreenRenderer
    ] = DEFAULT_RENDERER_FACTORY,
) -> npt.NDArray[np.float32]:
  """Renders GNM meshes.

  N frames, M meshes.

  Args:
    vertices: The GNM vertices in world space, (N, M, V, 3).
    triangles: The GNM triangles to render, (F, 3).
    world_to_camera: The world-to-camera transform, (N, 4, 4).
    camera_to_image: The camera-to-image transform, (N, 4, 4).
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
    renderer_factory: Optional factory function returning a
      pyrender.OffscreenRenderer.

  Returns:
    The rendered color image, float32 [0-1] (N, H, W, 3).
  """
  # pylint: disable=unexpected-keyword-arg
  with GNMMeshRenderer(
      triangles=triangles,
      triangle_uvs=triangle_uvs,
      vertex_colors=vertex_colors,
      texture=texture,
      multisample_antialiasing=multisample_antialiasing,
      include_shading=include_shading,
      ambient_light=ambient_light,
      backface_culling=backface_culling,
      verbose=verbose,
      renderer_factory=renderer_factory,
      _cameras_in_opengl=True,
  ) as renderer:
    # pylint: enable=unexpected-keyword-arg
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
  return output.composite(
      background=0.0 if background_color is None else background_color,
      alpha=alpha,
      dtype=np.float32,
  )


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
          'clamp(((dot(n, v) < 0.0 ? -dot(n, l) : dot(n, l)) + 0.8) / 1.8, 0.0,'
          ' 1.0)',
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
    """Initializes the cache wrapper around `original_cache`."""
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
    """Returns a shader program with a custom fragment shader."""
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


def _override_shader_cache(renderer: pyrender.OffscreenRenderer) -> None:
  """Overrides the shader cache with a custom cache."""
  # pylint: disable=protected-access
  if r := renderer._renderer:
    cache = r._program_cache
    r._program_cache = CustomShaderCache(cache)
  # pylint: enable=protected-access


def _align_vectors(
    a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]
) -> npt.NDArray[np.float32]:
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
