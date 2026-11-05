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

"""Tests for GNM pyrender visualization."""

# pylint: disable=protected-access

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from gnm.shape import gnm_numpy
from gnm.shape import gnm_test_utils
from gnm.shape import gnm_udim
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import gnm_pyrender
from gnm.shape.visualization import render_common
import mediapy as media
import numpy as np
import numpy.typing as npt

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS

_OUTPUTS_TMPDIR = tempfile.TemporaryDirectory()
_OUTPUTS_DIR = epath.Path(_OUTPUTS_TMPDIR.name)


class GNMPyrenderTest(parameterized.TestCase):

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {
        version: gnm_test_utils.load_gnm(
            gnm_numpy.GNM, version, gnm_numpy.GNMVariant.HEAD
        )
        for version in _MAINTAINED_MAJOR_GNM_VERSIONS
    }

  def setUp(self):
    super().setUp()

    self.world_to_camera = np.eye(4)
    self.world_to_camera[1, 3] = -0.2
    self.world_to_camera[2, 3] = -2.0

    self.camera_to_image = np.eye(4)
    self.camera_to_image[0, 0] = 15
    self.camera_to_image[1, 1] = 10
    self.camera_to_image[2:4, 2] = -1.0

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      num_frames=(1, 5),
  )
  def test_render_basic(self, version: str, num_frames: int):
    gnm_np = self.gnms[version]
    vertices = np.broadcast_to(
        gnm_np.template_vertex_positions,
        (num_frames, 1, *gnm_np.template_vertex_positions.shape),
    )

    color = gnm_pyrender.render(
        vertices=vertices,
        triangles=gnm_np.triangles,
        world_to_camera=np.broadcast_to(
            self.world_to_camera, (num_frames, 4, 4)
        ),
        camera_to_image=np.broadcast_to(
            self.camera_to_image, (num_frames, 4, 4)
        ),
        vertex_normals=gnm_np.compute_vertex_normals(vertices),
        triangle_uvs=gnm_np.triangle_uvs_udim,
        vertex_colors=np.ones_like(vertices),
    )
    self.assertEqual(color.shape, (num_frames, 320, 240, 3))
    if num_frames == 1:
      media.write_image(_OUTPUTS_DIR / 'render_basic.png', color[0])

  def test_render_clips_out_of_range_float_texture(self):
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    clipped_texture: dict[int, npt.NDArray[np.float32]] = {
        gnm_udim.UdimTile.SKIN: np.ones((1, 4, 4, 3), dtype=np.float32)
    }
    over_one_texture: dict[int, npt.NDArray[np.float32]] = {
        gnm_udim.UdimTile.SKIN: np.full((1, 4, 4, 3), 1.5, dtype=np.float32)
    }
    expected_color = gnm_pyrender.render(
        vertices=vertices,
        triangles=gnm_np.triangles,
        world_to_camera=self.world_to_camera[np.newaxis],
        camera_to_image=self.camera_to_image[np.newaxis],
        vertex_normals=gnm_np.compute_vertex_normals(vertices),
        triangle_uvs=gnm_np.triangle_uvs_udim,
        vertex_colors=np.ones_like(vertices),
        texture=clipped_texture,
    )
    actual_color = gnm_pyrender.render(
        vertices=vertices,
        triangles=gnm_np.triangles,
        world_to_camera=self.world_to_camera[np.newaxis],
        camera_to_image=self.camera_to_image[np.newaxis],
        vertex_normals=gnm_np.compute_vertex_normals(vertices),
        triangle_uvs=gnm_np.triangle_uvs_udim,
        vertex_colors=np.ones_like(vertices),
        texture=over_one_texture,
    )
    np.testing.assert_allclose(actual_color, expected_color)


class ProjectionMatrixCameraTest(parameterized.TestCase):

  @parameterized.parameters((np.eye(4),), (np.zeros((4, 4)),))
  def test_projection_matrix_camera(
      self, projection_matrix: npt.NDArray[np.floating]
  ):
    camera = gnm_pyrender.ProjectionMatrixCamera(projection_matrix)
    np.testing.assert_array_equal(
        camera.get_projection_matrix(), projection_matrix
    )

  @parameterized.parameters((np.eye(4),), (np.zeros((4, 4)),))
  def test_set_projection_matrix(
      self, projection_matrix: npt.NDArray[np.floating]
  ):
    camera = gnm_pyrender.ProjectionMatrixCamera(np.ones((4, 4)))
    camera.set_projection_matrix(projection_matrix)
    np.testing.assert_array_equal(
        camera.get_projection_matrix(), projection_matrix
    )


class GNMMeshRendererTest(parameterized.TestCase):

  gnm_np: gnm_numpy.GNM

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnm_np = gnm_test_utils.load_gnm(
        gnm_numpy.GNM,
        _MAINTAINED_MAJOR_GNM_VERSIONS[0],
        gnm_numpy.GNMVariant.HEAD,
    )

  def setUp(self):
    super().setUp()
    self.image_size = (160, 120)
    self.vertices = self.gnm_np.template_vertex_positions
    self.world_to_camera = render_common.get_look_at_world_to_camera(
        self.gnm_np, self.vertices
    )
    self.camera_to_image = render_common.get_fill_factor_camera_to_image(
        self.gnm_np, self.vertices, image_size=self.image_size
    )

  def test_render_single_frame(self):
    self.assertEqual(
        gnm_pyrender.GNMMeshRenderer._progress_bar_desc, 'Rendering (PyRender)'
    )
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(
        self.gnm_np,
        triangles='~eye_exteriors',
    )
    output = renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    self.assertIsInstance(output, render_common.MeshRenderOutput)
    self.assertEqual(output.image.shape, (120, 160, 3))
    self.assertEqual(output.alpha.shape, (120, 160, 1))
    self.assertEqual(output.image.dtype, np.uint8)
    self.assertEqual(output.alpha.dtype, np.float32)
    renderer.close()
    self.assertIsNone(renderer._renderer)
    self.assertIsNone(renderer._scene)

  def test_render_multi_frame(self):
    num_frames = 3
    batched_vertices = np.broadcast_to(
        self.vertices,
        (num_frames, self.gnm_np.num_vertices, 3),
    )
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    output = renderer.render(
        vertices=batched_vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    self.assertEqual(output.image.shape, (num_frames, 120, 160, 3))
    self.assertEqual(output.alpha.shape, (num_frames, 120, 160, 1))
    renderer.close()

  def test_reuse_offscreen_renderer(self):
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    first_renderer_instance = renderer._renderer
    self.assertIsNotNone(first_renderer_instance)

    # Rendering second frame with identical image size reuses the renderer.
    renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    self.assertIs(renderer._renderer, first_renderer_instance)

    # Changing image size creates a new renderer.
    camera_to_image_80_60 = render_common.get_fill_factor_camera_to_image(
        self.gnm_np, self.vertices, image_size=(80, 60)
    )
    renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=camera_to_image_80_60,
        image_size=(80, 60),
    )
    self.assertIsNot(renderer._renderer, first_renderer_instance)
    renderer.close()
    self.assertIsNone(renderer._renderer)

  def test_update_triangles(self):
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    # Updating triangles before scene initialization updates state cleanly.
    skin_triangles, skin_uvs = render_common.resolve_gnm_triangles(
        self.gnm_np, 'skin'
    )
    renderer.update_triangles(skin_triangles, skin_uvs)
    np.testing.assert_array_equal(renderer.triangles, skin_triangles)
    np.testing.assert_allclose(renderer.triangle_uvs, skin_uvs)
    output = renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    self.assertEqual(output.image.shape, (120, 160, 3))

    # Updating triangles after scene initialization rebuilds mesh node.
    sequence_triangles, sequence_uvs = render_common.resolve_gnm_triangles(
        self.gnm_np, ('~eye_exteriors', '-teeth')
    )
    renderer.update_triangles(sequence_triangles, sequence_uvs)
    np.testing.assert_array_equal(renderer.triangles, sequence_triangles)
    np.testing.assert_allclose(renderer.triangle_uvs, sequence_uvs)
    output_sequence = renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    self.assertEqual(output_sequence.image.shape, (120, 160, 3))
    renderer.close()

  def test_render_uninitialized_raises_error(self):
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    with self.assertRaises(RuntimeError):
      renderer._render_single_frame(width=64, height=64)
    renderer.close()

  def test_render_msaa_and_custom_textures(self):
    camera_to_image_64_48 = render_common.get_fill_factor_camera_to_image(
        self.gnm_np, self.vertices, image_size=(64, 48)
    )
    renderer_no_texture = gnm_pyrender.GNMMeshRenderer.from_gnm(
        self.gnm_np,
        texture=None,
        multisample_antialiasing=2,
    )
    output_no_texture = renderer_no_texture.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=camera_to_image_64_48,
        image_size=(64, 48),
    )
    self.assertEqual(output_no_texture.image.shape, (48, 64, 3))
    renderer_no_texture.close()

    batched_texture = {
        render_common.SKIN_UDIM_TILE: np.ones((1, 8, 8, 3), dtype=np.float32)
    }
    renderer_batched_texture = gnm_pyrender.GNMMeshRenderer.from_gnm(
        self.gnm_np,
        texture=batched_texture,
    )
    output_batched_texture = renderer_batched_texture.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=camera_to_image_64_48,
        image_size=(64, 48),
    )
    self.assertEqual(output_batched_texture.image.shape, (1, 48, 64, 3))
    renderer_batched_texture.close()

  def test_update_camera(self):
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    world_to_camera = np.eye(4, dtype=np.float32)
    camera_to_image = np.eye(4, dtype=np.float32)
    renderer.update_camera(
        world_to_camera=world_to_camera,
        camera_to_image=camera_to_image,
        image_size=self.image_size,
    )
    self.assertEqual(renderer._cached_image_size, self.image_size)
    renderer.close()

  def test_update_vertices(self):
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    renderer.update_vertices(vertices=self.gnm_np.template_vertex_positions)
    self.assertIsNotNone(renderer._cached_vertices)
    self.assertIsNotNone(renderer._cached_normals)
    renderer.close()

  def test_update_vertices_caching_and_invalidation(self):
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    self.assertTrue(renderer._mesh_nodes)
    first_node = renderer._mesh_nodes[0]

    # Calling update_vertices with identical vertices and normals reuses nodes.
    renderer.update_vertices(vertices=self.vertices)
    self.assertIs(renderer._mesh_nodes[0], first_node)

    # Calling _update_frame_appearance with vertex colors invalidates cache.
    new_colors = np.ones_like(renderer._active_vertex_colors)
    renderer._update_frame_appearance(vertex_colors=new_colors)
    self.assertIsNone(renderer._cached_vertices)
    self.assertIsNone(renderer._cached_normals)

    renderer.update_vertices(vertices=self.vertices)
    self.assertIsNot(renderer._mesh_nodes[0], first_node)
    second_node = renderer._mesh_nodes[0]

    # Calling _update_frame_appearance with frame textures invalidates cache.
    frame_texture = {
        int(render_common.SKIN_UDIM_TILE): np.ones((8, 8, 3), dtype=np.uint8)
    }
    renderer._update_frame_appearance(frame_textures=frame_texture)
    self.assertIsNone(renderer._cached_vertices)
    self.assertIsNone(renderer._cached_normals)

    renderer.update_vertices(vertices=self.vertices)
    self.assertIsNot(renderer._mesh_nodes[0], second_node)
    renderer.close()

  def test_close_resets_cached_state(self):
    with gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np) as renderer:
      renderer.render(
          vertices=self.vertices,
          world_to_camera=self.world_to_camera,
          camera_to_image=self.camera_to_image,
          image_size=self.image_size,
      )
      self.assertIsNotNone(renderer._cached_vertices)
      self.assertIsNotNone(renderer._cached_normals)
      self.assertIsNotNone(renderer._cached_world_to_camera)
      self.assertIsNotNone(renderer._cached_camera_to_image)
      self.assertIsNotNone(renderer._cached_image_size)

    self.assertIsNone(renderer._cached_vertices)
    self.assertIsNone(renderer._cached_normals)
    self.assertIsNone(renderer._cached_world_to_camera)
    self.assertIsNone(renderer._cached_camera_to_image)
    self.assertIsNone(renderer._cached_image_size)
    self.assertIsNone(renderer._renderer)
    self.assertIsNone(renderer._scene)

  def test_composite(self):
    renderer = gnm_pyrender.GNMMeshRenderer.from_gnm(self.gnm_np)
    output = renderer.render(
        vertices=self.vertices,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=self.image_size,
    )
    composited_default = output.composite()
    self.assertEqual(composited_default.shape, (120, 160, 3))
    self.assertEqual(composited_default.dtype, np.uint8)

    background_image = np.full((120, 160, 3), 128, dtype=np.uint8)
    composited_custom = output.composite(background=background_image, alpha=0.8)
    self.assertEqual(composited_custom.shape, (120, 160, 3))
    self.assertEqual(composited_custom.dtype, np.uint8)
    renderer.close()


if __name__ == '__main__':
  absltest.main()
