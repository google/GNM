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

"""Tests for GNM Mitsuba visualization."""

# pylint: disable=protected-access

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import drjit as dr
from gnm.shape import gnm_numpy
from gnm.shape import gnm_test_utils
from gnm.shape import gnm_udim
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import gnm_mitsuba
from gnm.shape.visualization import render_common
import mitsuba as mi
import numpy as np
import numpy.typing as npt

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS


class GNMMitsubaTest(parameterized.TestCase):
  """Tests for GNM Mitsuba visualization."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    if not mi.variant():
      mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
    cls.gnms = {
        version: gnm_test_utils.load_gnm(
            gnm_numpy.GNM, version, gnm_numpy.GNMVariant.HEAD
        )
        for version in _MAINTAINED_MAJOR_GNM_VERSIONS
    }

  def setUp(self):
    super().setUp()

    self.world_to_camera = np.eye(4, dtype=np.float32)
    self.world_to_camera[1, 3] = -0.2
    self.world_to_camera[2, 3] = 2.0

    self.camera_to_image = np.array(
        [
            [500.0, 0.0, 120.0, 0.0],
            [0.0, 500.0, 160.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      num_frames=(1, 3),
  )
  def test_render_basic(self, version: str, num_frames: int):
    """Tests basic rendering with Mitsuba."""
    gnm_np = self.gnms[version]
    vertices = np.broadcast_to(
        gnm_np.template_vertex_positions,
        (num_frames, 1, *gnm_np.template_vertex_positions.shape),
    )

    color = gnm_mitsuba.render(
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
        image_size=(240, 320),
    )
    self.assertEqual(color.shape, (num_frames, 320, 240, 3))
    self.assertTrue((color >= 0.0).all() and (color <= 1.0).all())

  def test_create_mitsuba_mesh(self):
    """Tests that a Mitsuba mesh can be created from GNM data."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions
    faces = gnm_np.triangles
    normals = gnm_np.compute_vertex_normals(vertices)
    uvs = gnm_np.vertex_uvs
    colors = np.ones_like(vertices, dtype=np.float32)

    mesh = gnm_mitsuba.create_mitsuba_mesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        texture_coordinates=uvs.astype(np.float32),
        vertex_normals=normals.astype(np.float32),
        vertex_colors=colors,
    )
    self.assertEqual(mesh.vertex_count(), vertices.shape[0])
    self.assertEqual(mesh.face_count(), faces.shape[0])
    if gnm_mitsuba.MITSUBA_USE_LEGACY_MESH_API:
      self.assertTrue(
          mesh.has_vertex_normals()  # pyrefly: ignore[missing-attribute]
      )
      self.assertTrue(
          mesh.has_vertex_texcoords()  # pyrefly: ignore[missing-attribute]
      )
    else:
      self.assertTrue(mesh.has_normals())
      self.assertTrue(mesh.has_texcoords())
    self.assertTrue(mesh.has_attribute('vertex_colors'))

  def test_create_mitsuba_mesh_rgba_colors(self):
    """Tests that a Mitsuba mesh can be created with RGBA colors."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions
    faces = gnm_np.triangles
    rgba_colors = np.ones((vertices.shape[0], 4), dtype=np.float32)

    mesh = gnm_mitsuba.create_mitsuba_mesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_colors=rgba_colors,
    )
    self.assertTrue(mesh.has_attribute('vertex_colors'))

  def test_create_mitsuba_mesh_no_normals(self):
    """Tests that a Mitsuba mesh can be created without vertex normals."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions
    faces = gnm_np.triangles

    mesh = gnm_mitsuba.create_mitsuba_mesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_normals=None,
    )
    if gnm_mitsuba.MITSUBA_USE_LEGACY_MESH_API:
      self.assertFalse(
          mesh.has_vertex_normals()  # pyrefly: ignore[missing-attribute]
      )
    else:
      self.assertTrue(mesh.has_normals())

  def test_create_mitsuba_mesh_no_colors(self):
    """Tests that a Mitsuba mesh can be created without vertex colors."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions
    faces = gnm_np.triangles

    mesh = gnm_mitsuba.create_mitsuba_mesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_colors=None,
    )
    self.assertFalse(mesh.has_attribute('vertex_colors'))

  def test_render_partial_texture_dict(self):
    """Tests that missing tiles in texture dict fall back to default texture."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    # Only provide a texture for the skin tile, omitting the other tiles.
    partial_texture: dict[int, npt.NDArray[np.float32]] = {
        gnm_udim.UdimTile.SKIN: np.ones((1, 16, 16, 3), np.float32),
    }
    color = gnm_mitsuba.render(
        vertices=vertices,
        triangles=gnm_np.triangles,
        world_to_camera=self.world_to_camera[np.newaxis],
        camera_to_image=self.camera_to_image[np.newaxis],
        vertex_normals=gnm_np.compute_vertex_normals(vertices),
        triangle_uvs=gnm_np.triangle_uvs_udim,
        vertex_colors=np.ones_like(vertices),
        image_size=(240, 320),
        texture=partial_texture,
    )
    self.assertEqual(color.shape, (1, 320, 240, 3))

  def test_render_multi_frame_updates_in_place(self):
    """Tests that multi-frame rendering matches single-frame renders."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    base_vertices = gnm_np.template_vertex_positions
    num_frames = 3
    vertices = np.stack(
        [
            base_vertices[np.newaxis]
            + np.array(
                [0.02 * frame, -0.01 * frame, 0.0],
                dtype=np.float32,
            )
            for frame in range(num_frames)
        ],
        axis=0,
    )
    vertex_normals = gnm_np.compute_vertex_normals(vertices)

    world_to_camera = np.stack(
        [self.world_to_camera.copy() for _ in range(num_frames)],
        axis=0,
    )
    # Apply a small camera rotation on frame 1 and translation on frame 2.
    angle = 0.1
    cos_angle, sin_angle = float(np.cos(angle)), float(np.sin(angle))
    rotation_y = np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=np.float32,
    )
    world_to_camera[1, :3, :3] = rotation_y @ world_to_camera[1, :3, :3]
    world_to_camera[2, 0, 3] = 0.05

    camera_to_image = np.stack(
        [self.camera_to_image.copy() for _ in range(num_frames)],
        axis=0,
    )
    camera_to_image[2, 0, 0] = 520.0
    camera_to_image[2, 1, 1] = 520.0

    vertex_colors = np.ones_like(vertices, dtype=np.float32)
    vertex_colors[1] *= np.array([0.9, 0.8, 1.0], dtype=np.float32)
    vertex_colors[2] *= np.array([0.8, 1.0, 0.9], dtype=np.float32)

    skin_texture_first = np.full((8, 8, 3), 0.8, dtype=np.float32)
    skin_texture_last = np.full((8, 8, 3), 0.5, dtype=np.float32)
    texture: dict[int, npt.NDArray[np.float32]] = {
        # Frame 0 and Frame 1 share the same texture; Frame 2 changes texture.
        gnm_udim.UdimTile.SKIN: np.stack(
            [skin_texture_first, skin_texture_first, skin_texture_last], axis=0
        ),
    }

    batch_color = gnm_mitsuba.render(
        vertices=vertices,
        triangles=gnm_np.triangles,
        world_to_camera=world_to_camera,
        camera_to_image=camera_to_image,
        vertex_normals=vertex_normals,
        triangle_uvs=gnm_np.triangle_uvs_udim,
        vertex_colors=vertex_colors,
        image_size=(60, 80),
        texture=texture,
    )
    self.assertEqual(batch_color.shape, (num_frames, 80, 60, 3))

    for frame in range(num_frames):
      single_color = gnm_mitsuba.render(
          vertices=vertices[frame : frame + 1],
          triangles=gnm_np.triangles,
          world_to_camera=world_to_camera[frame : frame + 1],
          camera_to_image=camera_to_image[frame : frame + 1],
          vertex_normals=vertex_normals[frame : frame + 1],
          triangle_uvs=gnm_np.triangle_uvs_udim,
          vertex_colors=vertex_colors[frame : frame + 1],
          image_size=(60, 80),
          texture={
              gnm_udim.UdimTile.SKIN: texture[gnm_udim.UdimTile.SKIN][
                  frame : frame + 1
              ]
          },
      )
      np.testing.assert_allclose(
          batch_color[frame], single_color[0], rtol=1e-5, atol=1e-5
      )

  def test_non_square_pixels_raises_error(self):
    """Tests that non-square pixels raise an error."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    non_square_camera_to_image = self.camera_to_image.copy()
    non_square_camera_to_image[0, 0] = 500.0
    non_square_camera_to_image[1, 1] = 600.0
    with self.assertRaisesRegex(NotImplementedError, 'square pixels'):
      gnm_mitsuba.render(
          vertices=vertices,
          triangles=gnm_np.triangles,
          world_to_camera=self.world_to_camera[np.newaxis],
          camera_to_image=non_square_camera_to_image[np.newaxis],
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          triangle_uvs=gnm_np.triangle_uvs_udim,
          vertex_colors=np.ones_like(vertices),
          image_size=(240, 320),
      )

  def test_skew_raises_error(self):
    """Tests that skew raises an error."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    skew_camera_to_image = self.camera_to_image.copy()
    skew_camera_to_image[0, 1] = 10.0
    with self.assertRaisesRegex(NotImplementedError, 'zero skew'):
      gnm_mitsuba.render(
          vertices=vertices,
          triangles=gnm_np.triangles,
          world_to_camera=self.world_to_camera[np.newaxis],
          camera_to_image=skew_camera_to_image[np.newaxis],
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          triangle_uvs=gnm_np.triangle_uvs_udim,
          vertex_colors=np.ones_like(vertices),
          image_size=(240, 320),
      )

  def test_render_suspends_gradients(self):
    """Tests that mesh creation and rendering suspend gradient tracking."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    tracked_value = mi.Float(1.0)
    dr.enable_grad(tracked_value)
    with dr.resume_grad():
      mesh = gnm_mitsuba.create_mitsuba_mesh(
          vertices=gnm_np.template_vertex_positions.astype(np.float32),
          faces=gnm_np.triangles.astype(np.int32),
      )
      mesh_parameters = mi.traverse(mesh)
      for _, parameter in mesh_parameters.items():
        self.assertFalse(dr.grad_enabled(parameter))

      color = gnm_mitsuba.render(
          vertices=vertices,
          triangles=gnm_np.triangles,
          world_to_camera=self.world_to_camera[np.newaxis],
          camera_to_image=self.camera_to_image[np.newaxis],
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          triangle_uvs=gnm_np.triangle_uvs_udim,
          vertex_colors=np.ones_like(vertices),
          image_size=(60, 80),
      )
      self.assertEqual(color.shape, (1, 80, 60, 3))
      self.assertTrue(dr.grad_enabled(tracked_value))

  def test_prepare_texture_bitmap_scales_uint8(self):
    """Tests that uint8 textures are normalized from [0, 255] to [0.0, 1.0]."""
    uint8_texture = np.full((4, 4, 3), 51, dtype=np.uint8)

    prepared = gnm_mitsuba._prepare_texture_bitmap(
        texture_frame=uint8_texture,
        max_texture_dim=16,
    )
    np.testing.assert_allclose(np.array(prepared), 0.2)

  def test_prepare_texture_bitmap_clips_float_to_unit_range(self):
    """Tests that float textures outside [0.0, 1.0] are clipped."""
    float_texture = np.broadcast_to(
        np.array([-0.5, 0.5, 1.5], dtype=np.float32),
        (2, 2, 3),
    )

    prepared = gnm_mitsuba._prepare_texture_bitmap(
        texture_frame=float_texture,
        max_texture_dim=16,
    )
    np.testing.assert_allclose(
        np.array(prepared),
        np.broadcast_to(np.array([0.0, 0.5, 1.0], dtype=np.float32), (2, 2, 3)),
    )


class GNMMeshRendererTest(parameterized.TestCase):
  """Tests for the stateful GNMMeshRenderer Mitsuba backend."""

  gnm_np: gnm_numpy.GNM

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not mi.variant():
      mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
    version = _MAINTAINED_MAJOR_GNM_VERSIONS[0]
    cls.gnm_np = gnm_test_utils.load_gnm(
        gnm_numpy.GNM, version, gnm_numpy.GNMVariant.HEAD
    )

  def setUp(self):
    super().setUp()
    self.world_to_camera = np.eye(4, dtype=np.float32)
    self.world_to_camera[1, 3] = -0.2
    self.world_to_camera[2, 3] = 2.0

    self.camera_to_image = np.array(
        [
            [500.0, 0.0, 120.0, 0.0],
            [0.0, 500.0, 160.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

  def test_mock_scene_reuse_across_frames(self):
    """Tests that scene and sensor are reused across multiple frames."""
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    mock_loaded = mock.MagicMock()
    mock_loaded.integrator.return_value = None
    mock_params = mock.MagicMock()

    dummy_film = mi.TensorXf(np.ones((64, 64, 4), dtype=np.float32))
    num_active_tiles = sum(
        1
        for corners, _, _ in renderer._tile_geometry.values()
        if corners.size > 0
    )

    with (
        mock.patch.object(
            mi, 'load_dict', return_value=mock_loaded
        ) as mock_load,
        mock.patch.object(
            mi, 'traverse', return_value=mock_params
        ) as mock_traverse,
        mock.patch.object(mi, 'render', return_value=dummy_film) as mock_render,
    ):
      output = renderer.render(
          vertices=np.broadcast_to(
              self.gnm_np.template_vertex_positions,
              (3, *self.gnm_np.template_vertex_positions.shape),
          ),
          world_to_camera=np.broadcast_to(self.world_to_camera, (3, 4, 4)),
          camera_to_image=np.broadcast_to(self.camera_to_image, (3, 4, 4)),
          image_size=(64, 64),
      )
      self.assertEqual(output.image.shape, (3, 64, 64, 3))
      self.assertEqual(output.alpha.shape, (3, 64, 64, 1))

      # Sensor loaded once, BSDFs loaded once per active tile, scene loaded once
      self.assertEqual(mock_load.call_count, 1 + num_active_tiles + 1)
      self.assertEqual(mock_traverse.call_count, 1)
      self.assertEqual(mock_params.update.call_count, 2)
      self.assertEqual(mock_render.call_count, 3)

  def test_real_render_multi_frame(self):
    """Tests real Mitsuba rendering on multiple frames."""
    self.assertEqual(
        gnm_mitsuba.GNMMeshRenderer._progress_bar_desc, 'Rendering (Mitsuba)'
    )
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    output = renderer.render(
        vertices=np.broadcast_to(
            self.gnm_np.template_vertex_positions,
            (2, *self.gnm_np.template_vertex_positions.shape),
        ),
        world_to_camera=np.broadcast_to(self.world_to_camera, (2, 4, 4)),
        camera_to_image=np.broadcast_to(self.camera_to_image, (2, 4, 4)),
        image_size=(64, 64),
    )
    self.assertIsInstance(output, render_common.MeshRenderOutput)
    self.assertEqual(output.image.shape, (2, 64, 64, 3))
    self.assertEqual(output.alpha.shape, (2, 64, 64, 1))
    self.assertEqual(output.image.dtype, np.uint8)
    self.assertEqual(output.alpha.dtype, np.float32)

  def test_update_camera_in_place_and_reload_on_resize(self):
    """Tests that update_camera updates in-place or reloads on size change."""
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    # Initial camera load at 64x64
    renderer.update_camera(
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(64, 64),
    )
    initial_sensor = renderer._sensor
    self.assertIsNotNone(initial_sensor)
    self.assertIsNone(renderer._sensor_params)

    # In-place camera update with same image size preserves sensor object
    modified_w2c = self.world_to_camera.copy()
    modified_w2c[0, 3] = 0.1
    renderer.update_camera(
        world_to_camera=modified_w2c,
        camera_to_image=self.camera_to_image,
        image_size=(64, 64),
    )
    self.assertIs(renderer._sensor, initial_sensor)
    self.assertIsNotNone(renderer._sensor_params)

    # Changing image size reloads sensor
    renderer.update_camera(
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
    )
    self.assertIsNot(renderer._sensor, initial_sensor)
    self.assertIsNone(renderer._sensor_params)
    self.assertEqual(renderer._cached_image_size, (32, 32))
    renderer.close()

  def test_update_frame_appearance_in_place(self):
    """Tests that _update_frame_appearance updates scene parameters in-place."""
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    renderer.render(
        vertices=self.gnm_np.template_vertex_positions,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
        vertex_colors=np.ones_like(self.gnm_np.template_vertex_positions),
    )
    initial_scene = renderer._scene
    self.assertIsNotNone(initial_scene)

    updated_colors = np.ones_like(self.gnm_np.template_vertex_positions) * 0.5
    updated_textures = {
        int(render_common.SKIN_UDIM_TILE): np.full(
            (8, 8, 3), 128, dtype=np.uint8
        )
    }
    renderer._update_frame_appearance(
        vertex_colors=updated_colors,
        frame_textures=updated_textures,
    )
    # Scene should NOT be invalidated or reset to None
    self.assertIs(renderer._scene, initial_scene)
    self.assertIsNotNone(renderer._scene_params)
    renderer.close()

  @parameterized.named_parameters(
      ('modern_api', False),
      ('legacy_api', True),
  )
  def test_update_frame_appearance_legacy_and_modern_api(
      self, use_legacy_api: bool
  ):
    """Tests _update_frame_appearance under both modern and legacy APIs."""
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    renderer.render(
        vertices=self.gnm_np.template_vertex_positions,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
        vertex_colors=np.ones_like(self.gnm_np.template_vertex_positions),
    )
    initial_scene = renderer._scene
    updated_colors = np.ones_like(self.gnm_np.template_vertex_positions) * 0.7

    mock_scene_params = mock.MagicMock()
    renderer._scene_params = mock_scene_params
    with mock.patch.object(
        gnm_mitsuba, 'MITSUBA_USE_LEGACY_MESH_API', use_legacy_api
    ):
      renderer._update_frame_appearance(
          vertex_colors=updated_colors,
      )
    mock_scene_params.update.assert_called_once()
    self.assertIs(renderer._scene, initial_scene)
    renderer.close()

  def test_update_triangles(self):
    """Tests updating triangles: no-op, in-place, and invalidation."""
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    no_eye_exteriors, no_eye_exteriors_uvs = (
        render_common.resolve_gnm_triangles(self.gnm_np, '~eye_exteriors')
    )
    no_teeth, no_teeth_uvs = render_common.resolve_gnm_triangles(
        self.gnm_np, ('~eye_exteriors', '-teeth')
    )
    # Updating triangles before scene initialization returns early cleanly.
    renderer.update_triangles(no_teeth, no_teeth_uvs)
    self.assertIsNone(renderer._scene)

    # Render once to initialize scene and sensor.
    renderer.render(
        vertices=self.gnm_np.template_vertex_positions,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
        triangles=no_eye_exteriors,
        triangle_uvs=no_eye_exteriors_uvs,
    )
    initial_scene = renderer._scene
    self.assertIsNotNone(initial_scene)

    # Identical triangles should be a no-op and preserve scene.
    renderer.update_triangles(no_eye_exteriors, no_eye_exteriors_uvs)
    self.assertIs(renderer._scene, initial_scene)

    # Updating triangles invalidates the scene so it gets reconstructed.
    renderer.update_triangles(no_teeth, no_teeth_uvs)
    np.testing.assert_array_equal(renderer.triangles, no_teeth)
    np.testing.assert_array_equal(renderer.triangle_uvs, no_teeth_uvs)
    self.assertIsNone(renderer._scene)
    self.assertIsNone(renderer._scene_params)

  def test_render_uninitialized_raises_error(self):
    """Tests that _render_single_frame raises RuntimeError before init."""
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    with self.assertRaises(RuntimeError):
      renderer._render_single_frame(width=32, height=32)

  def test_render_msaa_and_custom_textures(self):
    """Tests rendering with MSAA, no texture, and 4D texture dictionary."""
    renderer_no_texture = gnm_mitsuba.GNMMeshRenderer.from_gnm(
        self.gnm_np,
        texture=None,
        multisample_antialiasing=2,
        spp=1,
    )
    renderer_no_texture.update_vertices(self.gnm_np.template_vertex_positions)
    output_no_texture = renderer_no_texture.render(
        vertices=self.gnm_np.template_vertex_positions,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
    )
    self.assertEqual(output_no_texture.image.shape, (32, 32, 3))
    renderer_no_texture.close()

    batched_texture = {
        render_common.SKIN_UDIM_TILE: np.ones((1, 8, 8, 3), dtype=np.float32)
    }
    renderer_batched_texture = gnm_mitsuba.GNMMeshRenderer.from_gnm(
        self.gnm_np,
        texture=batched_texture,
        spp=1,
    )
    output_batched_texture = renderer_batched_texture.render(
        vertices=self.gnm_np.template_vertex_positions,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
    )
    self.assertEqual(output_batched_texture.image.shape, (1, 32, 32, 3))
    renderer_batched_texture.close()

  def test_close_clears_resources(self):
    """Tests that close() clears cached scene, sensor, and params."""
    renderer = gnm_mitsuba.GNMMeshRenderer.from_gnm(self.gnm_np, spp=1)
    renderer.render(
        vertices=self.gnm_np.template_vertex_positions,
        world_to_camera=self.world_to_camera,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
    )
    modified_w2c = self.world_to_camera.copy()
    modified_w2c[0, 3] = 0.2
    renderer.update_camera(
        world_to_camera=modified_w2c,
        camera_to_image=self.camera_to_image,
        image_size=(32, 32),
    )
    self.assertIsNotNone(renderer._scene)
    self.assertIsNotNone(renderer._sensor)
    self.assertIsNotNone(renderer._sensor_params)

    renderer.close()
    self.assertIsNone(renderer._scene)
    self.assertIsNone(renderer._scene_params)
    self.assertIsNone(renderer._sensor)
    self.assertIsNone(renderer._sensor_params)
    self.assertIsNone(renderer._light_direction)
    self.assertIsNone(renderer._cached_world_to_camera)
    self.assertIsNone(renderer._cached_camera_to_image)
    self.assertIsNone(renderer._cached_image_size)
    self.assertEmpty(renderer._cached_bitmaps)
    self.assertEqual(renderer._mesh_keys, ())
    self.assertEqual(renderer._active_tiles, ())


if __name__ == '__main__':
  absltest.main()
