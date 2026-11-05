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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import drjit as dr
from gnm.shape import gnm_numpy
from gnm.shape import gnm_test_utils
from gnm.shape import gnm_udim
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import gnm_mitsuba
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
      self.assertTrue(
          mesh.has_normals()
      )
      self.assertTrue(
          mesh.has_texcoords()
      )
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
      self.assertTrue(
          mesh.has_normals()
      )

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
              gnm_udim.UdimTile.SKIN: (
                  texture[gnm_udim.UdimTile.SKIN][frame : frame + 1]
              )
          },
      )
      np.testing.assert_allclose(
          batch_color[frame], single_color[0], rtol=1e-5, atol=1e-5
      )

  def test_non_square_pixels_raises_error(self):
    """Tests that non-square pixels raise an error."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    non_square_c2i = self.camera_to_image.copy()
    non_square_c2i[0, 0] = 500.0
    non_square_c2i[1, 1] = 600.0
    with self.assertRaisesRegex(NotImplementedError, 'square pixels'):
      gnm_mitsuba.render(
          vertices=vertices,
          triangles=gnm_np.triangles,
          world_to_camera=self.world_to_camera[np.newaxis],
          camera_to_image=non_square_c2i[np.newaxis],
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          triangle_uvs=gnm_np.triangle_uvs_udim,
          vertex_colors=np.ones_like(vertices),
          image_size=(240, 320),
      )

  def test_skew_raises_error(self):
    """Tests that skew raises an error."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    skew_c2i = self.camera_to_image.copy()
    skew_c2i[0, 1] = 10.0
    with self.assertRaisesRegex(NotImplementedError, 'zero skew'):
      gnm_mitsuba.render(
          vertices=vertices,
          triangles=gnm_np.triangles,
          world_to_camera=self.world_to_camera[np.newaxis],
          camera_to_image=skew_c2i[np.newaxis],
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          triangle_uvs=gnm_np.triangle_uvs_udim,
          vertex_colors=np.ones_like(vertices),
          image_size=(240, 320),
      )

  def test_render_reuses_sensor_of_image_size(self):
    """Tests that a reused sensor renders like a newly created one."""
    self.enter_context(
        mock.patch.dict(gnm_mitsuba._SENSORS, clear=True)  # pylint: disable=protected-access
    )
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[np.newaxis, np.newaxis]
    moved_world_to_camera = self.world_to_camera.copy()
    moved_world_to_camera[0, 3] = 0.05

    def render(world_to_camera: npt.NDArray[np.float32]) -> np.ndarray:
      return gnm_mitsuba.render(
          vertices=vertices,
          triangles=gnm_np.triangles,
          world_to_camera=world_to_camera[np.newaxis],
          camera_to_image=self.camera_to_image[np.newaxis],
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          triangle_uvs=gnm_np.triangle_uvs_udim,
          vertex_colors=np.ones_like(vertices),
          image_size=(60, 80),
      )

    render(self.world_to_camera)
    reused = render(moved_world_to_camera)
    self.assertLen(gnm_mitsuba._SENSORS, 1)  # pylint: disable=protected-access
    gnm_mitsuba._SENSORS.clear()  # pylint: disable=protected-access
    created = render(moved_world_to_camera)
    np.testing.assert_allclose(reused, created, rtol=1e-5, atol=1e-5)

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

  def test_prepare_tile_textures_scales_uint8(self):
    """Tests that uint8 textures are normalized from [0, 255] to [0.0, 1.0]."""
    uint8_texture = np.full((1, 4, 4, 3), 51, dtype=np.uint8)
    tile = gnm_udim.UdimTile.SKIN

    prepared = gnm_mitsuba._prepare_tile_textures(  # pylint: disable=protected-access
        texture={tile: uint8_texture},
        tiles=[tile],
        num_frames=1,
        max_texture_dimension=16,
    )
    np.testing.assert_allclose(prepared[tile][0], 0.2)

  def test_prepare_tile_textures_clips_float_to_unit_range(self):
    """Tests that float textures outside [0.0, 1.0] are clipped."""
    float_texture = np.broadcast_to(
        np.array([-0.5, 0.5, 1.5], dtype=np.float32),
        (1, 2, 2, 3),
    )
    tile = gnm_udim.UdimTile.SKIN

    prepared = gnm_mitsuba._prepare_tile_textures(  # pylint: disable=protected-access
        texture={tile: float_texture},
        tiles=[tile],
        num_frames=1,
        max_texture_dimension=16,
    )
    np.testing.assert_allclose(
        prepared[tile][0],
        np.broadcast_to(np.array([0.0, 0.5, 1.0], dtype=np.float32), (2, 2, 3)),
    )


if __name__ == '__main__':
  absltest.main()
