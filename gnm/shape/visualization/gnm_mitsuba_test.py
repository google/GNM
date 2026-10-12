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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
from gnm.shape import gnm_test_utils
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

  def broadcast(
      self, array: npt.NDArray[Any], num_frames: int = 1
  ) -> npt.NDArray[Any]:
    return np.broadcast_to(array, (num_frames, *array.shape))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      num_frames=(1, 3),
  )
  def test_render_basic(self, version: str, num_frames: int):
    """Tests basic rendering with Mitsuba."""
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions[None, :, :]
    vertices = self.broadcast(vertices, num_frames)

    triangles = {
        component_name: gnm_np.triangles_group(component_name)
        for component_name in gnm_np.mesh_component_names
    }

    color = gnm_mitsuba.render(
        vertices=vertices,
        triangles=triangles,
        world_to_camera=self.broadcast(self.world_to_camera, num_frames),
        camera_to_image=self.broadcast(self.camera_to_image, num_frames),
        vertex_normals=gnm_np.compute_vertex_normals(vertices),
        vertex_uvs=gnm_np.vertex_uvs,
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
    """Tests that missing parts in texture dict fall back to default texture."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[None, None, ...]
    triangles = {
        component_name: gnm_np.triangles_group(component_name)
        for component_name in gnm_np.mesh_component_names
    }
    # Only provide texture for 'skin', omitting other parts like eye_exteriors
    partial_texture = {
        'skin': np.ones((1, 16, 16, 3), dtype=np.float32),
    }
    color = gnm_mitsuba.render(
        vertices=vertices,
        triangles=triangles,
        world_to_camera=self.broadcast(self.world_to_camera, 1),
        camera_to_image=self.broadcast(self.camera_to_image, 1),
        vertex_normals=gnm_np.compute_vertex_normals(vertices),
        vertex_uvs=gnm_np.vertex_uvs,
        vertex_colors=np.ones_like(vertices),
        image_size=(240, 320),
        texture=partial_texture,
    )
    self.assertEqual(color.shape, (1, 320, 240, 3))

  def test_non_square_pixels_raises_error(self):
    """Tests that non-square pixels raise an error."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[None, None, ...]
    triangles = {'skin': gnm_np.triangles}
    non_square_c2i = self.camera_to_image.copy()
    non_square_c2i[0, 0] = 500.0
    non_square_c2i[1, 1] = 600.0
    with self.assertRaisesRegex(NotImplementedError, 'square pixels'):
      gnm_mitsuba.render(
          vertices=vertices,
          triangles=triangles,
          world_to_camera=self.broadcast(self.world_to_camera, 1),
          camera_to_image=self.broadcast(non_square_c2i, 1),
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          vertex_uvs=gnm_np.vertex_uvs,
          vertex_colors=np.ones_like(vertices),
          image_size=(240, 320),
      )

  def test_skew_raises_error(self):
    """Tests that skew raises an error."""
    gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    vertices = gnm_np.template_vertex_positions[None, None, ...]
    triangles = {'skin': gnm_np.triangles}
    skew_c2i = self.camera_to_image.copy()
    skew_c2i[0, 1] = 10.0
    with self.assertRaisesRegex(NotImplementedError, 'zero skew'):
      gnm_mitsuba.render(
          vertices=vertices,
          triangles=triangles,
          world_to_camera=self.broadcast(self.world_to_camera, 1),
          camera_to_image=self.broadcast(skew_c2i, 1),
          vertex_normals=gnm_np.compute_vertex_normals(vertices),
          vertex_uvs=gnm_np.vertex_uvs,
          vertex_colors=np.ones_like(vertices),
          image_size=(240, 320),
      )


if __name__ == '__main__':
  absltest.main()
