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
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import gnm_pyrender
from gnm.shape.visualization import render_common
import mediapy as media
import numpy as np

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_test_catalog.MAJOR_VERSION_TO_VARIANTS_MAP

_OUTPUTS_TMPDIR = tempfile.TemporaryDirectory()
_OUTPUTS_DIR = epath.Path(_OUTPUTS_TMPDIR.name)


class GNMPyrenderTest(parameterized.TestCase):

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {
        version: gnm_numpy.GNM.from_remote(
            gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
            gnm_numpy.GNMVariant.HEAD,
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

  def broadcast(self, array: np.ndarray, num_frames: int = 1) -> np.ndarray:
    return np.broadcast_to(array, (num_frames, *array.shape))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      num_frames=(1, 5),
  )
  def test_render_basic(self, version: str, num_frames: int):
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions[None, :, :]
    vertices = self.broadcast(vertices, num_frames)

    triangles = {
        component_name: gnm_np.triangles_group(component_name)
        for component_name in gnm_np.mesh_component_names
    }

    color = gnm_pyrender.render(
        vertices=vertices,
        triangles=triangles,
        world_to_camera=self.broadcast(self.world_to_camera, num_frames),
        camera_to_image=self.broadcast(self.camera_to_image, num_frames),
        vertex_normals=gnm_np.compute_vertex_normals(vertices),
        vertex_uvs=gnm_np.vertex_uvs,
        vertex_colors=np.ones_like(vertices),
    )
    self.assertEqual(color.shape, (num_frames, 320, 240, 3))
    if num_frames == 1:
      media.write_image(_OUTPUTS_DIR / 'render_basic.png', color[0])


class ProjectionMatrixCameraTest(parameterized.TestCase):

  @parameterized.parameters((np.eye(4),), (np.zeros((4, 4)),))
  def test_projection_matrix_camera(self, projection_matrix: np.ndarray):
    camera = gnm_pyrender.ProjectionMatrixCamera(projection_matrix)
    np.testing.assert_array_equal(
        camera.get_projection_matrix(), projection_matrix
    )

  @parameterized.parameters((np.eye(4),), (np.zeros((4, 4)),))
  def test_set_projection_matrix(self, projection_matrix: np.ndarray):
    camera = gnm_pyrender.ProjectionMatrixCamera(np.ones((4, 4)))
    camera.set_projection_matrix(projection_matrix)
    np.testing.assert_array_equal(
        camera.get_projection_matrix(), projection_matrix
    )


class ChangedFlagTest(absltest.TestCase):

  def test_changed_flag(self):
    num_frames = 25
    indices_to_change = np.array([10, 16, 21])

    expected_flags = np.zeros(num_frames, dtype=np.bool_)
    expected_flags[0] = True
    expected_flags[indices_to_change] = True
    expected_flags[indices_to_change + 1] = True

    array = np.zeros((num_frames, 100, 100, 3), dtype=np.float32)
    for index in indices_to_change:
      array[index, 0, 0] = 1.0

    np.testing.assert_array_equal(
        gnm_pyrender._changed_flag(array), expected_flags
    )

  def test_changed_flag_broadcast_unchanged(self):
    num_frames = 25
    array = np.zeros((10, 10, 3), dtype=np.float32)
    array = np.broadcast_to(array, (num_frames, *array.shape))

    # All false except the first frame.
    expected_flags = np.zeros(25, dtype=np.bool_)
    expected_flags[0] = True

    np.testing.assert_array_equal(
        gnm_pyrender._changed_flag(array), expected_flags
    )


class GNMMeshRendererTest(parameterized.TestCase):

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {
        version: gnm_numpy.GNM.from_remote(
            gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
            gnm_numpy.GNMVariant.HEAD,
        )
        for version in _MAINTAINED_MAJOR_GNM_VERSIONS
    }

  def setUp(self):
    super().setUp()
    self.gnm_np = self.gnms[_MAINTAINED_MAJOR_GNM_VERSIONS[0]]
    self.image_size = (160, 120)

  def test_render_single_frame(self):
    renderer = gnm_pyrender.GNMMeshRenderer(
        gnm_np=self.gnm_np,
        triangles='~eye_exteriors',
    )
    output = renderer.render(image_size=self.image_size)
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
        self.gnm_np.template_vertex_positions,
        (num_frames, self.gnm_np.num_vertices, 3),
    )
    renderer = gnm_pyrender.GNMMeshRenderer(gnm_np=self.gnm_np)
    output = renderer.render(
        vertices=batched_vertices, image_size=self.image_size
    )
    self.assertEqual(output.image.shape, (num_frames, 120, 160, 3))
    self.assertEqual(output.alpha.shape, (num_frames, 120, 160, 1))
    renderer.close()

  def test_reuse_offscreen_renderer(self):
    renderer = gnm_pyrender.GNMMeshRenderer(gnm_np=self.gnm_np)
    renderer.render(image_size=self.image_size)
    first_renderer_instance = renderer._renderer
    self.assertIsNotNone(first_renderer_instance)

    # Rendering second frame with identical image size reuses the renderer.
    renderer.render(image_size=self.image_size)
    self.assertIs(renderer._renderer, first_renderer_instance)

    # Changing image size creates a new renderer.
    renderer.render(image_size=(80, 60))
    self.assertIsNot(renderer._renderer, first_renderer_instance)
    renderer.close()
    self.assertIsNone(renderer._renderer)

  def test_update_triangles(self):
    renderer = gnm_pyrender.GNMMeshRenderer(gnm_np=self.gnm_np)
    renderer.render(image_size=self.image_size)
    renderer.update_triangles('skin')
    self.assertEqual(renderer.triangles, 'skin')
    output = renderer.render(image_size=self.image_size)
    self.assertEqual(output.image.shape, (120, 160, 3))

    renderer.update_triangles(('~eye_exteriors', '-teeth'))
    self.assertEqual(renderer.triangles, ('~eye_exteriors', '-teeth'))
    output_seq = renderer.render(image_size=self.image_size)
    self.assertEqual(output_seq.image.shape, (120, 160, 3))
    renderer.close()

  def test_update_camera(self):
    renderer = gnm_pyrender.GNMMeshRenderer(gnm_np=self.gnm_np)
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
    renderer = gnm_pyrender.GNMMeshRenderer(gnm_np=self.gnm_np)
    renderer.update_vertices(vertices=self.gnm_np.template_vertex_positions)
    self.assertIsNotNone(renderer._cached_vertices)
    self.assertIsNotNone(renderer._cached_normals)
    renderer.close()

  def test_composite(self):
    renderer = gnm_pyrender.GNMMeshRenderer(gnm_np=self.gnm_np)
    output = renderer.render(image_size=self.image_size)
    composited_default = output.composite()
    self.assertEqual(composited_default.shape, (120, 160, 3))
    self.assertEqual(composited_default.dtype, np.uint8)

    background_image = np.full((120, 160, 3), 128, dtype=np.uint8)
    composited_custom = output.composite(
        background_image=background_image, alpha=0.8
    )
    self.assertEqual(composited_custom.shape, (120, 160, 3))
    self.assertEqual(composited_custom.dtype, np.uint8)
    renderer.close()


if __name__ == '__main__':
  absltest.main()
