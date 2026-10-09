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

"""Tests for rendering GNM meshes using the PyRender backend."""

# pylint: disable=protected-access

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from gnm.shape import gnm_numpy
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import render_common
from gnm.shape.visualization import render_gnm
from gnm.shape.visualization import render_gnm_test_base
import numpy as np

_OUTPUTS_DIR = render_gnm_test_base._OUTPUTS_DIR
_write_images = render_gnm_test_base.write_images
_write_gif = render_gnm_test_base.write_gif


class RenderGNMTest(render_gnm_test_base.RenderGNMTestBase):
  """Tests for rendering GNM meshes using the PyRender backend."""

  render_fn = staticmethod(render_gnm.render_gnm)
  test_all_versions = True


class RenderGNMBatchTest(render_gnm_test_base.RenderGNMBatchTestBase):
  """Tests batching of arguments to render_gnm."""

  render_fn = staticmethod(render_gnm.render_gnm)
  mock_target = 'gnm.shape.visualization.gnm_pyrender.render'


class TestProjectPointsForGNM(parameterized.TestCase):
  """Tests projection of points for GNM."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {
        version: render_gnm_test_base.load_head_gnm(version)
        for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
    }

  def setUp(self):
    super().setUp()
    np.random.seed(0)

    self.outputs_dir = _OUTPUTS_DIR / self.__class__.__name__
    self.outputs_dir.mkdir(parents=True, exist_ok=True)

    self.height, self.width = 320, 240
    image_size = (self.width, self.height)

    # Store all rendering keyword arguments in a single dictionary.
    self.rendering_kwargs = {
        'image_size': image_size,
        'background_color': 0.0,  # So we can easily calculate mask.
    }

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_project_points_default_render(self, version):
    """Tests project_points_for_gnm with default render."""
    gnm_np = self.gnms[version]

    # Default render.
    image = render_gnm.render_gnm(gnm_np, **self.rendering_kwargs)

    # Project face joints under the same camera setup.
    joints_image = render_common.project_points_for_gnm(
        gnm_np=gnm_np,
        points_world=gnm_np.template_joint_positions,
        **self.rendering_kwargs,
    )

    # Project a point above the face under the same camera setup.
    points_world = np.array(
        [[0, gnm_np.template_vertex_positions[:, 1].max() + 0.05, 0]]
    )
    external_point_image = render_common.project_points_for_gnm(
        gnm_np=gnm_np,
        points_world=points_world,
        **self.rendering_kwargs,
    )

    mask = (image > 0).any(axis=-1)

    with self.subTest('Face joints in mask'):
      x, y = joints_image.T.astype(np.int32)
      self.assertTrue(mask[y, x].all())

    with self.subTest('Point above face not in mask'):
      x, y = external_point_image.T.astype(np.int32)
      self.assertFalse(mask[y, x].all())

    # Draw points on the image and save.
    image = (image * 255).astype(np.uint8)
    for x, y in joints_image:
      cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1, cv2.LINE_AA)
    for x, y in external_point_image:
      cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1, cv2.LINE_AA)

    _write_images(self.outputs_dir, 'project_points', image[None, :])

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_project_points_spin(self, version):
    """Tests projection of points for GNM in a spin."""
    gnm_np = self.gnms[version]
    spin_period = 30
    world_to_camera = render_common.get_spin_world_to_camera(
        gnm_np=gnm_np,
        vertices=gnm_np.template_vertex_positions,
        spin_period=spin_period,
    )

    image = render_gnm.render_gnm(
        gnm_np, world_to_camera=world_to_camera, **self.rendering_kwargs
    )

    # Project face joints under the same camera setup.
    joints_image = render_common.project_points_for_gnm(
        gnm_np=gnm_np,
        points_world=gnm_np.template_joint_positions,
        world_to_camera=world_to_camera,
        **self.rendering_kwargs,
    )

    with self.subTest('Joints shape is correct.'):
      self.assertEqual(joints_image.shape, (spin_period, gnm_np.num_joints, 2))

    with self.subTest('All joints inside mask.'):
      # image is shape (spin_period, height, width, 3)
      # joints is shape (spin_period, num_joints, 2)
      mask = (image > 0).any(axis=-1)

      with self.subTest('Face joints in mask'):
        x = joints_image[..., 0].astype(np.int32)
        y = joints_image[..., 1].astype(np.int32)
        for i in range(spin_period):
          is_in_mask = mask[i, y[i], x[i]].all()
          self.assertTrue(is_in_mask)

    # Draw points on the image and save.
    image = (image * 255).astype(np.uint8)
    for i in range(spin_period):
      for x, y in joints_image[i]:
        cv2.circle(image[i], (int(x), int(y)), 5, (0, 255, 0), -1, cv2.LINE_AA)

    _write_gif(
        self.outputs_dir, f'project_points_spin_{version}', image, fps=30
    )


if __name__ == '__main__':
  absltest.main()
