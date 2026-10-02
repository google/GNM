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

"""Tests parity between render_gnm and render_gnm_mitsuba."""

import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import render_gnm
from gnm.shape.visualization import render_gnm_mitsuba
from gnm.shape.visualization import render_gnm_test_base
import numpy as np

_OUTPUTS_TMPDIR = tempfile.TemporaryDirectory()
_OUTPUTS_DIR = epath.Path(_OUTPUTS_TMPDIR.name)


class RenderGNMParityTest(parameterized.TestCase):
  """Tests parity between render_gnm and render_gnm_mitsuba."""

  def setUp(self):
    super().setUp()
    self.gnm_np = render_gnm_test_base.load_head_gnm(
        gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]
    )
    self.outputs_dir = _OUTPUTS_DIR / self.__class__.__name__
    self.outputs_dir.mkdir(parents=True, exist_ok=True)
    self.vertices = self.gnm_np.template_vertex_positions
    self.image_size = (240, 320)

  def test_parity_flat_shading(self):
    """Tests rendering parity in flat shading mode."""
    kwargs = {
        'gnm_np': self.gnm_np,
        'vertices': self.vertices,
        'include_shading': False,
        'texture': None,
        'image_size': self.image_size,
    }
    image_pyrender = render_gnm.render_gnm(**kwargs)
    image_mitsuba = render_gnm_mitsuba.render_gnm(**kwargs)
    images = np.hstack([image_pyrender, image_mitsuba])
    render_gnm_test_base.write_images(
        self.outputs_dir, 'parity_flat_shading', images
    )

    diff = np.linalg.norm(image_pyrender - image_mitsuba, axis=-1)
    # Background and unshaded geometry should be very close.
    with self.subTest('Average pixels are close.'):
      self.assertLess(np.mean(diff), 0.01)
    with self.subTest('Few pixels are very different.'):
      self.assertGreater(np.mean(diff < 0.05), 0.99)

  def test_parity_vertex_colors(self):
    """Tests rendering parity with custom vertex colors."""
    colors = np.zeros_like(self.vertices)
    colors[:, 0] = 0.8
    kwargs = {
        'gnm_np': self.gnm_np,
        'vertices': self.vertices,
        'vertex_colors': colors,
        'include_shading': False,
        'texture': None,
        'image_size': self.image_size,
    }
    image_pyrender = render_gnm.render_gnm(**kwargs)
    image_mitsuba = render_gnm_mitsuba.render_gnm(**kwargs)
    diff = np.linalg.norm(image_pyrender - image_mitsuba, axis=-1)
    with self.subTest('Average pixels are close.'):
      self.assertLess(np.mean(diff), 0.01)
    with self.subTest('Few pixels are very different.'):
      self.assertGreater(np.mean(diff < 0.05), 0.99)


if __name__ == '__main__':
  absltest.main()
