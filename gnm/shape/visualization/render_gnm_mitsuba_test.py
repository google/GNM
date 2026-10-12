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

"""Tests for rendering GNM meshes using the Mitsuba backend."""

from absl.testing import absltest
from gnm.shape.visualization import render_gnm_mitsuba
from gnm.shape.visualization import render_gnm_test_base


class RenderGNMMitsubaTest(render_gnm_test_base.RenderGNMTestBase):
  """Tests for rendering GNM meshes using the Mitsuba backend."""

  render_fn = staticmethod(render_gnm_mitsuba.render_gnm)
  test_all_versions = False


class RenderGNMBatchTest(render_gnm_test_base.RenderGNMBatchTestBase):
  """Tests batching of arguments to render_gnm_mitsuba."""

  render_fn = staticmethod(render_gnm_mitsuba.render_gnm)
  mock_target = 'gnm.shape.visualization.gnm_mitsuba.render'


class RenderGNMMitsubaRendererTest(render_gnm_test_base.RenderGNMTestBase):
  """Tests for stateful GNMMeshRenderer exposed via render_gnm_mitsuba."""

  render_fn = staticmethod(render_gnm_mitsuba.render_gnm)
  test_all_versions = False

  def test_renderer_class_exposed_and_renders(self):
    """Tests that GNMMeshRenderer and MeshRenderOutput render template mesh."""
    self.assertIs(
        render_gnm_mitsuba.GNMMeshRenderer,
        render_gnm_mitsuba.gnm_mitsuba.GNMMeshRenderer,
    )
    renderer = render_gnm_mitsuba.GNMMeshRenderer(gnm_np=self.gnm_np, spp=1)
    render_output = renderer.render(image_size=(32, 32))
    self.assertIsInstance(render_output, render_gnm_mitsuba.MeshRenderOutput)
    self.assertEqual(render_output.image.shape, (32, 32, 3))
    self.assertEqual(render_output.alpha.shape, (32, 32, 1))
    renderer.close()


if __name__ == '__main__':
  absltest.main()
