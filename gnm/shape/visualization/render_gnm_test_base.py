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

"""Shared base test classes and helpers for GNM mesh rendering backends."""

from collections.abc import Callable
import functools

import tempfile
from typing import Any
from unittest import mock

from absl.testing import parameterized
import cv2
from etils import epath
from gnm.shape import gnm_numpy
from gnm.shape import gnm_test_utils
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import render_common
from gnm.shape.visualization import vertex_colors as vertex_colors_module
import mediapy as media
import numpy as np

_OUTPUTS_TMPDIR = tempfile.TemporaryDirectory()
_OUTPUTS_DIR = epath.Path(_OUTPUTS_TMPDIR.name)


@functools.cache
def load_head_gnm(version: str) -> gnm_numpy.GNM:
  """Loads the GNM head model of `version` once, shared by all tests.

  Tests must not modify the returned model.

  Args:
    version: The GNM major version, e.g. 'v3'.

  Returns:
    The GNM head model.
  """
  return gnm_test_utils.load_gnm(
      gnm_numpy.GNM, version, gnm_numpy.GNMVariant.HEAD
  )


def write_gif(
    outputs_dir: epath.Path | str,
    name: str,
    images: np.ndarray,
    fps: int = 10,
) -> None:
  """Writes an animated GIF to the undeclared outputs directory."""
  outputs_dir = epath.Path(outputs_dir)
  gif_path = outputs_dir / f'{name}.gif'
  media.write_video(gif_path, images, codec='gif', fps=fps)


def write_images(
    outputs_dir: epath.Path | str,
    name: str,
    images: np.ndarray,
) -> None:
  """Writes a row of images to the undeclared outputs directory."""
  outputs_dir = epath.Path(outputs_dir)
  png_path = outputs_dir / f'{name}.png'
  height, width = images.shape[-3:-1]
  reshaped = images.reshape(-1, height, width, 3)
  stack = np.hstack(list(reshaped))
  media.write_image(png_path, stack)


def get_random_parameters(
    batch_dims: tuple[int, ...],
    gnm_np: gnm_numpy.GNM,
) -> dict[str, np.ndarray]:
  """Returns random GNM parameters."""
  identity = np.random.uniform(
      -1.5, 1.5, size=batch_dims + (gnm_np.identity_dim,)
  )
  expression = np.random.uniform(
      -1.5, 1.5, size=batch_dims + (gnm_np.expression_dim,)
  )
  rotations = np.random.uniform(
      -0.2, 0.2, size=batch_dims + (gnm_np.num_joints, 3)
  )
  translation = np.random.uniform(-0.5, 0.5, size=batch_dims + (3,)) * 0.0
  return dict(
      identity=identity.astype(np.float32),
      expression=expression.astype(np.float32),
      rotations=rotations.astype(np.float32),
      translation=translation.astype(np.float32),
  )


def _dummy_render_fn(*args, **kwargs) -> render_common.FloatArray:
  """Dummy render function placeholder to satisfy static analysis."""
  del args, kwargs
  return np.empty((0,), dtype=np.float32)


class RenderGNMTestBase(parameterized.TestCase):
  """Base test class for functional tests of render_gnm backends."""

  render_fn: Callable[..., render_common.FloatArray] = staticmethod(
      _dummy_render_fn
  )
  test_all_versions: bool = False

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {
        version: load_head_gnm(version)
        for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
    }

  def setUp(self):
    super().setUp()
    if (
        self.render_fn is None
        or getattr(self.render_fn, '__func__', self.render_fn)
        is _dummy_render_fn
    ):
      raise ValueError('Subclasses must define `render_fn`.')
    np.random.seed(0)
    self.gnm_np = self.gnms[gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]]
    self.outputs_dir = _OUTPUTS_DIR / self.__class__.__name__
    self.outputs_dir.mkdir(parents=True, exist_ok=True)

    self.height, self.width = 320, 240
    self.image_size = (self.width, self.height)
    self.rendering_kwargs = {
        'image_size': self.image_size,
    }

  def _get_target_gnm_versions(self) -> list[tuple[str, gnm_numpy.GNM]]:
    """Returns the GNM versions to test."""
    if self.test_all_versions:
      return [
          (v, self.gnms[v]) for v in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
      ]
    return [(gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0], self.gnm_np)]

  def test_no_args(self):
    """Tests we can render without any parameters."""
    for version, gnm_np in self._get_target_gnm_versions():
      image = self.render_fn(gnm_np=gnm_np, image_size=self.image_size)

      with self.subTest(f'Renders shape ({version})'):
        self.assertEqual(image.shape, (self.height, self.width, 3))

      with self.subTest(f'Something has been rendered ({version})'):
        unique_colors = np.unique(image.reshape(-1, 3), axis=0)
        self.assertGreater(len(unique_colors), 1)

      write_images(self.outputs_dir, f'test_no_args_{version}', image)

  @parameterized.parameters((5,), (15,))
  def test_spin_period(self, spin_period: int):
    """Tests we can render spins of different length."""
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        world_to_camera = render_common.get_spin_world_to_camera(
            gnm_np=gnm_np,
            vertices=gnm_np.template_vertex_positions,
            spin_period=spin_period,
        )

        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            world_to_camera=world_to_camera,
        )
        self.assertLen(renders, spin_period)
        write_gif(
            self.outputs_dir,
            f'spin_period_{spin_period}_{version}',
            renders,
            fps=spin_period,
        )

  def test_spin_period_with_time_dimension(self):
    """Tests we can render spins with a time dimension."""
    num_frames = 5
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        parameters = get_random_parameters((num_frames,), gnm_np)
        vertices = gnm_np(**parameters)

        world_to_camera = render_common.get_spin_world_to_camera(
            gnm_np=gnm_np,
            vertices=vertices,
            has_time_dimension=True,
            spin_period=num_frames,
        )

        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            vertices=vertices,
            world_to_camera=world_to_camera,
        )
        self.assertLen(renders, num_frames)
        write_gif(
            self.outputs_dir,
            f'vary_vertices_{num_frames}_{version}',
            renders,
            fps=2,
        )

  def test_msaa(self):
    """Tests we can render with MSAA."""
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        default_render = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            multisample_antialiasing=1,
        )
        msaa_render = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            multisample_antialiasing=2,
        )
        images = np.hstack([default_render, msaa_render])
        write_images(self.outputs_dir, f'msaa_{version}', images)

  def test_multi_gnm_image(self):
    """Tests we can render multiple GNMs per frame."""
    width, height = 640, 480
    num_gnms = 3
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        parameters = get_random_parameters((num_gnms,), gnm_np)
        parameters['rotations'][0, :, :] = 0.0
        parameters['translation'] = np.zeros((num_gnms, 3), dtype=np.float32)
        parameters['translation'][1, 0] = -0.2
        parameters['translation'][2, 0] = 0.2

        vertices = gnm_np(**parameters)

        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=(width, height),
            vertices=vertices,
            multiple_gnms=True,
        )
        self.assertSequenceEqual(renders.shape, (height, width, 3))
        write_images(self.outputs_dir, f'multi_gnm_image_{version}', renders)

  def test_error_raised_if_multiple_gnms_and_no_batch_dims(self):
    """Tests error if rendering multiple GNMs without batch dimension."""
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        with self.assertRaisesRegex(
            ValueError, 'multiple_gnms=True, but vertices is only 2D'
        ):
          self.render_fn(
              gnm_np=gnm_np,
              image_size=self.image_size,
              multiple_gnms=True,
          )

  def test_background_color(self):
    """Tests we can render with a background color."""
    background_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            background_color=background_color,
        )
        write_images(self.outputs_dir, f'green_background_{version}', renders)

  def test_alpha(self):
    """Tests we can render with alpha."""
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        alpha_1 = self.render_fn(
            gnm_np=gnm_np, image_size=self.image_size, alpha=1.0
        )
        alpha_half = self.render_fn(
            gnm_np=gnm_np, image_size=self.image_size, alpha=0.5
        )
        images = np.hstack([alpha_1, alpha_half])
        write_images(self.outputs_dir, f'alpha_{version}', images)

  @parameterized.product(
      batch_dims=[(), (2,)],
      dtype=[np.float32, np.uint8],
  )
  def test_background_image(self, batch_dims: tuple[int, ...], dtype: Any):
    """Tests we can render with a background image."""
    np.random.seed(0)

    image_shape = (self.height, self.width)
    background_image = np.random.uniform(size=(*batch_dims, *image_shape, 3))

    multiplier = np.random.uniform(0.5, 1.5, size=batch_dims)
    background_image = background_image * multiplier[..., None, None, None]

    if dtype == np.uint8:
      background_image = (background_image * 255).astype(np.uint8)

    dtype_name = np.dtype(dtype).name
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        vertices = gnm_np.template_vertex_positions[None].astype(np.float32)
        vertices = np.tile(vertices, (*batch_dims, 1, 1))

        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            vertices=vertices,
            background_color=background_image,
        )

        write_images(
            self.outputs_dir,
            f'background_image_{batch_dims}_{dtype_name}_{version}',
            renders,
        )

  def test_no_shading(self):
    """Tests we can render without shading."""
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        render_with_shading = self.render_fn(
            gnm_np=gnm_np, image_size=self.image_size
        )
        render_without_shading = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            include_shading=False,
        )
        renders = np.hstack([render_with_shading, render_without_shading])
        write_images(self.outputs_dir, f'no_shading_{version}', renders)

  def test_batch_vertex_colors(self):
    """Tests we can render with per-sample vertex colors in a batch."""
    batch_size = 2
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        vertices = gnm_np.template_vertex_positions
        vertices = np.broadcast_to(vertices, (batch_size,) + vertices.shape)

        num_vertices = vertices.shape[1]

        batch_colors = np.zeros((batch_size, num_vertices, 3), dtype=np.float32)
        batch_colors[0, :, 0] = 1.0  # Red for first
        batch_colors[1, :, 2] = 1.0  # Blue for second

        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            vertices=vertices,
            vertex_colors=batch_colors,
        )
        self.assertEqual(
            renders.shape, (batch_size, self.height, self.width, 3)
        )
        red_0, _, blue_0 = np.mean(renders[0], axis=(0, 1))
        red_1, _, blue_1 = np.mean(renders[1], axis=(0, 1))
        self.assertGreater(red_0, blue_0, msg='Red > Blue for first image.')
        self.assertLess(red_1, blue_1, msg='Red < Blue for second image.')
        write_images(
            self.outputs_dir, f'batch_vertex_colors_{version}', renders
        )

  def test_batch_vertex_colors_with_multiple_gnms(self):
    """Tests we can render multiple GNMs per frame with different colors."""
    width, height = 640, 480
    num_gnms = 3
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        parameters = get_random_parameters((num_gnms,), gnm_np)

        parameters['rotations'][0, :, :] = 0.0
        parameters['translation'] = np.zeros((num_gnms, 3), dtype=np.float32)
        parameters['translation'][1, 0] = -0.2
        parameters['translation'][2, 0] = 0.2

        vertices = gnm_np(**parameters)
        vertex_colors = np.zeros_like(vertices)
        colors = [
            vertex_colors_module.ORANGE,
            vertex_colors_module.GREEN,
            vertex_colors_module.CYAN,
        ]
        for i in range(num_gnms):
          vertex_colors[i] = vertex_colors_module.get_vertex_colors(
              color=colors[i], gnm_np=gnm_np
          )

        renders = self.render_fn(
            gnm_np=gnm_np,
            vertices=vertices,
            vertex_colors=vertex_colors,
            multiple_gnms=True,
            image_size=(width, height),
        )
        write_images(
            self.outputs_dir,
            f'batch_vertex_colors_multi_gnm_{version}',
            renders,
        )

  def test_no_texture(self):
    """Tests we can render without texture."""
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        render_with_texture = self.render_fn(
            gnm_np=gnm_np, image_size=self.image_size
        )
        render_without_texture = self.render_fn(
            gnm_np=gnm_np, image_size=self.image_size, texture=None
        )

        self.assertFalse(
            np.array_equal(render_with_texture, render_without_texture)
        )

        renders = np.hstack([render_with_texture, render_without_texture])
        write_images(self.outputs_dir, f'no_texture_{version}', renders)

  def test_custom_texture(self):
    """Tests we can render with a custom per-sample texture."""
    custom_texture = np.zeros((self.height, self.width, 3), dtype=np.float32)
    custom_texture[:, : self.width // 2, 0] = 1.0
    custom_texture[:, self.width // 2 :, 1] = 1.0

    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            texture=custom_texture,
        )
        write_images(self.outputs_dir, f'custom_texture_{version}', renders)

  def test_per_part_texture(self):
    """Tests we can render with a per-part texture."""
    green = np.zeros((64, 64, 3), dtype=np.float32)
    green[..., 1] = 1.0
    red = np.zeros((64, 64, 3), dtype=np.float32)
    red[..., 0] = 1.0

    texture = {'skin': red, 'left_eye': green, 'right_eye': green}

    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        vertices = gnm_np.template_vertex_positions

        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            texture=texture,
            include_shading=False,
            vertex_colors=np.ones_like(vertices),
            background_color=0.0,
        )

        eye_indices = gnm_np.vertex_group_indices('eye_interiors')
        eye_image_points = render_common.project_points_for_gnm(
            points_world=vertices[eye_indices],
            vertices=vertices,
            gnm_np=gnm_np,
            image_size=self.image_size,
        )

        # Compute eye bounding box in image space.
        xmin, ymin = np.min(eye_image_points, axis=0).astype(int)
        xmax, ymax = np.max(eye_image_points, axis=0).astype(int)

        eye_mask = np.zeros(renders.shape[:2], dtype=bool)
        eye_mask[ymin:ymax, xmin:xmax] = True

        self.assertEqual(np.max(renders[eye_mask][..., 1]), 1.0)
        self.assertEqual(np.min(renders[~eye_mask][..., 1]), 0.0)

        cv2.rectangle(renders, (xmin, ymin), (xmax, ymax), (1.0, 1.0, 1.0), 1)
        write_images(self.outputs_dir, f'per_part_texture_{version}', renders)

  def test_incorrect_texture_part_name_raises_error(self):
    """Tests error if texture part name is not a GNM part name."""
    texture = {
        'wrong_part': np.zeros((self.height, self.width, 3), dtype=np.float32)
    }
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        with self.assertRaisesRegex(
            ValueError, r"Texture keys \{'wrong_part'\} are not GNM part names"
        ):
          self.render_fn(
              gnm_np=gnm_np,
              image_size=self.image_size,
              texture=texture,
          )

  def test_rgba_vertex_colors(self):
    """Tests rendering with 4-channel vertex colors and custom background."""
    for version, gnm_np in self._get_target_gnm_versions():
      with self.subTest(version=version):
        vertices = gnm_np.template_vertex_positions
        rgba_colors = np.ones((vertices.shape[0], 4), dtype=np.float32)
        rgba_colors[:, :3] = [1.0, 0.0, 0.0]
        rgba_colors[:, 3] = 0.5
        renders = self.render_fn(
            gnm_np=gnm_np,
            image_size=self.image_size,
            vertices=vertices,
            vertex_colors=rgba_colors,
            include_shading=False,
            background_color=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(renders.shape, (self.height, self.width, 3))


class RenderGNMBatchTestBase(parameterized.TestCase):
  """Base test class for batching tests of render_gnm backends."""

  BATCH_DIMS = [(5,), (5, 10), (3, 6, 2)]
  render_fn: Callable[..., render_common.FloatArray] = staticmethod(
      _dummy_render_fn
  )
  mock_target: str = ''

  def setUp(self):
    super().setUp()
    if (
        self.render_fn is None
        or getattr(self.render_fn, '__func__', self.render_fn)
        is _dummy_render_fn
        or not self.mock_target
    ):
      raise ValueError('Subclasses must define `render_fn` and `mock_target`.')

    self.outputs_dir = _OUTPUTS_DIR / self.__class__.__name__
    self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def mock_render(vertices, image_size, **kwargs):
      del kwargs
      batch_shape = vertices.shape[:-3]
      return np.zeros(
          (*batch_shape, image_size[1], image_size[0], 3), dtype=np.float32
      )

    self.mock_render = self.enter_context(
        mock.patch(
            self.mock_target,
            autospec=True,
        )
    )
    self.mock_render.side_effect = mock_render

    self.gnm_np = load_head_gnm(gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0])
    self.image_size = (240, 320)
    self.image_dims = (self.image_size[1], self.image_size[0], 3)
    self.vertices_shape = self.gnm_np.template_vertex_positions.shape

  @parameterized.product(
      batch_dims=[(5,), (5, 10), (3, 6, 2)],
      multiple_gnms=[False, True],
  )
  def test_batch_vertices(
      self, batch_dims: tuple[int, ...], multiple_gnms: bool
  ):
    """Tests batching of vertices."""
    gnm_dim = (1,) if multiple_gnms else ()
    vertices = np.zeros(
        (*batch_dims, *gnm_dim, *self.vertices_shape), dtype=np.float32
    )
    renders = self.render_fn(
        gnm_np=self.gnm_np,
        image_size=self.image_size,
        vertices=vertices,
        multiple_gnms=multiple_gnms,
    )
    self.assertSequenceEqual(renders.shape, (*batch_dims, *self.image_dims))

  @parameterized.product(
      batch_dims=[(5,), (5, 10), (3, 6, 2)],
      multiple_gnms=[False, True],
  )
  def test_batch_vertex_colors(
      self, batch_dims: tuple[int, ...], multiple_gnms: bool
  ):
    """Tests batching of vertex colors."""
    gnm_dim = (1,) if multiple_gnms else ()
    vertices = np.zeros((*gnm_dim, *self.vertices_shape), dtype=np.float32)
    vertex_colors = np.zeros(
        (*batch_dims, *gnm_dim, *self.vertices_shape), dtype=np.float32
    )
    renders = self.render_fn(
        gnm_np=self.gnm_np,
        image_size=self.image_size,
        vertices=vertices,
        vertex_colors=vertex_colors,
        multiple_gnms=multiple_gnms,
    )
    self.assertSequenceEqual(renders.shape, (*batch_dims, *self.image_dims))

  @parameterized.named_parameters(
      ('batch_size_5', (5,)),
      ('batch_size_5_10', (5, 10)),
      ('batch_size_3_6_2', (3, 6, 2)),
  )
  def test_batch_world_to_camera(self, batch_dims: tuple[int, ...]):
    """Tests batching of world_to_camera."""
    world_to_camera = np.zeros((*batch_dims, 4, 4), dtype=np.float32)
    renders = self.render_fn(
        gnm_np=self.gnm_np,
        image_size=self.image_size,
        world_to_camera=world_to_camera,
    )
    self.assertSequenceEqual(renders.shape, (*batch_dims, *self.image_dims))

  @parameterized.named_parameters(
      ('batch_size_5', (5,)),
      ('batch_size_5_10', (5, 10)),
      ('batch_size_3_6_2', (3, 6, 2)),
  )
  def test_batch_camera_to_image(self, batch_dims: tuple[int, ...]):
    """Tests batching of camera_to_image."""
    camera_to_image = np.zeros((*batch_dims, 4, 4), dtype=np.float32)
    renders = self.render_fn(
        gnm_np=self.gnm_np,
        image_size=self.image_size,
        camera_to_image=camera_to_image,
    )
    self.assertSequenceEqual(renders.shape, (*batch_dims, *self.image_dims))

  @parameterized.named_parameters(
      ('batch_size_5', (5,)),
      ('batch_size_5_10', (5, 10)),
      ('batch_size_3_6_2', (3, 6, 2)),
  )
  def test_batch_background_image(self, batch_dims: tuple[int, ...]):
    """Tests batching of background_image."""
    background_image = np.zeros(
        (*batch_dims, *self.image_dims), dtype=np.float32
    )
    renders = self.render_fn(
        gnm_np=self.gnm_np,
        image_size=self.image_size,
        background_color=background_image,
    )
    self.assertSequenceEqual(renders.shape, (*batch_dims, *self.image_dims))

  @parameterized.named_parameters(
      ('batch_size_5', (5,)),
      ('batch_size_5_10', (5, 10)),
      ('batch_size_3_6_2', (3, 6, 2)),
  )
  def test_batch_texture(self, batch_dims: tuple[int, ...]):
    """Tests batching of texture."""
    texture = np.zeros((*batch_dims, 64, 64, 3), dtype=np.float32)
    renders = self.render_fn(
        gnm_np=self.gnm_np,
        image_size=self.image_size,
        texture=texture,
    )
    self.assertSequenceEqual(renders.shape, (*batch_dims, *self.image_dims))

  def test_error_on_batch_mismatch(self):
    """Tests that an error is raised if batch dimensions are incompatible."""
    vertices = np.zeros((5, 10, *self.vertices_shape), dtype=np.float32)
    world_to_camera = np.zeros((6, 4, 4), dtype=np.float32)
    with self.assertRaisesRegex(ValueError, 'Batch dimensions incompatible'):
      self.render_fn(
          gnm_np=self.gnm_np,
          image_size=self.image_size,
          vertices=vertices,
          world_to_camera=world_to_camera,
      )
