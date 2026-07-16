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

"""Tests for render_gnm."""

# pylint: disable=protected-access

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from etils import epath
from gnm.shape import gnm_numpy
from gnm.shape.data.versions import gnm_catalog
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import render_gnm
from gnm.shape.visualization import vertex_colors as vertex_colors_module
import mediapy as media
import numpy as np

_OUTPUTS_DIR = epath.Path(os.environ['TEST_UNDECLARED_OUTPUTS_DIR'])

_TupleOfInts = tuple[int, ...]


def _write_gif(
    outputs_dir: epath.Path | str,
    name: str,
    images: np.ndarray,
    fps: int = 10,
) -> None:
  """Writes an animated GIF to the undeclared outputs directory."""
  outputs_dir = epath.Path(outputs_dir)
  gif_path = outputs_dir / f'{name}.gif'
  media.write_video(gif_path, images, codec='gif', fps=fps)


def _write_images(
    outputs_dir: epath.Path | str,
    name: str,
    images: np.ndarray,
) -> None:
  """Writes a row of images to the undeclared outputs directory."""
  png_path = (
      outputs_dir / f'{name}.png'  # pyrefly: ignore[unsupported-operation]
  )
  height, width = images.shape[-3:-1]
  reshaped = images.reshape(-1, height, width, 3)
  stack = np.hstack(reshaped)  # pyrefly: ignore[no-matching-overload]
  media.write_image(png_path, stack)


def _get_random_parameters(
    batch_dims: _TupleOfInts,
    gnm_np: gnm_numpy.GNM,
) -> dict[str, np.ndarray]:
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


class RenderGNMTest(parameterized.TestCase):

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {}
    for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

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
    }

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_no_args(self, version):
    """Tests we can render without any parameters."""
    gnm_np = self.gnms[version]
    image = render_gnm.render_gnm(gnm_np, **self.rendering_kwargs)

    with self.subTest('Something has been rendered.'):
      unique_colors = np.unique(image.reshape(-1, 3), axis=0)
      self.assertGreater(len(unique_colors), 1)

    _write_images(
        self.outputs_dir,
        f'test_no_args_{version}',
        image,
    )

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      spin_period=[5, 30],
  )
  def test_spin_period(self, version, spin_period: int):
    """Tests we can render spins of different length."""
    gnm_np = self.gnms[version]

    world_to_camera = render_gnm.get_spin_world_to_camera(
        gnm_np=gnm_np,
        vertices=gnm_np.template_vertex_positions,
        spin_period=spin_period,
    )

    cam_dict = {'world_to_camera': world_to_camera}
    kwargs = (
        self.rendering_kwargs | cam_dict  # pyrefly: ignore[bad-argument-type]
    )
    renders = render_gnm.render_gnm(gnm_np, **kwargs)
    self.assertLen(renders, spin_period)
    _write_gif(
        self.outputs_dir,
        f'spin_period_{spin_period}_{version}',
        renders,
        fps=spin_period,
    )

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_spin_period_with_time_dimension(self, version):
    gnm_np = self.gnms[version]

    num_frames = 10
    parameters = _get_random_parameters((num_frames,), gnm_np)
    vertices = gnm_np(**parameters)

    world_to_camera = render_gnm.get_spin_world_to_camera(
        gnm_np=gnm_np,
        vertices=vertices,
        has_time_dimension=True,
        spin_period=num_frames,
    )

    renders = render_gnm.render_gnm(
        gnm_np,
        vertices=vertices,
        world_to_camera=world_to_camera,
        **self.rendering_kwargs,
    )
    self.assertLen(renders, num_frames)
    _write_gif(
        self.outputs_dir,
        f'vary_vertices_{num_frames}_{version}',
        renders,
        fps=2,
    )

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_msaa(self, version):
    """Tests we can render with MSAA."""
    gnm_np = self.gnms[version]
    default_render = render_gnm.render_gnm(
        gnm_np, multisample_antialiasing=1, **self.rendering_kwargs
    )
    msaa_render = render_gnm.render_gnm(
        gnm_np, multisample_antialiasing=4, **self.rendering_kwargs
    )
    images = np.hstack([default_render, msaa_render])
    _write_images(self.outputs_dir, f'msaa_{version}', images)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_multi_gnm_image(self, version):
    """Tests we can render multiple GNMs per frame."""
    gnm_np = self.gnms[version]
    width, height = 640, 480

    rendering_kwargs = self.rendering_kwargs.copy()
    rendering_kwargs['image_size'] = (width, height)

    num_gnms = 3
    parameters = _get_random_parameters((num_gnms,), gnm_np)

    # Camera parameters decided by the first GNM, so:
    # 1) Zero out their rotations for nice camera placement.
    parameters['rotations'][0, :, :] = 0.0

    # 2) Place the other two GNMs to the left and right of the first.
    parameters['translation'] = np.zeros((num_gnms, 3), dtype=np.float32)
    parameters['translation'][1, 0] = -0.2
    parameters['translation'][2, 0] = 0.2

    vertices = gnm_np(**parameters)

    renders = render_gnm.render_gnm(
        gnm_np,
        vertices=vertices,
        multiple_gnms=True,
        **rendering_kwargs,  # pyrefly: ignore[bad-argument-type]
    )

    # Check we've rendered a single image with multiple GNMs.
    with self.subTest('Renders shape'):
      self.assertSequenceEqual(renders.shape, (height, width, 3))

    _write_images(self.outputs_dir, 'multi_gnm_image', renders)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_error_raised_if_multiple_gnms_and_no_batch_dims(self, version):
    """Tests error if rendering multiple GNMs without batch dimension."""
    gnm_np = self.gnms[version]
    with self.assertRaisesRegex(
        ValueError, 'multiple_gnms=True, but vertices is only 2D'
    ):
      render_gnm.render_gnm(gnm_np, multiple_gnms=True, **self.rendering_kwargs)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_background_color(self, version):
    """Tests we can render with a background color."""
    gnm_np = self.gnms[version]
    background_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    renders = render_gnm.render_gnm(
        gnm_np,
        background_color=background_color,
        **self.rendering_kwargs,
    )
    _write_images(self.outputs_dir, f'green_background_{version}', renders)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_alpha(self, version):
    """Tests we can render with alpha."""
    gnm_np = self.gnms[version]
    alpha_1 = render_gnm.render_gnm(gnm_np, alpha=1.0, **self.rendering_kwargs)
    alpha_half = render_gnm.render_gnm(
        gnm_np, alpha=0.5, **self.rendering_kwargs
    )
    images = np.hstack([alpha_1, alpha_half])
    _write_images(self.outputs_dir, f'alpha_{version}', images)

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_dims=[(), (5,)],
      dtype=[np.float32, np.uint8],
  )
  def test_background_image(
      self, version, batch_dims: tuple[int, ...], dtype: np.dtype
  ):
    """Tests we can render with a background image."""
    gnm_np = self.gnms[version]
    np.random.seed(0)

    image_shape = (self.height, self.width)
    background_image = np.random.uniform(size=(*batch_dims, *image_shape, 3))

    # Apply a random transformation to each background image to distinguish.
    multiplier = np.random.uniform(0.5, 1.5, size=batch_dims)
    background_image = background_image * multiplier[..., None, None, None]

    if dtype == np.uint8:
      background_image = (background_image * 255).astype(np.uint8)

    vertices = gnm_np.template_vertex_positions[None].astype(np.float32)
    vertices = np.tile(vertices, (*batch_dims, 1, 1))

    renders = render_gnm.render_gnm(
        gnm_np,
        vertices=vertices,
        background_color=background_image,
        **self.rendering_kwargs,
    )

    dtype_name = np.dtype(dtype).name
    _write_images(
        self.outputs_dir, f'background_image_{batch_dims}_{dtype_name}', renders
    )

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_no_shading(self, version):
    """Tests we can render without shading."""
    gnm_np = self.gnms[version]
    render_with_shading = render_gnm.render_gnm(gnm_np, **self.rendering_kwargs)
    render_without_shading = render_gnm.render_gnm(
        gnm_np, include_shading=False, **self.rendering_kwargs
    )
    renders = np.hstack([render_with_shading, render_without_shading])
    _write_images(self.outputs_dir, f'no_shading_{version}', renders)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_batch_vertex_colors(self, version):
    """Tests we can render with per-sample vertex colors in a batch."""
    gnm_np = self.gnms[version]
    batch_size = 2
    vertices = gnm_np.template_vertex_positions
    vertices = np.broadcast_to(vertices, (batch_size,) + vertices.shape)

    num_vertices = vertices.shape[1]

    batch_colors = np.zeros((batch_size, num_vertices, 3), dtype=np.float32)
    batch_colors[0, :, 0] = 1.0  # Red for the first batch item.
    batch_colors[1, :, 2] = 1.0  # Blue for the second batch item.

    renders = render_gnm.render_gnm(
        gnm_np,
        vertices=vertices,
        vertex_colors=batch_colors,
        **self.rendering_kwargs,
    )

    self.assertLen(renders, batch_size)

    # Get mean RGB per render.
    red_0, _, blue_0 = np.mean(renders[0], axis=(0, 1))
    red_1, _, blue_1 = np.mean(renders[1], axis=(0, 1))

    self.assertGreater(red_0, blue_0, msg='Red > Blue for first image.')
    self.assertLess(red_1, blue_1, msg='Red < Blue for second image.')

    _write_images(self.outputs_dir, 'batch_vertex_colors', renders)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_batch_vertex_colors_with_multiple_gnms(self, version):
    """Tests we can render multiple GNMs per frame with different colors."""
    gnm_np = self.gnms[version]

    width, height = 640, 480
    rendering_kwargs = self.rendering_kwargs.copy()
    rendering_kwargs['image_size'] = (width, height)

    num_gnms = 3
    parameters = _get_random_parameters((num_gnms,), gnm_np)

    # Camera parameters decided by the first GNM, so:
    # 1) Zero out their rotations for nice camera placement.
    parameters['rotations'][0, :, :] = 0.0

    # 2) Place the other two GNMs to the left and right of the first.
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

    renders = render_gnm.render_gnm(
        gnm_np,
        vertices=vertices,
        vertex_colors=vertex_colors,
        multiple_gnms=True,
        **rendering_kwargs,  # pyrefly: ignore[bad-argument-type]
    )

    _write_images(
        self.outputs_dir, f'batch_vertex_colors_multi_gnm_{version}', renders
    )

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_no_texture(self, version):
    """Tests we can render without texture."""
    gnm_np = self.gnms[version]
    render_with_texture = render_gnm.render_gnm(gnm_np, **self.rendering_kwargs)
    render_without_texture = render_gnm.render_gnm(
        gnm_np, texture=None, **self.rendering_kwargs
    )

    with self.subTest('Renders are different.'):
      self.assertFalse(
          np.array_equal(render_with_texture, render_without_texture)
      )

    renders = np.hstack([render_with_texture, render_without_texture])
    _write_images(self.outputs_dir, f'no_texture_{version}', renders)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_custom_texture(self, version):
    """Tests we can render with a custom per-sample texture."""
    gnm_np = self.gnms[version]
    custom_texture = np.zeros((self.height, self.width, 3), dtype=np.float32)
    custom_texture[:, : self.width // 2, 0] = 1.0
    custom_texture[:, self.width // 2 :, 1] = 1.0

    renders = render_gnm.render_gnm(
        gnm_np, texture=custom_texture, **self.rendering_kwargs
    )
    _write_images(self.outputs_dir, f'custom_texture_{version}', renders)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_per_part_texture(self, version):
    """Tests we can render with a per-part texture."""
    gnm_np = self.gnms[version]
    green = np.zeros((64, 64, 3), dtype=np.float32)
    green[..., 1] = 1.0
    red = np.zeros((64, 64, 3), dtype=np.float32)
    red[..., 0] = 1.0

    texture = {'skin': red, 'left_eye': green, 'right_eye': green}

    vertices = gnm_np.template_vertex_positions

    renders = render_gnm.render_gnm(
        gnm_np,
        texture=texture,
        include_shading=False,
        vertex_colors=np.ones_like(vertices),
        background_color=0.0,
        **self.rendering_kwargs,
    )

    eye_indices = gnm_np.vertex_group_indices('eye_interiors')
    proj_fn = (
        render_gnm.project_points_for_gnm  # pytype: disable=module-attr
    )
    eye_image_points = proj_fn(
        points_world=vertices[eye_indices],
        vertices=vertices,
        gnm_np=gnm_np,
        **self.rendering_kwargs,
    )

    # Compute eye bounding box in image space.
    xmin, ymin = np.min(eye_image_points, axis=0).astype(int)
    xmax, ymax = np.max(eye_image_points, axis=0).astype(int)

    eye_mask = np.zeros(renders.shape[:2], dtype=bool)
    eye_mask[ymin:ymax, xmin:xmax] = True

    with self.subTest('Green inside the eye bounding box.'):
      self.assertEqual(np.max(renders[eye_mask][..., 1]), 1.0)

    with self.subTest('No green outside the eye bounding box.'):
      self.assertEqual(np.min(renders[~eye_mask][..., 1]), 0.0)

    # Draw bounding box for visualization.
    cv2.rectangle(renders, (xmin, ymin), (xmax, ymax), (1.0, 1.0, 1.0), 1)
    _write_images(self.outputs_dir, f'per_part_texture_{version}', renders)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_incorrect_texture_part_name_raises_error(self, version):
    """Tests error if texture part name is not a GNM part name."""
    gnm_np = self.gnms[version]
    texture = {
        'wrong_part': np.zeros((self.height, self.width, 3), dtype=np.float32)
    }
    with self.assertRaisesRegex(
        ValueError, r"Texture keys \{'wrong_part'\} are not GNM part names"
    ):
      render_gnm.render_gnm(gnm_np, texture=texture, **self.rendering_kwargs)


class RenderGNMBatchTest(parameterized.TestCase):
  """Tests batching of arguments to render_gnm."""

  BATCH_DIMS = [(5,), (5, 10), (3, 6, 2)]

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {}
    for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

  def setUp(self):
    super().setUp()
    self.outputs_dir = _OUTPUTS_DIR / self.__class__.__name__
    self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # For this test, Pyrender just returns a black image of the correct size.
    def mock_render(vertices, image_size, **kwargs):
      del kwargs
      batch_shape = vertices.shape[:-3]
      return np.zeros((*batch_shape, image_size[1], image_size[0], 3))

    self.mock_render = self.enter_context(
        mock.patch(
            'gnm.shape.visualization.gnm_pyrender.render',
            autospec=True,
        )
    )
    self.mock_render.side_effect = mock_render

    self.image_size = (240, 320)
    self.image_dims = (self.image_size[1], self.image_size[0], 3)
    self.rendering_kwargs = {
        'image_size': self.image_size,
    }

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_dims=list(BATCH_DIMS),
      multiple_gnms=[False, True],
  )
  def test_batch_vertices(
      self, version, batch_dims: _TupleOfInts, multiple_gnms: bool
  ):
    """Tests we can render with a batch of vertices."""
    gnm_np = self.gnms[version]
    gnm_dim = (1,) if multiple_gnms else ()
    vertices_shape = gnm_np.template_vertex_positions.shape
    vertices = np.zeros(
        (*batch_dims, *gnm_dim, *vertices_shape),
        dtype=np.float32,
    )
    image = render_gnm.render_gnm(
        gnm_np,
        vertices=vertices,
        multiple_gnms=multiple_gnms,
        **self.rendering_kwargs,
    )
    self.assertEqual(image.shape, (*batch_dims, *self.image_dims))

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_dims=list(BATCH_DIMS),
      multiple_gnms=[False, True],
  )
  def test_batch_vertex_colors(
      self, version, batch_dims: _TupleOfInts, multiple_gnms: bool
  ):
    """Tests we can render with a batch of vertex colors."""
    gnm_np = self.gnms[version]
    gnm_dim = (1,) if multiple_gnms else ()
    vertices_shape = gnm_np.template_vertex_positions.shape
    vertex_colors = np.zeros(
        (*batch_dims, *gnm_dim, *vertices_shape), dtype=np.float32
    )
    vertices = np.zeros((*gnm_dim, *vertices_shape), dtype=np.float32)
    image = render_gnm.render_gnm(
        gnm_np,
        vertices=vertices,
        vertex_colors=vertex_colors,
        multiple_gnms=multiple_gnms,
        **self.rendering_kwargs,
    )
    self.assertEqual(image.shape, (*batch_dims, *self.image_dims))

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_dims=list(BATCH_DIMS),
  )
  def test_batch_world_to_camera(self, version, batch_dims: _TupleOfInts):
    """Tests we can render with a batch of world-to-camera matrices."""
    gnm_np = self.gnms[version]
    world_to_camera = np.zeros((*batch_dims, 4, 4), dtype=np.float32)
    image = render_gnm.render_gnm(
        gnm_np, world_to_camera=world_to_camera, **self.rendering_kwargs
    )
    self.assertEqual(image.shape, (*batch_dims, *self.image_dims))

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_dims=list(BATCH_DIMS),
  )
  def test_batch_camera_to_image(self, version, batch_dims: _TupleOfInts):
    """Tests we can render with a batch of camera-to-image matrices."""
    gnm_np = self.gnms[version]
    camera_to_image = np.zeros((*batch_dims, 4, 4), dtype=np.float32)
    image = render_gnm.render_gnm(
        gnm_np, camera_to_image=camera_to_image, **self.rendering_kwargs
    )
    self.assertEqual(image.shape, (*batch_dims, *self.image_dims))

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_dims=list(BATCH_DIMS),
  )
  def test_batch_background_image(self, version, batch_dims: _TupleOfInts):
    """Tests we can render with a batch of background images."""
    gnm_np = self.gnms[version]
    background_image = np.zeros(
        (*batch_dims, *self.image_dims), dtype=np.float32
    )
    image = render_gnm.render_gnm(
        gnm_np, background_color=background_image, **self.rendering_kwargs
    )
    self.assertEqual(image.shape, (*batch_dims, *self.image_dims))

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_dims=list(BATCH_DIMS),
  )
  def test_batch_texture(self, version, batch_dims: _TupleOfInts):
    """Tests we can render with a batch of textures."""
    gnm_np = self.gnms[version]
    texture = np.zeros((*batch_dims, 128, 128, 3), dtype=np.float32)
    image = render_gnm.render_gnm(
        gnm_np, texture=texture, **self.rendering_kwargs
    )
    self.assertEqual(image.shape, (*batch_dims, *self.image_dims))

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_error_on_batch_mismatch(self, version):
    """Tests error if batch dimensions don't match."""
    gnm_np = self.gnms[version]
    vertices_shape = gnm_np.template_vertex_positions.shape
    vertices = np.zeros((5, 10, *vertices_shape), dtype=np.float32)
    world_to_camera = np.zeros((6, 4, 4), dtype=np.float32)
    with self.assertRaisesRegex(ValueError, 'Batch dimensions incompatible'):
      render_gnm.render_gnm(
          gnm_np,
          vertices=vertices,
          world_to_camera=world_to_camera,
          **self.rendering_kwargs,
      )


class TestGetBatchDim(parameterized.TestCase):
  """Tests for _get_batch_dim helper."""

  def test_get_batch_dim(self):
    a, b, c = 1, 2, 3
    array_a = np.zeros((a, b, c, 10))
    array_b = None
    array_c = np.zeros((b, c, 5, 5))
    batch_dims = render_gnm._get_batch_dim(
        (array_a, 1), (array_b, 2), (array_c, 2)
    )
    self.assertEqual(batch_dims, (a, b, c))

  def test_get_batch_dim_raises_error(self):
    array_a = np.zeros((1, 2, 3, 10))
    array_b = np.zeros((4, 5, 6, 5))
    with self.assertRaises(ValueError):
      render_gnm._get_batch_dim((array_a, 1), (array_b, 1))


class TestProjectPointsForGNM(parameterized.TestCase):
  """Tests projection of points for GNM."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {}
    for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

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

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_project_points_default_render(self, version):
    """Tests project_points_for_gnm with default render."""
    gnm_np = self.gnms[version]

    # Default render.
    image = render_gnm.render_gnm(gnm_np, **self.rendering_kwargs)

    # Project face joints under the same camera setup.
    proj_fn = (
        render_gnm.project_points_for_gnm  # pytype: disable=module-attr
    )
    joints_image = proj_fn(
        gnm_np=gnm_np,
        points_world=gnm_np.template_joint_positions,
        **self.rendering_kwargs,
    )

    # Project a point above the face under the same camera setup.
    points_world = np.array(
        [[0, gnm_np.template_vertex_positions[:, 1].max() + 0.05, 0]]
    )
    proj_fn = (
        render_gnm.project_points_for_gnm  # pytype: disable=module-attr
    )
    external_point_image = proj_fn(
        gnm_np=gnm_np,
        points_world=points_world,
        **self.rendering_kwargs,
    )

    mask = (image > 0).any(axis=-1)

    with self.subTest('Face joints in mask'):
      x, y = joints_image.T.astype(np.int32)
      self.assertTrue(mask[y, x].all())  # pyrefly: ignore[bad-index]

    with self.subTest('Point above face not in mask'):
      x, y = external_point_image.T.astype(np.int32)
      self.assertFalse(mask[y, x].all())  # pyrefly: ignore[bad-index]

    # Draw points on the image and save.
    image = (image * 255).astype(np.uint8)
    for x, y in joints_image:
      cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1, cv2.LINE_AA)
    for x, y in external_point_image:
      cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1, cv2.LINE_AA)

    _write_images(self.outputs_dir, 'project_points', image[None, :])

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_project_points_spin(self, version):
    """Tests projection of points for GNM in a spin."""
    gnm_np = self.gnms[version]
    spin_period = 30
    world_to_camera = render_gnm.get_spin_world_to_camera(
        gnm_np=gnm_np,
        vertices=gnm_np.template_vertex_positions,
        spin_period=spin_period,
    )

    image = render_gnm.render_gnm(
        gnm_np, world_to_camera=world_to_camera, **self.rendering_kwargs
    )

    # Project face joints under the same camera setup.
    proj_fn = (
        render_gnm.project_points_for_gnm  # pytype: disable=module-attr
    )
    joints_image = proj_fn(
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
          is_in_mask = (
              mask[i, y[i], x[i]].all()  # pyrefly: ignore[bad-index]
          )
          self.assertTrue(is_in_mask)

    # Draw points on the image and save.
    image = (image * 255).astype(np.uint8)
    for i in range(spin_period):
      for x, y in joints_image[i]:
        cv2.circle(image[i], (int(x), int(y)), 5, (0, 255, 0), -1, cv2.LINE_AA)

    _write_gif(
        self.outputs_dir, f'project_points_spin_{version}', image, fps=30
    )


class TestGetLookAtWorldToCamera(parameterized.TestCase):
  """Tests for get_look_at_world_to_camera."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {}
    for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_basic(self, version):
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions
    world_to_camera_opencv = render_gnm.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
    )
    world_to_camera_opengl = camera_conversions.opencv_extrinsics_to_opengl(
        world_to_camera_opencv
    )
    self.assertEqual(world_to_camera_opengl.shape, (4, 4))

    with self.subTest('Approximately identity rotation.'):
      np.testing.assert_allclose(
          world_to_camera_opengl[:3, :3], np.eye(3), atol=0.02
      )

    with self.subTest('Translated from hockey mask.'):
      hockey_mask_indices = gnm_np.vertex_group_indices('hockey_mask')
      hockey_mask_z = vertices[hockey_mask_indices, 2].mean()
      self.assertAlmostEqual(
          -world_to_camera_opengl[2, 3],
          hockey_mask_z + render_gnm._DEFAULT_CAMERA_DISTANCE,
          delta=0.01,
      )

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_share_camera_no_batch(self, version):
    gnm_np = self.gnms[version]
    parameters = _get_random_parameters((), gnm_np)
    vertices = gnm_np(**parameters)
    world_to_camera_no_share = render_gnm.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
        share_camera=False,
    )

    world_to_camera_share = render_gnm.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
        share_camera=True,
    )

    np.testing.assert_allclose(world_to_camera_no_share, world_to_camera_share)

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_share_camera(self, version):
    gnm_np = self.gnms[version]
    parameters = _get_random_parameters((10,), gnm_np)
    vertices = gnm_np(**parameters)

    world_to_camera_0 = render_gnm.get_look_at_world_to_camera(
        gnm_np=gnm_np, vertices_world=vertices[0]
    )

    world_to_camera_share = render_gnm.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
        share_camera=True,
    )

    world_to_camera_no_share = render_gnm.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
        share_camera=False,
    )

    with self.subTest('Shared camera matches first camera.'):
      np.testing.assert_allclose(
          world_to_camera_share,
          np.broadcast_to(world_to_camera_0, world_to_camera_share.shape),
      )

    with self.subTest('Unshared cameras are all different.'):
      self.assertFalse(
          np.allclose(
              world_to_camera_no_share,
              np.broadcast_to(
                  world_to_camera_0, world_to_camera_no_share.shape
              ),
          )
      )


class TestGetFillFactorCameraToImage(parameterized.TestCase):
  """Tests for get_fill_factor_camera_to_image."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {}
    for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

  @parameterized.named_parameters(
      *[(version, version) for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS]
  )
  def test_basic(self, version):
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions
    camera_to_image_opencv = render_gnm.get_fill_factor_camera_to_image(
        gnm_np=gnm_np,
        vertices=vertices,
    )
    camera_to_image_opengl = (
        camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
            camera_to_image_opencv,
            width=320,
            height=240,
            near=0.1,
            far=100.0,
        )
    )
    self.assertEqual(camera_to_image_opengl.shape, (4, 4))
    with self.subTest('Lower triangular is all zeros.'):
      submatrix = camera_to_image_opengl[:3, :3]
      self.assertTrue((np.tril(submatrix, k=-1) == 0.0).all())


if __name__ == '__main__':
  absltest.main()
