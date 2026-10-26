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

"""Tests for backend-agnostic helpers in render_common."""

import functools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
from gnm.shape import gnm_test_utils
from gnm.shape import gnm_udim
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import render_common
import numpy as np

_TupleOfInts = tuple[int, ...]


@functools.cache
def _load_head_gnms() -> dict[str, gnm_numpy.GNM]:
  """Loads the GNM head models once, shared by all test classes."""
  return {
      version: gnm_test_utils.load_gnm(
          gnm_numpy.GNM, version, gnm_numpy.GNMVariant.HEAD
      )
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  }


def _get_random_parameters(
    batch_dims: _TupleOfInts,
    gnm_np: gnm_numpy.GNM,
) -> dict[str, np.ndarray]:
  """Returns random GNM parameters.

  Args:
    batch_dims: The batch dimensions.
    gnm_np: The GNM model.

  Returns:
    A dictionary of random GNM parameters.
  """
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


class TestGetBatchDim(parameterized.TestCase):
  """Tests for get_batch_dim."""

  def test_get_batch_dim(self):
    """Tests that get_batch_dim returns the correct batch dimensions."""
    a, b, c = 1, 2, 3
    array_a = np.zeros((a, b, c, 10))
    array_b = None
    array_c = np.zeros((b, c, 5, 5))
    batch_dims = render_common.get_batch_dim(
        (array_a, 1), (array_b, 2), (array_c, 2)
    )
    self.assertEqual(batch_dims, (a, b, c))

  def test_get_batch_dim_raises_error(self):
    """Tests that get_batch_dim raises an error for incompatible dimensions."""
    array_a = np.zeros((1, 2, 3, 10))
    array_b = np.zeros((4, 5, 6, 5))
    with self.assertRaises(ValueError):
      render_common.get_batch_dim((array_a, 1), (array_b, 1))

  def test_get_batch_dim_broadcast_singleton(self):
    """Tests that get_batch_dim properly broadcasts singleton dimensions."""
    array_a = np.zeros((3, 1, 10))
    array_b = np.zeros((1, 4, 10))
    batch_dims = render_common.get_batch_dim((array_a, 1), (array_b, 1))
    self.assertEqual(batch_dims, (3, 4))


class TestGetLookAtWorldToCamera(parameterized.TestCase):
  """Tests for get_look_at_world_to_camera."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = _load_head_gnms()

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_basic(self, version):
    """Tests that get_look_at_world_to_camera returns the correct matrix."""
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions
    world_to_camera_opencv = render_common.get_look_at_world_to_camera(
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
          hockey_mask_z + render_common.DEFAULT_CAMERA_DISTANCE,
          delta=0.01,
      )

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_share_camera_no_batch(self, version):
    """Tests that share_camera=False has no effect for no batch dimensions."""
    gnm_np = self.gnms[version]
    parameters = _get_random_parameters((), gnm_np)
    vertices = gnm_np(**parameters)
    world_to_camera_no_share = render_common.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
        share_camera=False,
    )

    world_to_camera_share = render_common.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
        share_camera=True,
    )

    np.testing.assert_allclose(world_to_camera_no_share, world_to_camera_share)

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_share_camera(self, version):
    """Tests that share_camera=True/False have the correct effect."""
    gnm_np = self.gnms[version]
    parameters = _get_random_parameters((10,), gnm_np)
    vertices = gnm_np(**parameters)

    world_to_camera_0 = render_common.get_look_at_world_to_camera(
        gnm_np=gnm_np, vertices_world=vertices[0]
    )

    world_to_camera_share = render_common.get_look_at_world_to_camera(
        gnm_np=gnm_np,
        vertices_world=vertices,
        share_camera=True,
    )

    world_to_camera_no_share = render_common.get_look_at_world_to_camera(
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
    cls.gnms = _load_head_gnms()

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_basic(self, version):
    """Tests that get_fill_factor_camera_to_image returns the correct matrix."""
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions
    camera_to_image_opencv = render_common.get_fill_factor_camera_to_image(
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


class TestLoadTexture(parameterized.TestCase):
  """Tests for load_texture."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = _load_head_gnms()

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_load_default_texture(self, version):
    """Tests that load_texture returns the default texture for the skin tile."""
    gnm_np = self.gnms[version]
    textures = render_common.load_texture(gnm_np)
    self.assertEqual(list(textures), [render_common.SKIN_UDIM_TILE])
    skin_tex = textures[render_common.SKIN_UDIM_TILE]
    self.assertEqual(skin_tex.dtype, np.uint8)
    self.assertEqual(skin_tex.shape[-1], 3)

  def test_load_custom_numpy_texture(self):
    """Tests that load_texture returns the custom texture."""
    gnm_np = self.gnms[gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]]
    custom_tex = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    textures = render_common.load_texture(gnm_np, texture=custom_tex)
    self.assertEqual(list(textures), [render_common.SKIN_UDIM_TILE])
    skin_tex = textures[render_common.SKIN_UDIM_TILE]
    self.assertEqual(skin_tex.dtype, np.uint8)
    self.assertEqual(skin_tex.shape, (32, 32, 3))

  def test_load_no_texture(self):
    """Tests that load_texture returns no tile for texture=None."""
    gnm_np = self.gnms[gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]]
    self.assertEmpty(render_common.load_texture(gnm_np, texture=None))

  @parameterized.named_parameters(('int', int), ('numpy_int', np.int64))
  def test_load_tile_keyed_texture(self, key_type):
    """Tests that textures keyed by (numpy) tile numbers map to their tile."""
    gnm_np = self.gnms[gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]]
    tile = gnm_udim.UdimTile.TONGUE
    tongue_tex = np.zeros((8, 8, 3), dtype=np.float32)
    textures = render_common.load_texture(gnm_np, {key_type(tile): tongue_tex})
    self.assertEqual(list(textures), [tile])
    np.testing.assert_array_equal(textures[tile], 0)

  def test_load_part_name_texture_is_deprecated(self):
    """Tests that part-named textures warn and cover all tiles of the part."""
    gnm_np = self.gnms[gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]]
    eye_tex = np.zeros((8, 8, 3), dtype=np.float32)
    with self.assertWarns(DeprecationWarning):
      textures = render_common.load_texture(gnm_np, {'left_eye': eye_tex})
    tiles = [
        gnm_udim.UdimTile.LEFT_EYE_INTERIOR,
        gnm_udim.UdimTile.LEFT_EYE_EXTERIOR,
    ]
    self.assertCountEqual(textures, tiles)
    for tile in tiles:
      np.testing.assert_array_equal(textures[tile], 0)
    with self.assertRaisesRegex(ValueError, "'wrong_part' is not a GNM part"):
      render_common.load_texture(gnm_np, {'wrong_part': eye_tex})


class TestProjectPointsForGnm(parameterized.TestCase):
  """Tests for project_points_for_gnm."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = _load_head_gnms()

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_project_points(self, version):
    """Tests that project_points_for_gnm returns the correct points."""
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions
    image_size = (240, 320)
    projected = render_common.project_points_for_gnm(
        gnm_np=gnm_np,
        vertices=vertices,
        image_size=image_size,
    )
    self.assertEqual(projected.shape, (vertices.shape[0], 2))
    # Points should roughly fall within the image boundary.
    self.assertTrue((projected[:, 0] >= -100).all())
    self.assertTrue((projected[:, 0] <= image_size[0] + 100).all())
    self.assertTrue((projected[:, 1] >= -100).all())
    self.assertTrue((projected[:, 1] <= image_size[1] + 100).all())


class TestGetSpinWorldToCamera(parameterized.TestCase):
  """Tests for get_spin_world_to_camera."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = _load_head_gnms()

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_spin_camera(self, version):
    """Tests that get_spin_world_to_camera returns the correct matrices."""
    gnm_np = self.gnms[version]
    vertices = gnm_np.template_vertex_positions
    num_frames = 5
    spin_w2c = render_common.get_spin_world_to_camera(
        gnm_np=gnm_np,
        vertices=vertices,
        num_frames=num_frames,
    )
    self.assertEqual(spin_w2c.shape, (num_frames, 4, 4))

    spin_period_w2c = render_common.get_spin_world_to_camera(
        gnm_np=gnm_np,
        vertices=vertices,
        spin_period=num_frames,
    )
    self.assertEqual(spin_period_w2c.shape, (num_frames, 4, 4))


class TestRenderGNMMesh(parameterized.TestCase):
  """Tests for render_gnm_mesh."""

  gnm_np: gnm_numpy.GNM

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnm_np = _load_head_gnms()[
        gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]
    ]

  def test_render_gnm_mesh_mock(self):
    """Tests render_gnm_mesh prepares parameters and calls backend_render_fn."""
    passed_kwargs = {}

    def mock_backend(*args: Any, **kwargs: Any) -> render_common.FloatArray:
      del args
      nonlocal passed_kwargs
      passed_kwargs = kwargs
      n = kwargs['vertices'].shape[0]
      w, h = kwargs['image_size']
      return np.ones((n, h, w, 3), dtype=np.float32)

    image_size = (64, 48)
    res = render_common.render_gnm_mesh(
        gnm_np=self.gnm_np,
        backend_render_fn=mock_backend,
        image_size=image_size,
    )

    self.assertEqual(res.shape, (48, 64, 3))
    self.assertIn('vertices', passed_kwargs)
    self.assertIn('world_to_camera', passed_kwargs)
    self.assertIn('camera_to_image', passed_kwargs)
    self.assertIn('vertex_colors', passed_kwargs)
    selected = self.gnm_np.triangle_indices_for_group('~eye_exteriors')
    np.testing.assert_array_equal(
        passed_kwargs['triangles'], self.gnm_np.triangles[selected]
    )
    np.testing.assert_array_equal(
        passed_kwargs['triangle_uvs'], self.gnm_np.triangle_uvs_udim[selected]
    )
    self.assertEqual(
        list(passed_kwargs['texture']), [render_common.SKIN_UDIM_TILE]
    )
    self.assertEqual(passed_kwargs['image_size'], image_size)

  def test_render_gnm_mesh_convert_cameras(self):
    """Tests that convert_cameras_to_opengl applies OpenGL conversion."""
    passed_w2c = None

    def mock_backend(*args: Any, **kwargs: Any) -> render_common.FloatArray:
      del args
      nonlocal passed_w2c
      passed_w2c = kwargs['world_to_camera']
      n = kwargs['vertices'].shape[0]
      w, h = kwargs['image_size']
      return np.zeros((n, h, w, 3), dtype=np.float32)

    w2c_opencv = np.eye(4, dtype=np.float32)[None, ...]

    # Without conversion
    render_common.render_gnm_mesh(
        gnm_np=self.gnm_np,
        backend_render_fn=mock_backend,
        world_to_camera=w2c_opencv,
        convert_cameras_to_opengl=False,
    )
    assert passed_w2c is not None
    np.testing.assert_allclose(passed_w2c, w2c_opencv)

    # With conversion
    render_common.render_gnm_mesh(
        gnm_np=self.gnm_np,
        backend_render_fn=mock_backend,
        world_to_camera=w2c_opencv,
        convert_cameras_to_opengl=True,
    )
    expected_opengl = camera_conversions.opencv_extrinsics_to_opengl(w2c_opencv)
    assert passed_w2c is not None
    np.testing.assert_allclose(passed_w2c, expected_opengl)

  def test_render_gnm_mesh_errors(self):
    """Tests validation errors in render_gnm_mesh."""

    def mock_backend(*args: Any, **kwargs: Any) -> render_common.FloatArray:
      del args, kwargs
      return np.zeros((1, 10, 10, 3), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, 'multiple_gnms=True, but vertices is only 2D'
    ):
      render_common.render_gnm_mesh(
          gnm_np=self.gnm_np,
          backend_render_fn=mock_backend,
          multiple_gnms=True,
      )

    with self.assertRaisesRegex(ValueError, 'are not UDIM tiles'):
      render_common.render_gnm_mesh(
          gnm_np=self.gnm_np,
          backend_render_fn=mock_backend,
          texture={1099: np.zeros((10, 10, 3), dtype=np.float32)},
      )

    with self.assertRaisesRegex(
        ValueError, 'Vertex colors must have 3 or 4 channels'
    ):
      render_common.render_gnm_mesh(
          gnm_np=self.gnm_np,
          backend_render_fn=mock_backend,
          vertex_colors=np.zeros(
              (*self.gnm_np.template_vertex_positions.shape[:-1], 2),
              dtype=np.float32,
          ),
      )

    with self.assertRaisesRegex(ValueError, 'Batch dimensions incompatible'):
      render_common.render_gnm_mesh(
          gnm_np=self.gnm_np,
          backend_render_fn=mock_backend,
          vertices=np.zeros(
              (5, 10, *self.gnm_np.template_vertex_positions.shape),
              dtype=np.float32,
          ),
          world_to_camera=np.zeros((6, 4, 4), dtype=np.float32),
      )

  def test_render_gnm_mesh_broadcast_singleton_batch_dims(self):
    """Tests that render_gnm_mesh properly broadcasts singleton dimensions."""

    def mock_backend(*args: Any, **kwargs: Any) -> render_common.FloatArray:
      del args
      n = kwargs['vertices'].shape[0]
      w, h = kwargs['image_size']
      return np.zeros((n, h, w, 3), dtype=np.float32)

    image_size = (64, 48)
    res = render_common.render_gnm_mesh(
        gnm_np=self.gnm_np,
        backend_render_fn=mock_backend,
        vertices=np.zeros(
            (3, 1, *self.gnm_np.template_vertex_positions.shape),
            dtype=np.float32,
        ),
        world_to_camera=np.zeros((1, 4, 4, 4), dtype=np.float32),
        image_size=image_size,
    )
    self.assertEqual(res.shape, (3, 4, 48, 64, 3))


class TestMeshRenderOutput(parameterized.TestCase):
  """Tests for MeshRenderOutput dataclass and compositing."""

  def test_composite_with_default_background(self):
    """Tests compositing with default solid background."""
    image = np.full((10, 20, 3), 200, dtype=np.uint8)
    alpha = np.ones((10, 20, 1), dtype=np.float32)
    output = render_common.MeshRenderOutput(image=image, alpha=alpha)

    composited = output.composite()
    self.assertEqual(composited.shape, (10, 20, 3))
    self.assertEqual(composited.dtype, np.uint8)
    # With alpha=1.0, foreground should match exactly.
    np.testing.assert_array_equal(composited, image)

    # With alpha=0.0, background should match DEFAULT_BACKGROUND_COLOR (0.95).
    composited_background = output.composite(alpha=0.0)
    expected_background = int(render_common.DEFAULT_BACKGROUND_COLOR[0] * 255.0)
    np.testing.assert_allclose(
        composited_background, expected_background, atol=1
    )

  def test_composite_with_custom_background(self):
    """Tests compositing with custom background image and 1D uint8 RGB color."""
    image = np.full((12, 16, 3), 255, dtype=np.uint8)
    alpha = np.zeros((12, 16, 1), dtype=np.float32)
    output = render_common.MeshRenderOutput(image=image, alpha=alpha)

    background = np.full((12, 16, 3), 50, dtype=np.uint8)
    composited = output.composite(background=background)
    self.assertEqual(composited.shape, (12, 16, 3))
    # With alpha=0.0, output should match background.
    np.testing.assert_array_equal(composited, background)

    # 1D uint8 RGB background color array is normalized to [0, 1] before blend.
    background_color_uint8 = np.array([255, 128, 0], dtype=np.uint8)
    composited_1d_uint8 = output.composite(background=background_color_uint8)
    expected_1d_uint8 = np.broadcast_to(background_color_uint8, (12, 16, 3))
    np.testing.assert_allclose(composited_1d_uint8, expected_1d_uint8, atol=1)

  def test_composite_with_alpha_scaling(self):
    """Tests compositing with alpha=0.5 scaling."""
    image = np.full((4, 4, 3), 200, dtype=np.uint8)
    alpha = np.ones((4, 4, 1), dtype=np.float32)
    output = render_common.MeshRenderOutput(image=image, alpha=alpha)

    background = np.full((4, 4, 3), 100, dtype=np.uint8)
    composited = output.composite(background=background, alpha=0.5)
    # Expected: 200 * 0.5 + 100 * 0.5 = 150
    np.testing.assert_allclose(composited, 150, atol=1)

  def test_composite_float_dtype_with_colors_and_images(self):
    """Tests composite with dtype=np.float32 for colors, images, and errors."""
    image = np.full((10, 20, 3), 255, dtype=np.uint8)
    alpha = np.ones((10, 20, 1), dtype=np.float32)
    output = render_common.MeshRenderOutput(image=image, alpha=alpha)

    # Default background color with alpha=1.0.
    composited_default = output.composite(dtype=np.float32)
    self.assertEqual(composited_default.shape, (10, 20, 3))
    self.assertEqual(composited_default.dtype, np.float32)
    np.testing.assert_allclose(composited_default, 1.0, atol=1e-5)

    # Default background color with alpha=0.0.
    composited_background = output.composite(alpha=0.0, dtype=np.float32)
    np.testing.assert_allclose(
        composited_background,
        np.broadcast_to(render_common.DEFAULT_BACKGROUND_COLOR, (10, 20, 3)),
        atol=1e-5,
    )

    # Float32 background image.
    background_float = np.full((10, 20, 3), 0.25, dtype=np.float32)
    composited_float = output.composite(
        background=background_float, alpha=0.5, dtype=np.float32
    )
    # Expected: 1.0 * 0.5 + 0.25 * 0.5 = 0.625
    np.testing.assert_allclose(composited_float, 0.625, atol=1e-5)

    # UInt8 background image.
    background_uint8 = np.full((10, 20, 3), 51, dtype=np.uint8)
    composited_uint8 = output.composite(
        background=background_uint8, alpha=0.0, dtype=np.float32
    )
    np.testing.assert_allclose(composited_uint8, 51.0 / 255.0, atol=1e-5)

    # Batched background image.
    batched_image = np.full((2, 10, 20, 3), 255, dtype=np.uint8)
    batched_alpha = np.ones((2, 10, 20, 1), dtype=np.float32)
    batched_output = render_common.MeshRenderOutput(
        image=batched_image, alpha=batched_alpha
    )
    batched_background = np.zeros((2, 10, 20, 3), dtype=np.float32)
    composited_batched = batched_output.composite(
        background=batched_background, alpha=0.5, dtype=np.float32
    )
    self.assertEqual(composited_batched.shape, (2, 10, 20, 3))
    np.testing.assert_allclose(composited_batched, 0.5, atol=1e-5)

    # Incompatible batch dimensions.
    incompatible_background = np.zeros((3, 10, 20, 3), dtype=np.float32)
    with self.assertRaisesRegex(ValueError, 'Batch dimensions incompatible'):
      batched_output.composite(
          background=incompatible_background, dtype=np.float32
      )

    # Invalid dtype.
    with self.assertRaisesRegex(ValueError, 'Unsupported output dtype'):
      output.composite(dtype=np.int32)


class TestResolveGNMTriangles(parameterized.TestCase):
  """Tests for resolve_gnm_triangles."""

  gnm_np: gnm_numpy.GNM

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnm_np = _load_head_gnms()[
        gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]
    ]

  def test_group_string_input(self):
    """Tests resolving group string expression."""
    expected_indices = self.gnm_np.triangle_indices_for_group('~eye_exteriors')
    selected_triangles, selected_triangle_uvs = (
        render_common.resolve_gnm_triangles(self.gnm_np, '~eye_exteriors')
    )
    self.assertEqual(selected_triangles.dtype, np.int32)
    self.assertEqual(selected_triangle_uvs.dtype, np.float32)
    self.assertEqual(selected_triangles.shape, (len(expected_indices), 3))
    self.assertEqual(selected_triangle_uvs.shape, (len(expected_indices), 3, 2))
    np.testing.assert_array_equal(
        selected_triangles, self.gnm_np.triangles[expected_indices]
    )
    np.testing.assert_allclose(
        selected_triangle_uvs, self.gnm_np.triangle_uvs_udim[expected_indices]
    )

  def test_sequence_of_groups_input(self):
    """Tests resolving sequence of group expressions."""
    group_names = ('skin', '-left_eye')
    expected_indices = self.gnm_np.triangle_indices_for_group(*group_names)
    selected_triangles, selected_triangle_uvs = (
        render_common.resolve_gnm_triangles(self.gnm_np, group_names)
    )
    self.assertEqual(selected_triangles.dtype, np.int32)
    self.assertEqual(selected_triangle_uvs.dtype, np.float32)
    self.assertEqual(selected_triangles.shape, (len(expected_indices), 3))
    self.assertEqual(selected_triangle_uvs.shape, (len(expected_indices), 3, 2))
    np.testing.assert_array_equal(
        selected_triangles, self.gnm_np.triangles[expected_indices]
    )
    np.testing.assert_allclose(
        selected_triangle_uvs, self.gnm_np.triangle_uvs_udim[expected_indices]
    )

  def test_face_indices_array_input(self):
    """Tests resolving 1D face indices array."""
    face_indices = np.array([0, 1, 2], dtype=np.int32)
    selected_triangles, selected_triangle_uvs = (
        render_common.resolve_gnm_triangles(self.gnm_np, face_indices)
    )
    self.assertEqual(selected_triangles.dtype, np.int32)
    self.assertEqual(selected_triangle_uvs.dtype, np.float32)
    self.assertEqual(selected_triangles.shape, (3, 3))
    self.assertEqual(selected_triangle_uvs.shape, (3, 3, 2))
    np.testing.assert_array_equal(
        selected_triangles, self.gnm_np.triangles[face_indices]
    )
    np.testing.assert_allclose(
        selected_triangle_uvs, self.gnm_np.triangle_uvs_udim[face_indices]
    )


class TestNormalizeTextures(parameterized.TestCase):
  """Tests for _normalize_textures."""

  def test_bare_array_maps_to_skin_udim_tile(self):
    """Tests that a single image array maps to SKIN_UDIM_TILE."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    normalized = render_common._normalize_textures(
        image, tiles=[int(render_common.SKIN_UDIM_TILE)]
    )
    self.assertIn(int(render_common.SKIN_UDIM_TILE), normalized)
    np.testing.assert_array_equal(
        normalized[int(render_common.SKIN_UDIM_TILE)], image
    )

  def test_mapping_with_udim_tile_keys(self):
    """Tests that mapping with UdimTile keys converts keys to int."""
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mapping = {gnm_udim.UdimTile.SKIN: image}
    normalized = render_common._normalize_textures(
        mapping, tiles=[int(gnm_udim.UdimTile.SKIN)]
    )
    self.assertIn(int(gnm_udim.UdimTile.SKIN), normalized)
    np.testing.assert_array_equal(
        normalized[int(gnm_udim.UdimTile.SKIN)], image
    )

  def test_missing_tile_filled_with_white(self):
    """Tests that missing tiles in tiles sequence are filled with white."""
    normalized = render_common._normalize_textures(None, tiles=[1001, 1002])
    self.assertIn(1001, normalized)
    self.assertIn(1002, normalized)
    self.assertEqual(normalized[1001].shape, (1, 1, 3))
    self.assertEqual(normalized[1001].dtype, np.uint8)
    np.testing.assert_array_equal(
        normalized[1001], np.full((1, 1, 3), 255, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        normalized[1002], np.full((1, 1, 3), 255, dtype=np.uint8)
    )


class TestResolveGNMCameraAndVertices(parameterized.TestCase):
  """Tests for resolve_gnm_camera_and_vertices."""

  gnm_np: gnm_numpy.GNM

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnm_np = _load_head_gnms()[
        gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]
    ]

  def test_defaults(self):
    """Tests default resolution using GNM template vertices and cameras."""
    vertices, world_to_camera, camera_to_image = (
        render_common.resolve_gnm_camera_and_vertices(self.gnm_np)
    )
    self.assertEqual(
        vertices.shape, self.gnm_np.template_vertex_positions.shape
    )
    self.assertEqual(world_to_camera.shape, (4, 4))
    self.assertEqual(camera_to_image.shape, (4, 4))
    self.assertEqual(vertices.dtype, np.float32)
    self.assertEqual(world_to_camera.dtype, np.float32)
    self.assertEqual(camera_to_image.dtype, np.float32)

  def test_explicit_arguments(self):
    """Tests passing explicit vertices and camera matrices."""
    explicit_vertices = np.zeros((10, 3), dtype=np.float32)
    explicit_world_to_camera = np.eye(4, dtype=np.float32)
    explicit_camera_to_image = np.eye(4, dtype=np.float32)
    vertices, world_to_camera, camera_to_image = (
        render_common.resolve_gnm_camera_and_vertices(
            self.gnm_np,
            vertices=explicit_vertices,
            world_to_camera=explicit_world_to_camera,
            camera_to_image=explicit_camera_to_image,
        )
    )
    np.testing.assert_array_equal(vertices, explicit_vertices)
    np.testing.assert_array_equal(world_to_camera, explicit_world_to_camera)
    np.testing.assert_array_equal(camera_to_image, explicit_camera_to_image)

  def test_multiple_gnms_error(self):
    """Tests that multiple_gnms=True with 2D vertices raises ValueError."""
    with self.assertRaisesRegex(ValueError, 'Called with multiple_gnms=True'):
      render_common.resolve_gnm_camera_and_vertices(
          self.gnm_np,
          vertices=np.zeros((10, 3), dtype=np.float32),
          multiple_gnms=True,
      )

  def test_multiple_gnms_success(self):
    """Tests multiple_gnms=True with 3D vertices."""
    template_vertices = self.gnm_np.template_vertex_positions
    num_vertices = template_vertices.shape[0]
    vertices_3d = np.broadcast_to(template_vertices, (2, num_vertices, 3))
    vertices, world_to_camera, camera_to_image = (
        render_common.resolve_gnm_camera_and_vertices(
            self.gnm_np,
            vertices=vertices_3d,
            multiple_gnms=True,
        )
    )
    self.assertEqual(vertices.shape, (2, num_vertices, 3))
    self.assertEqual(world_to_camera.shape, (4, 4))
    self.assertEqual(camera_to_image.shape, (4, 4))


if __name__ == '__main__':
  absltest.main()
