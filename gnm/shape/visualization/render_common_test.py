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

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import render_common
import numpy as np

_TupleOfInts = tuple[int, ...]


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
    cls.gnms = {}
    for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

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
    cls.gnms = {}
    for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

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
    cls.gnms = {}
    for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

  @parameterized.named_parameters(*[
      (version, version)
      for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
  ])
  def test_load_default_texture(self, version):
    """Tests that load_texture returns the default texture."""
    gnm_np = self.gnms[version]
    textures = render_common.load_texture(gnm_np)
    self.assertIsInstance(textures, dict)
    self.assertIn('skin', textures)
    skin_tex = textures['skin']
    self.assertEqual(skin_tex.dtype, np.uint8)
    self.assertEqual(skin_tex.shape[-1], 3)

  def test_load_custom_numpy_texture(self):
    """Tests that load_texture returns the custom texture."""
    gnm_np = self.gnms[gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0]]
    custom_tex = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    textures = render_common.load_texture(gnm_np, texture=custom_tex)
    self.assertIn('skin', textures)
    self.assertEqual(textures['skin'].dtype, np.uint8)
    self.assertEqual(textures['skin'].shape, (32, 32, 3))


class TestProjectPointsForGnm(parameterized.TestCase):
  """Tests for project_points_for_gnm."""

  gnms: dict[str, gnm_numpy.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {}
    for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

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
    cls.gnms = {}
    for version in gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS:
      cls.gnms[version] = gnm_numpy.GNM.from_local(
          gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
          gnm_numpy.GNMVariant.HEAD,
      )

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
    cls.gnm_np = gnm_numpy.GNM.from_local(
        gnm_numpy.GNMMajorVersion(
            gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[0].removeprefix('v')
        ),
        gnm_numpy.GNMVariant.HEAD,
    )

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
    self.assertIn('triangles', passed_kwargs)
    self.assertIn('world_to_camera', passed_kwargs)
    self.assertIn('camera_to_image', passed_kwargs)
    self.assertIn('texture', passed_kwargs)
    self.assertIn('vertex_colors', passed_kwargs)
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

    with self.assertRaisesRegex(ValueError, 'are not GNM part names'):
      render_common.render_gnm_mesh(
          gnm_np=self.gnm_np,
          backend_render_fn=mock_backend,
          texture={
              'invalid_component': np.zeros((10, 10, 3), dtype=np.float32)
          },
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

  def test_render_gnm_mesh_forwards_backend_kwargs(self):
    captured_kwargs = {}

    def mock_backend(*args: Any, **kwargs: Any) -> render_common.FloatArray:
      del args
      captured_kwargs.update(kwargs)
      n = kwargs['vertices'].shape[0]
      w, h = kwargs['image_size']
      return np.zeros((n, h, w, 3), dtype=np.float32)

    render_common.render_gnm_mesh(
        gnm_np=self.gnm_np,
        backend_render_fn=mock_backend,
        custom_arg='test_value',
        another_kwarg=42,
    )
    self.assertEqual(captured_kwargs.get('custom_arg'), 'test_value')
    self.assertEqual(captured_kwargs.get('another_kwarg'), 42)


if __name__ == '__main__':
  absltest.main()
