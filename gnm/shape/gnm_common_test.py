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

"""Tests for the backend-agnostic GNM math."""

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from etils import enp
from gnm.shape import gnm_common
import numpy as np

# The array backends supported by the GNM math, as (name, xnp module) pairs.
# Every test is run against all of them so that backend-specific regressions
# are caught. The xnp modules mirror those used by the per-backend GNM classes
# (gnm_numpy, gnm_jax, gnm_tensorflow, gnm_pytorch).
_XNP_BACKENDS = (
    ('numpy', np),
    ('jax', enp.lazy.jnp),
    ('tensorflow', enp.lazy.tnp),
    ('pytorch', enp.lazy.torch),
)


class ArrayHelpersTest(parameterized.TestCase):
  """Tests for the backend-agnostic array helpers."""

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_take_selects_along_axis(self, xnp: enp.NpModule):
    array = xnp.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=xnp.float32)
    indices = xnp.asarray([0, 2], dtype=xnp.int32)
    taken = gnm_common.take(array, indices, axis=0, xnp=xnp)
    np.testing.assert_array_equal(np.asarray(taken), [[0, 1, 2], [6, 7, 8]])

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_take_selects_along_last_axis(self, xnp: enp.NpModule):
    array = xnp.asarray([[0, 1, 2], [3, 4, 5]], dtype=xnp.float32)
    indices = xnp.asarray([2, 0], dtype=xnp.int32)
    taken = gnm_common.take(array, indices, axis=-1, xnp=xnp)
    np.testing.assert_array_equal(np.asarray(taken), [[2, 0], [5, 3]])

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_eye_returns_identity(self, xnp: enp.NpModule):
    reference = xnp.asarray([0.0, 0.0, 0.0, 0.0], dtype=xnp.float32)
    identity = gnm_common.eye(
        3, dtype=xnp.float32, reference_array=reference, xnp=xnp
    )
    self.assertEqual(tuple(identity.shape), (3, 3))
    np.testing.assert_array_equal(np.asarray(identity), np.eye(3))

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_reshape_with_batch_dims(self, xnp: enp.NpModule):
    reference = xnp.asarray(np.zeros((2, 5, 3)), dtype=xnp.float32)
    array = xnp.asarray(np.arange(2 * 6), dtype=xnp.float32)
    reshaped = gnm_common.reshape_with_batch_dims(
        array,
        target_suffix=(6,),
        reference_array=reference,
        num_reference_non_batch_dims=2,
    )
    self.assertEqual(tuple(reshaped.shape), (2, 6))

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_zeros_with_batch_dims(self, xnp: enp.NpModule):
    reference = xnp.asarray(np.ones((4, 5, 3)), dtype=xnp.float32)
    zeros = gnm_common.zeros_with_batch_dims(
        reference,
        num_reference_non_batch_dims=2,
        suffix_shape=(4, 4),
        dtype=xnp.float32,
    )
    self.assertEqual(tuple(zeros.shape), (4, 4, 4))
    np.testing.assert_array_equal(np.asarray(zeros), np.zeros((4, 4, 4)))


class AxisAngleToRotationMatrixTest(parameterized.TestCase):
  """Tests for axis_angle_to_rotation_matrix."""

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_zero_returns_identity(self, xnp: enp.NpModule):
    axis_angle = xnp.asarray([0.0, 0.0, 0.0], dtype=xnp.float32)
    matrix = np.asarray(gnm_common.axis_angle_to_rotation_matrix(axis_angle))
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-5)

  @parameterized.product(
      xnp=tuple(xnp for _, xnp in _XNP_BACKENDS),
      case=(
          ([np.pi / 2, 0.0, 0.0], [1.0, 0.0, 0.0]),
          ([0.0, np.pi / 2, 0.0], [0.0, 1.0, 0.0]),
          ([0.0, 0.0, np.pi / 2], [0.0, 0.0, 1.0]),
      ),
  )
  def test_axis_is_invariant(
      self,
      xnp: enp.NpModule,
      case: tuple[Sequence[float], Sequence[float]],
  ) -> None:
    # Rotating about an axis leaves that axis unchanged.
    axis_angle, axis = case
    matrix = np.asarray(
        gnm_common.axis_angle_to_rotation_matrix(
            xnp.asarray(axis_angle, dtype=xnp.float32)
        )
    )
    np.testing.assert_allclose(matrix @ np.asarray(axis), axis, atol=1e-5)

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_ninety_degrees_about_z(self, xnp: enp.NpModule):
    axis_angle = xnp.asarray([0.0, 0.0, np.pi / 2], dtype=xnp.float32)
    matrix = np.asarray(gnm_common.axis_angle_to_rotation_matrix(axis_angle))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-5)

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_matrix_is_orthonormal_with_unit_determinant(self, xnp: enp.NpModule):
    rng = np.random.default_rng(0)
    axis_angle = rng.uniform(-3.0, 3.0, size=(8, 3))
    matrices = np.asarray(
        gnm_common.axis_angle_to_rotation_matrix(
            xnp.asarray(axis_angle, dtype=xnp.float32)
        )
    )
    self.assertEqual(matrices.shape, (8, 3, 3))
    identity = np.einsum('...ij,...kj->...ik', matrices, matrices)
    np.testing.assert_allclose(
        identity, np.broadcast_to(np.eye(3), (8, 3, 3)), atol=1e-4
    )
    np.testing.assert_allclose(np.linalg.det(matrices), np.ones(8), atol=1e-4)

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_preserves_batch_shape(self, xnp: enp.NpModule):
    axis_angle = xnp.asarray(np.zeros((2, 4, 3)), dtype=xnp.float32)
    matrices = gnm_common.axis_angle_to_rotation_matrix(axis_angle)
    self.assertEqual(tuple(matrices.shape), (2, 4, 3, 3))


class JointTransformsWorldTest(parameterized.TestCase):
  """Tests for the forward-kinematics joint transforms."""

  def setUp(self):
    super().setUp()
    # A simple two-joint chain: joint 1 is a child of the root joint 0.
    self.joints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    self.parents = [0, 0]

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_rest_pose_places_joints_at_bind_positions(self, xnp: enp.NpModule):
    transforms = np.asarray(
        gnm_common.joint_transforms_world(
            xnp.asarray(self.joints, dtype=xnp.float32),
            xnp.asarray(np.zeros((2, 3)), dtype=xnp.float32),
            xnp.asarray(np.zeros(3), dtype=xnp.float32),
            self.parents,
        )
    )
    self.assertEqual(transforms.shape, (2, 4, 4))
    np.testing.assert_allclose(transforms[..., :3, 3], self.joints, atol=1e-5)
    # No rotation, so both rotation blocks are the identity.
    np.testing.assert_allclose(
        transforms[:, :3, :3],
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=1e-5,
    )

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_global_translation_shifts_all_joints(self, xnp: enp.NpModule):
    transforms = np.asarray(
        gnm_common.joint_transforms_world(
            xnp.asarray(self.joints, dtype=xnp.float32),
            xnp.asarray(np.zeros((2, 3)), dtype=xnp.float32),
            xnp.asarray([5.0, 0.0, 0.0], dtype=xnp.float32),
            self.parents,
        )
    )
    np.testing.assert_allclose(
        transforms[..., :3, 3],
        [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        atol=1e-5,
    )

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_root_rotation_propagates_to_child(self, xnp: enp.NpModule):
    # Rotate the root 90 degrees about z; the child at (1, 0, 0) maps to
    # (0, 1, 0).
    rotations = xnp.asarray(
        [[0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.0]], dtype=xnp.float32
    )
    transforms = np.asarray(
        gnm_common.joint_transforms_world(
            xnp.asarray(self.joints, dtype=xnp.float32),
            rotations,
            xnp.asarray(np.zeros(3), dtype=xnp.float32),
            self.parents,
        )
    )
    np.testing.assert_allclose(
        transforms[..., :3, 3][1], [0.0, 1.0, 0.0], atol=1e-5
    )


class LinearBlendSkinningTest(parameterized.TestCase):
  """Tests for linear_blend_skinning."""

  def setUp(self):
    super().setUp()
    self.joints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    self.parents = [0, 0]
    # One-hot skinning weights (J, V): each vertex is bound to a single joint.
    self.weights = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    self.vertices = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 1.0, 0.0]]

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_rest_pose_is_identity(self, xnp: enp.NpModule):
    posed = np.asarray(
        gnm_common.linear_blend_skinning(
            xnp.asarray(self.vertices, dtype=xnp.float32),
            xnp.asarray(self.joints, dtype=xnp.float32),
            xnp.asarray(np.zeros((2, 3)), dtype=xnp.float32),
            xnp.asarray(np.zeros(3), dtype=xnp.float32),
            xnp.asarray(self.weights, dtype=xnp.float32),
            self.parents,
        )
    )
    self.assertEqual(posed.shape, (3, 3))
    np.testing.assert_allclose(posed, self.vertices, atol=1e-5)

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_global_translation_translates_vertices(self, xnp: enp.NpModule):
    translation = [10.0, 0.0, 0.0]
    posed = np.asarray(
        gnm_common.linear_blend_skinning(
            xnp.asarray(self.vertices, dtype=xnp.float32),
            xnp.asarray(self.joints, dtype=xnp.float32),
            xnp.asarray(np.zeros((2, 3)), dtype=xnp.float32),
            xnp.asarray(translation, dtype=xnp.float32),
            xnp.asarray(self.weights, dtype=xnp.float32),
            self.parents,
        )
    )
    np.testing.assert_allclose(
        posed, np.asarray(self.vertices) + translation, atol=1e-5
    )

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_preserves_batch_shape(self, xnp: enp.NpModule):
    batch_vertices = np.array(np.broadcast_to(self.vertices, (4, 3, 3)))
    batch_joints = np.array(np.broadcast_to(self.joints, (4, 2, 3)))
    posed = gnm_common.linear_blend_skinning(
        xnp.asarray(batch_vertices, dtype=xnp.float32),
        xnp.asarray(batch_joints, dtype=xnp.float32),
        xnp.asarray(np.zeros((4, 2, 3)), dtype=xnp.float32),
        xnp.asarray(np.zeros((4, 3)), dtype=xnp.float32),
        xnp.asarray(self.weights, dtype=xnp.float32),
        self.parents,
    )
    self.assertEqual(tuple(posed.shape), (4, 3, 3))


class BindPoseTest(parameterized.TestCase):
  """Tests for the bind-pose vertex and joint helpers."""

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_vertex_positions_none_params_return_template(
      self, xnp: enp.NpModule
  ):
    template = xnp.asarray(np.ones((3, 3)), dtype=xnp.float32)
    identity_basis = xnp.asarray(np.zeros((2, 3, 3)), dtype=xnp.float32)
    expression_basis = xnp.asarray(np.zeros((2, 3, 3)), dtype=xnp.float32)
    result = gnm_common.vertex_positions_bind_pose(
        None, None, template, identity_basis, expression_basis
    )
    np.testing.assert_allclose(np.asarray(result), np.ones((3, 3)), atol=1e-5)

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_vertex_positions_apply_identity_and_expression(
      self, xnp: enp.NpModule
  ):
    template = xnp.asarray(np.zeros((2, 3)), dtype=xnp.float32)
    # (I=1, V=2, 3)
    identity_basis = xnp.asarray(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=xnp.float32
    )
    # (E=1, V=2, 3)
    expression_basis = xnp.asarray(
        [[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=xnp.float32
    )
    result = gnm_common.vertex_positions_bind_pose(
        xnp.asarray([3.0], dtype=xnp.float32),
        xnp.asarray([1.0], dtype=xnp.float32),
        template,
        identity_basis,
        expression_basis,
    )
    expected = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    np.testing.assert_allclose(np.asarray(result), expected, atol=1e-5)

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_joint_positions_none_identity_returns_template(
      self, xnp: enp.NpModule
  ):
    template = xnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=xnp.float32
    )
    joint_basis = xnp.asarray(np.zeros((2, 2, 3)), dtype=xnp.float32)
    result = gnm_common.joint_positions_bind_pose(None, template, joint_basis)
    np.testing.assert_allclose(
        np.asarray(result), [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-5
    )

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_joint_positions_apply_identity(self, xnp: enp.NpModule):
    template = xnp.asarray(np.zeros((2, 3)), dtype=xnp.float32)
    # (I=1, J=2, 3)
    joint_basis = xnp.asarray(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=xnp.float32
    )
    result = gnm_common.joint_positions_bind_pose(
        xnp.asarray([2.0], dtype=xnp.float32), template, joint_basis
    )
    np.testing.assert_allclose(
        np.asarray(result), [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], atol=1e-5
    )


class PoseCorrectivesTest(parameterized.TestCase):
  """Tests for compute_pose_correctives."""

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_none_inputs_return_zeros(self, xnp: enp.NpModule):
    template = xnp.asarray(np.ones((3, 3)), dtype=xnp.float32)
    result = gnm_common.compute_pose_correctives(
        None, None, template, num_joints=2, num_vertices=3
    )
    self.assertEqual(tuple(result.shape), (3, 3))
    np.testing.assert_array_equal(np.asarray(result), np.zeros((3, 3)))

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_zero_rotations_produce_no_correctives(self, xnp: enp.NpModule):
    template = xnp.asarray(np.ones((3, 3)), dtype=xnp.float32)
    regressor = xnp.asarray(np.ones((2 * 9, 3 * 3)), dtype=xnp.float32)
    result = gnm_common.compute_pose_correctives(
        xnp.asarray(np.zeros((2, 3)), dtype=xnp.float32),
        regressor,
        template,
        num_joints=2,
        num_vertices=3,
    )
    self.assertEqual(tuple(result.shape), (3, 3))
    np.testing.assert_allclose(np.asarray(result), np.zeros((3, 3)), atol=1e-5)

  @parameterized.named_parameters(*_XNP_BACKENDS)
  def test_preserves_batch_shape(self, xnp: enp.NpModule):
    template = xnp.asarray(np.ones((3, 3)), dtype=xnp.float32)
    regressor = xnp.asarray(np.ones((2 * 9, 3 * 3)), dtype=xnp.float32)
    result = gnm_common.compute_pose_correctives(
        xnp.asarray(np.zeros((4, 2, 3)), dtype=xnp.float32),
        regressor,
        template,
        num_joints=2,
        num_vertices=3,
    )
    self.assertEqual(tuple(result.shape), (4, 3, 3))


class TestConstantBatch(parameterized.TestCase):
  """Tests for _is_constant_batch."""

  def test_single_or_empty_batch(self):
    self.assertTrue(gnm_common._is_constant_batch(np.ones((1, 5)), np))
    self.assertTrue(gnm_common._is_constant_batch(np.ones((5,)), np))
    self.assertTrue(gnm_common._is_constant_batch(np.ones((4, 0)), np))

  def test_constant_batch_detected(self):
    tiled = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
    self.assertTrue(gnm_common._is_constant_batch(tiled, np))

    broadcast = np.broadcast_to(np.array([1.0, 2.0, 3.0]), (10, 3))
    self.assertTrue(gnm_common._is_constant_batch(broadcast, np))

  def test_varying_batch_not_constant(self):
    varying = np.arange(20.0).reshape(10, 2)
    self.assertFalse(gnm_common._is_constant_batch(varying, np))

    middle_diff = np.tile(np.array([1.0, 2.0]), (5, 1))
    middle_diff[2] = [3.0, 4.0]
    self.assertFalse(gnm_common._is_constant_batch(middle_diff, np))

  def test_non_numpy_backend_returns_false(self):
    mock_xnp = enp.lazy.jnp
    self.assertFalse(gnm_common._is_constant_batch(np.ones((10, 3)), mock_xnp))


class ApplyLinearBasisTest(parameterized.TestCase):
  """Tests for _apply_linear_basis."""

  def test_apply_linear_basis_constant(self):
    params = np.tile(np.array([1.0, 2.0]), (4, 1)).astype(np.float32)
    basis = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    expected = np.einsum('...i,ijk->...jk', params, basis)
    result = gnm_common._apply_linear_basis(params, basis, np)
    np.testing.assert_allclose(result, expected)

  def test_apply_linear_basis_varying(self):
    params = np.arange(8, dtype=np.float32).reshape(4, 2)
    basis = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    expected = np.einsum('...i,ijk->...jk', params, basis)
    result = gnm_common._apply_linear_basis(params, basis, np)
    np.testing.assert_allclose(result, expected)

  def test_apply_linear_basis_empty_features(self):
    params = np.zeros((2, 3, 0), dtype=np.float32)
    basis = np.zeros((0, 4, 3), dtype=np.float32)
    expected = np.einsum('...i,ijk->...jk', params, basis)
    result = gnm_common._apply_linear_basis(params, basis, np)
    np.testing.assert_allclose(result, expected)
    self.assertEqual(result.shape, (2, 3, 4, 3))


if __name__ == '__main__':
  absltest.main()
