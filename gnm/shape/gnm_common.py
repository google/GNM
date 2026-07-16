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

"""Backend-agnostic core math functions of the GNM model using etils.enp."""

from typing import Any, Sequence
from etils import enp

enpt = enp.typing

_EPSILON = 1e-8


def take(
    array: enpt.FloatArray['...'],
    indices: enpt.IntArray['...'],
    axis: int = 0,
    xnp: Any = None,
) -> enpt.FloatArray['...']:
  """Extracts elements from an array along a specified axis.

  This is a backend-agnostic wrapper around backend specific operations.
  Specifically, on PyTorch it uses index_select, while on other backends it uses
  the backend's take.

  Args:
    array: The source array to select values from.
    indices: 1D or multi-dimensional array of indices to extract.
    axis: The axis along which to select values.
    xnp: Optional array module (e.g. numpy, jax.numpy, torch) to use. If not
      provided, it is auto-detected from the array argument.

  Returns:
    The array with selected indices along the specified axis.
  """
  if xnp is None:
    xnp = enp.lazy.get_xnp(array)
  if enp.lazy.is_torch_xnp(xnp):
    axis = axis % array.ndim
    indices_shape = indices.shape
    if len(indices_shape) > 1:
      indices_flat = indices.reshape(-1)
      taken = xnp.index_select(array, axis, indices_flat)
      out_shape = array.shape[:axis] + indices_shape + array.shape[axis + 1 :]
      return taken.reshape(out_shape)
    else:
      return xnp.index_select(array, axis, indices)
  else:
    return xnp.take(array, indices, axis=axis)


def eye(
    size: int,
    dtype: Any,
    reference_array: enpt.FloatArray['...'],
    xnp: Any = None,
) -> enpt.FloatArray['...']:
  """Creates a 2D identity matrix.

  Args:
    size: The number of rows (and columns) in the output matrix.
    dtype: The data type of the output matrix.
    reference_array: An array used to determine the device (e.g. for PyTorch).
    xnp: Optional array module (e.g. numpy, jax.numpy, torch) to use. If not
      provided, it is auto-detected from reference_array.

  Returns:
    A 2D identity matrix of shape (size, size).
  """
  if xnp is None:
    xnp = enp.lazy.get_xnp(reference_array)
  if enp.lazy.is_torch(reference_array):
    return xnp.eye(size, dtype=dtype, device=reference_array.device)
  else:
    return xnp.eye(size, dtype=dtype)


def reshape_with_batch_dims(
    array: enpt.FloatArray['...'],
    target_suffix: tuple[int, ...],
    reference_array: enpt.FloatArray['...'],
    num_reference_non_batch_dims: int,
) -> enpt.FloatArray['...']:
  """Reshapes an array by prepending the batch shape of a reference array.

  This function extracts the batch dimensions of `reference_array` (all but the
  last `num_reference_non_batch_dims` dimensions) and prepends them to the
  `target_suffix` shape. The input `array` is then reshaped to this combined
  shape.

  Args:
    array: The array to reshape.
    target_suffix: The shape suffix to append after the batch dimensions.
    reference_array: The array from which to extract the batch dimensions.
    num_reference_non_batch_dims: The number of trailing dimensions in
      `reference_array` that represent non-batch dimensions.

  Returns:
    The reshaped array with shape `(*batch_shape, *target_suffix)`.
  """
  xnp = enp.lazy.get_xnp(reference_array)
  shape = _graph_shape(reference_array)
  batch_shape = shape[:-num_reference_non_batch_dims]

  if enp.lazy.is_tf(reference_array):
    output_shape = enp.lazy.tf.concat(
        [batch_shape, list(target_suffix)], axis=0
    )
  else:
    output_shape = batch_shape + target_suffix
  return xnp.reshape(array, output_shape)


def zeros_with_batch_dims(
    reference_array: enpt.FloatArray['...'],
    num_reference_non_batch_dims: int,
    suffix_shape: tuple[int, ...],
    dtype: Any,
) -> enpt.FloatArray['...']:
  """Creates a zeros array with the batch shape of a reference array.

  This function extracts the batch dimensions of `reference_array` (all but the
  last `num_reference_non_batch_dims` dimensions) and creates an array of zeros
  with the combined shape `(*batch_shape, *suffix_shape)`.

  Args:
    reference_array: The array from which to extract the batch dimensions.
    num_reference_non_batch_dims: The number of trailing dimensions in
      `reference_array` that represent non-batch dimensions.
    suffix_shape: The shape suffix of the output array.
    dtype: The data type of the output array.

  Returns:
    An array of zeros of shape `(*batch_shape, *suffix_shape)` on the same
    device as `reference_array`.
  """
  xnp = enp.lazy.get_xnp(reference_array)

  shape = _graph_shape(reference_array)
  batch_shape = shape[:-num_reference_non_batch_dims]
  array_kwargs = dict(dtype=dtype)

  if enp.lazy.is_tf_xnp(xnp):
    full_shape = enp.lazy.tf.concat([batch_shape, list(suffix_shape)], axis=0)
  else:
    full_shape = batch_shape + suffix_shape
  if enp.lazy.is_torch(reference_array):
    array_kwargs['device'] = reference_array.device
  return xnp.zeros(full_shape, **array_kwargs)


def axis_angle_to_rotation_matrix(
    axis_angle: enpt.FloatArray['... 3'],
    epsilon: float = _EPSILON,
) -> enpt.FloatArray['... 3 3']:
  """Builds a 3x3 rotation matrix from an axis-angle vector.

  Uses Rodrigues' rotation formula:
    R = I + sin(theta) * K + (1 - cos(theta)) * K^2
  where K is the skew-symmetric cross-product matrix of the unit rotation axis.

  Args:
    axis_angle: The rotation vector where the direction represents the rotation
      axis and the magnitude represents the rotation angle in radians. Shape:
      `(..., 3)`.
    epsilon: A small value to avoid division by zero when the axis-angle vector
      has zero magnitude.

  Returns:
    A batch of 3x3 rotation matrices. Shape: `(..., 3, 3)`.
  """
  xnp = enp.get_np_module(axis_angle)

  norm_squared = xnp.sum(xnp.square(axis_angle), axis=-1, keepdims=True)
  angle = xnp.sqrt(
      xnp.maximum(norm_squared, xnp.asarray(epsilon, dtype=norm_squared.dtype))
  )
  axis = axis_angle / angle

  sin_angle, cos_angle = xnp.sin(angle), xnp.cos(angle)
  sin_angle, cos_angle = sin_angle[..., None], cos_angle[..., None]

  matrix = xnp.broadcast_to(
      eye(3, dtype=angle.dtype, reference_array=axis_angle, xnp=xnp),
      (*axis_angle.shape[:-1], 3, 3),
  )

  skew_01 = -axis[..., 2]
  skew_02 = axis[..., 1]
  skew_10 = axis[..., 2]
  skew_12 = -axis[..., 0]
  skew_20 = -axis[..., 1]
  skew_21 = axis[..., 0]
  row0 = xnp.stack([xnp.zeros_like(skew_01), skew_01, skew_02], axis=-1)
  row1 = xnp.stack([skew_10, xnp.zeros_like(skew_12), skew_12], axis=-1)
  row2 = xnp.stack([skew_20, skew_21, xnp.zeros_like(skew_21)], axis=-1)
  skew_symmetric_axis_matrix = xnp.stack([row0, row1, row2], axis=-2)

  matrix = (
      matrix
      + sin_angle * skew_symmetric_axis_matrix
      + (1.0 - cos_angle)
      * xnp.matmul(skew_symmetric_axis_matrix, skew_symmetric_axis_matrix)
  )

  return matrix


def joint_transforms_world(
    joints: enpt.FloatArray['... J 3'],
    rotations: enpt.FloatArray['... J 3'],
    translation: enpt.FloatArray['... 3'],
    joint_parent_indices: Sequence[int],
) -> enpt.FloatArray['... J 4 4']:
  """Computes the world-space transformation matrices for each joint.

  This function traverses the joint hierarchy and applies forward kinematics
  to compute the global/world space transforms of each joint, combining
  joint rotations and joint translations.

  Args:
    joints: The bind-pose joints positions. Shape: `(..., J, 3)`.
    rotations: The joint rotations as axis-angle vectors. Shape: `(..., J, 3)`.
    translation: The translation vector for the root joint. Shape: `(..., 3)`.
    joint_parent_indices: A sequence of parent indices for each joint. The root
      joint must have itself or a sentinel as parent (usually index 0 has parent
      0).

  Returns:
    A batch of 4x4 transformation matrices in world space for each joint.
    Shape: `(..., J, 4, 4)`.
  """
  xnp = enp.get_np_module(joints)
  num_joints = len(joint_parent_indices)

  # Gather the parent's joints position for each joint.
  # Under JAX/TF/NumPy, we can index with parents.
  # But we must exclude the root (index 0 has parent 0, but is not used).
  parent_indices_except_root = xnp.asarray(
      joint_parent_indices[1:], dtype=xnp.int32
  )
  joint_parents = take(joints, parent_indices_except_root, axis=-2, xnp=xnp)

  # Since we cannot do in-place assignments, we construct the local
  # transformation matrices by combining rotation matrices and translation
  # vectors.
  local_rotations = axis_angle_to_rotation_matrix(rotations)

  # Construct local translations.
  root_translation = joints[..., 0, :] + translation
  non_root_translations = joints[..., 1:, :] - joint_parents
  local_translations = xnp.concatenate(
      [root_translation[..., None, :], non_root_translations], axis=-2
  )

  # Combine rotation and translation.
  # local_transforms: (..., J, 3, 4)
  local_transforms_3x4 = xnp.concatenate(
      [local_rotations, local_translations[..., None]], axis=-1
  )

  # Append [0, 0, 0, 1] row.
  bottom_row = zeros_with_batch_dims(
      joints, 2, (num_joints, 1, 4), dtype=joints.dtype
  )
  # Set the last element to 1.0.
  bottom_row = bottom_row + xnp.asarray(
      [0.0, 0.0, 0.0, 1.0], dtype=joints.dtype
  )

  local_transforms = xnp.concatenate(
      [local_transforms_3x4, bottom_row], axis=-2
  )

  # Compute forward kinematics.
  # We cannot loop with direct array assignment in TF/JAX graph mode.
  # We must stack them sequentially.
  # FK: T_world[i] = T_world[parent[i]] @ T_local[i]
  # Since skeleton is a tree, we can build the list of transforms sequentially.
  world_transforms_list = [local_transforms[..., 0, :, :]]
  for joint_index in range(1, num_joints):
    parent_transform = world_transforms_list[joint_parent_indices[joint_index]]
    current_transform = (
        parent_transform @ local_transforms[..., joint_index, :, :]
    )
    world_transforms_list.append(current_transform)

  return xnp.stack(world_transforms_list, axis=-3)


def linear_blend_skinning(
    local_vertices: enpt.FloatArray['... V 3'],
    joint_positions: enpt.FloatArray['... J 3'],
    rotations: enpt.FloatArray['... J 3'],
    translation: enpt.FloatArray['... 3'],
    skinning_weights: enpt.FloatArray['J V'],
    joint_parent_indices: Sequence[int],
) -> enpt.FloatArray['... V 3']:
  """Poses the GNM vertices using Linear Blend Skinning (LBS).

  Linear Blend Skinning computes posed vertex positions by taking a weighted
  sum of the transformations applied to each joint, where the weights are
  defined per vertex and per joint.

  Args:
    local_vertices: The vertex positions in the local character space, with
      shape `(..., V, 3)`.
    joint_positions: The joint positions in bind pose, with shape `(..., J, 3)`.
    rotations: Joint rotations as axis-angle vectors, with shape `(..., J, 3)`.
    translation: Translation vector for the root joint, with shape: `(..., 3)`.
    skinning_weights: Skinning weights defining joint influence per vertex with
      shape `(J, V)`.
    joint_parent_indices: Sequence of parent indices for each joint.

  Returns:
    The posed skinned vertices with shape `(..., V, 3)`.
  """
  xnp = enp.get_np_module(local_vertices)
  num_joints = skinning_weights.shape[0]

  # The local-to-world transforms of each joint, after posing, (N, J, 4, 4).
  joint_transforms_world_local = joint_transforms_world(
      joint_positions, rotations, translation, joint_parent_indices
  )

  # deltas = T_world[..., :3, :3] @ joints
  deltas = xnp.einsum(
      '...jik,...jk->...ji',
      joint_transforms_world_local[..., :3, :3],
      joint_positions,
  )
  deltas = deltas[..., None]  # (..., J, 3, 1)

  offset_3x3 = zeros_with_batch_dims(
      local_vertices, 2, (num_joints, 3, 3), dtype=local_vertices.dtype
  )
  bottom_row = zeros_with_batch_dims(
      local_vertices, 2, (num_joints, 1, 4), dtype=local_vertices.dtype
  )

  offset = xnp.concatenate([offset_3x3, deltas], axis=-1)
  offset = xnp.concatenate([offset, bottom_row], axis=-2)
  joint_transforms = joint_transforms_world_local - offset

  # Convert the vertices to homogeneous coordinates.
  ones = (
      zeros_with_batch_dims(
          local_vertices,
          2,
          (local_vertices.shape[-2], 1),
          dtype=local_vertices.dtype,
      )
      + 1.0
  )
  vertices_h = xnp.concatenate([local_vertices, ones], axis=-1)

  # Perform Linear Blend Skinning.
  vertices_skinned = xnp.einsum(
      'jv,...jmn,...vn->...vm',
      skinning_weights,
      joint_transforms,
      vertices_h,
  )[..., :3]

  return vertices_skinned


def vertex_positions_bind_pose(
    identity: enpt.FloatArray['... I'] | None,
    expression: enpt.FloatArray['... E'] | None,
    template_vertex_positions: enpt.FloatArray['V 3'],
    vertex_identity_basis: enpt.FloatArray['I V 3'],
    expression_basis: enpt.FloatArray['E V 3'],
) -> enpt.FloatArray['... V 3']:
  """Computes vertices in the bind pose, with identity and expression applied.

  This function computes the neutral shape of the model by applying the identity
  (shape) blendshapes and expression (expression) blendshapes to the template
  vertices.

  Args:
    identity: The identity parameter vector. Shape: `(..., I)`.
    expression: The expression parameter vector. Shape: `(..., E)`.
    template_vertex_positions: The base template vertices. Shape: `(V, 3)`.
    vertex_identity_basis: The identity displacement basis. Shape: `(I, V, 3)`.
    expression_basis: The expression displacement basis. Shape: `(E, V, 3)`.

  Returns:
    The template vertices with shape and expression offsets applied. Shape:
    `(..., V, 3)`.
  """
  xnp = enp.get_np_module(template_vertex_positions)

  # Apply linear identity and expression bases.
  identity_deltas = 0.0
  if identity is not None:
    identity_deltas = xnp.einsum(
        '...i,ijk->...jk', identity, vertex_identity_basis
    )

  expression_deltas = 0.0
  if expression is not None:
    expression_deltas = xnp.einsum(
        '...i,ijk->...jk', expression, expression_basis
    )

  return template_vertex_positions + identity_deltas + expression_deltas


def joint_positions_bind_pose(
    identity: enpt.FloatArray['... I'] | None,
    template_joint_positions: enpt.FloatArray['J 3'],
    joint_identity_basis: enpt.FloatArray['I J 3'],
) -> enpt.FloatArray['... J 3']:
  """Joint positions in the bind pose, with identity basis applied.

  Computes joint locations in the bind pose by applying the identity shape
  regressor basis to the base template joint positions.

  Args:
    identity: The identity parameter vector. Shape: `(..., I)`.
    template_joint_positions: Base template joint positions. Shape: `(J, 3)`.
    joint_identity_basis: Identity regressor basis for joints. Shape: `(I, J,
      3)`.

  Returns:
    The template joints with shape offsets applied. Shape: `(..., J, 3)`.
  """
  xnp = enp.get_np_module(template_joint_positions)
  deltas = 0.0
  if identity is not None:
    deltas = xnp.einsum('...i,ijk->...jk', identity, joint_identity_basis)

  return template_joint_positions + deltas


def compute_pose_correctives(
    rotations: enpt.FloatArray['... J 3'] | None,
    pose_correctives_regressor: enpt.FloatArray['F F2'] | None,
    template_vertex_positions: enpt.FloatArray['V 3'],
    num_joints: int,
    num_vertices: int,
) -> enpt.FloatArray['... V 3']:
  """Applies pose-dependent corrective shape offsets to vertices.

  Pose correctives modify the mesh vertices based on joint rotation angles to
  fix skinning artifacts (e.g. at elbows, knees).

  Args:
    rotations: Joint rotations as axis-angle vectors, with shape `(..., J, 3)`.
    pose_correctives_regressor: The regressor mapping pose features to vertex
      offsets, with shape `(F, F2)`.
    template_vertex_positions: The base template vertices, used as reference for
      device and shape, with shape: `(V, 3)`.
    num_joints: The total number of joints.
    num_vertices: The total number of vertices.

  Returns:
    The corrective vertex offsets. Shape: `(..., V, 3)`.
  """
  xnp = enp.get_np_module(template_vertex_positions)

  if pose_correctives_regressor is None or rotations is None:
    return zeros_with_batch_dims(
        template_vertex_positions,
        2,
        (num_vertices, 3),
        dtype=template_vertex_positions.dtype,
    )

  rotation_matrices = axis_angle_to_rotation_matrix(rotations)

  # Construct eye matrices.
  eye_matrix = eye(3, dtype=rotations.dtype, reference_array=rotations, xnp=xnp)
  # Broadcast to matching shape.
  eye_reshaped = xnp.reshape(eye_matrix, (1,) * (rotations.ndim - 1) + (3, 3))
  pose_features = rotation_matrices - eye_reshaped

  pose_features = reshape_with_batch_dims(
      pose_features, (num_joints * 9,), rotations, 2
  )

  pose_deltas = xnp.einsum(
      '...f,fv->...v', pose_features, pose_correctives_regressor
  )
  return reshape_with_batch_dims(pose_deltas, (num_vertices, 3), rotations, 2)


def _graph_shape(array: enpt.FloatArray['...']) -> tuple[int, ...]:
  """Returns the shape of an array, supporting graph-mode arrays."""
  if enp.lazy.is_tf(array):
    return enp.lazy.tf.shape(array)
  else:
    return array.shape
