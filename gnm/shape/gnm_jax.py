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

"""JAX implementation of the GNM model.

Usage:
  ```
  gnm = gnm_jax.GNM.from_local(
      version=gnm_jax.GNMMajorVersion.V3, variant=gnm_jax.GNMVariant.HEAD
  )

  # Generate batches of parameters.
  n_batch = 5
  identity = np.random.uniform(size=(n_batch, gnm.identity_dim))
  expression = np.random.uniform(size=(n_batch, gnm.expression_dim))
  rotations = np.random.uniform(size=[n_batch, gnm.num_joints, 3])
  translation = np.random.uniform(size=(n_batch, 3))
  vertices = gnm(identity, expression, rotations, translation)
  ```

  # This module is differentiable.
  grad_func = jax.grad(
     lambda *args: jnp.square(gnm(*args)).mean(),
     argnums=(0, 1, 2, 3))
  grads = grad_func(identity, expression, rotations, translation)
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

from absl import logging
from gnm.shape import gnm_common
from gnm.shape import gnm_landmarks
from gnm.shape import gnm_xnp
from gnm.shape.data.versions import gnm_specs
import jax
import jax.numpy as jnp
import jaxtyping as jt

GNMVersion = gnm_specs.GNMVersion
GNMMajorVersion = gnm_specs.GNMMajorVersion
GNMVariant = gnm_specs.GNMVariant
GNMBodyPart = gnm_specs.GNMBodyPart
GNMLandmarksType = gnm_landmarks.GNMLandmarksType


@dataclasses.dataclass(frozen=False, kw_only=True, init=False)
class GNM(gnm_xnp.GNM):
  """JAX batched implementation of the GNM parametric model.

  GNM is a mesh-generating function. Given identity, expression, joint
  rotation, and translation parameters, it produces vertices of a mesh.

  This JAX implementation evaluates a batch of [A1, A2, ..., An] parameters, and
  produces a batch of vertex positions ([A1, A2, ..., An], V, 3).

  The GNM class also surfaces useful data for down-stream users, e.g. the
  names of each expression dimension, and the topology of the mesh.

  Shape dimensions are denoted:
  * N: Size of batch.
  * V: Number of vertices.
  * J: Number of joints.
  * I: Identity basis dimensionality.
  * E: Expression basis dimensionality.
  * Q: The number of quads in the mesh topology.
  * T: The number of triangles, in a triangulated version of the mesh topology.
  * G: Number of vertex groups.

  Attributes:
    version: The version of the loaded GNM model.
    variant: The variant of the loaded GNM model.
    template_vertex_positions: Vertex positions in the template mesh, (V, 3).
    template_joint_positions: Joint positions in the template GNM, (J, 3).
    vertex_identity_basis: The vertex identity basis of the model, (I, V, 3).
    joint_identity_basis: The joint identity basis of the model, (I, J, 3).
    expression_basis: The vertex expression basis, aka blend-shapes, (E, V, 3).
    identity_names: The name of each identity in the identity basis.
    joint_names: The name of each joint in the skeleton.
    expression_names: The name of each expression in the expression basis.
    joint_parent_indices: Parent's index for each joint in the skeleton, (J).
    skinning_weights: The model's skinning weights, (J, V).
    quads: The mesh topology as quads, (Q, 4).
    triangles: The mesh topology as triangles, (T, 3).
    quad_uvs: Texture coordinates per quad, (Q, 4, 2).
    triangle_uvs: Texture coordinates per triangle, (T, 3, 2).
    mesh_component_names: The vertex group name corresponding to each separate
      mesh part.
    mirror_indices: The index of each vertex on the other side of the mesh.
    joint_regressor: Mapping from vertices to joints, (J, V).
    pose_correctives_regressor: Matrix for pose correctives, (9*J, 3*V).
    bone_aligned_template_joint_orientations: The bone-aligned rotations for
      each joint, (J, 3, 3). If they do not exist in the GNM npz, they are set
      to the identity matrix. Note that these are not used to compute the GNM
      joint and vertex positions.
    vertex_groups: The weights in each vertex group, (G, V).
    vertex_group_names: The name of each vertex group, (G,).
    num_vertices: The number of vertices in the mesh V.
    num_joints: The number of joints in the skeleton J.
    identity_dim: The dimensionality of the linear identity basis I.
    expression_dim: The dimensionality of the linear expression basis E.
  """

  _shape_error_type = jt.TypeCheckError

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> GNM:
    """Creates a JAX GNM instance from model data."""
    return cls._from_model_data_with_xnp(model_data, xnp=jnp)

  def __call__(
      self,
      identity: jt.Float[jnp.ndarray, "A1 ... An I"] | None = None,
      expression: jt.Float[jnp.ndarray, "A1 ... An E"] | None = None,
      rotations: jt.Float[jnp.ndarray, "A1 ... An J 3"] | None = None,
      translation: jt.Float[jnp.ndarray, "A1 ... An 3"] | None = None,
  ) -> jt.Float[jnp.ndarray, "A1 ... An V 3"]:
    """Evaluates the GNM mesh-generating function."""
    with jax.default_matmul_precision("float32"):
      return super().__call__(
          identity=identity,
          expression=expression,
          rotations=rotations,
          translation=translation,
      )

  def vertex_positions_world(
      self,
      vertices: jt.Float[jnp.ndarray, "A1 ... An V 3"],
      joints: jt.Float[jnp.ndarray, "A1 ... An J 3"],
      rotations: jt.Float[jnp.ndarray, "A1 ... An J 3"],
      translation: jt.Float[jnp.ndarray, "A1 ... An 3"],
  ) -> jt.Float[jnp.ndarray, "A1 ... An V 3"]:
    """Applies linear blend skinning to GNM vertices."""
    with jax.default_matmul_precision("float32"):
      return super().vertex_positions_world(
          vertices=vertices,
          joints=joints,
          rotations=rotations,
          translation=translation,
      )

  apply_linear_blend_skinning = vertex_positions_world

  def vertex_positions_bind_pose(
      self,
      identity: jt.Float[jnp.ndarray, "A1 ... An I"] | None,
      expression: jt.Float[jnp.ndarray, "A1 ... An E"] | None,
  ) -> jt.Float[jnp.ndarray, "A1 ... An V 3"]:
    """Computes vertices in the bind pose, with identity and expression applied."""
    with jax.default_matmul_precision("float32"):
      return super().vertex_positions_bind_pose(
          identity=identity, expression=expression
      )

  def joint_positions_bind_pose(
      self,
      identity: jt.Float[jnp.ndarray, "A1 ... An I"] | None,
  ) -> jt.Float[jnp.ndarray, "A1 ... An J 3"]:
    """Joint positions in the bind pose, with identity basis applied."""
    with jax.default_matmul_precision("float32"):
      return super().joint_positions_bind_pose(identity=identity)

  def compute_pose_correctives(
      self,
      rotations: jt.Float[jnp.ndarray, "A1 ... An J 3"] | None,
  ) -> jt.Float[jnp.ndarray, "A1 ... An V 3"]:
    """Applies pose-dependent corrective shape offsets to vertices."""
    with jax.default_matmul_precision("float32"):
      return super().compute_pose_correctives(rotations=rotations)

  def joint_transforms_world(
      self,
      joints: jt.Float[jnp.ndarray, "A1 ... An J 3"],
      rotations: jt.Float[jnp.ndarray, "A1 ... An J 3"],
      translation: jt.Float[jnp.ndarray, "A1 ... An 3"],
  ) -> jt.Float[jnp.ndarray, "A1 ... An J 4 4"]:
    """Computes the world-space transformation matrices for each joint."""
    with jax.default_matmul_precision("float32"):
      return super().joint_transforms_world(
          joints=joints, rotations=rotations, translation=translation
      )

  def get_posed_joint_transforms(
      self,
      identity: jt.Float[jnp.ndarray, "A1 ... An I"],
      rotations: jt.Float[jnp.ndarray, "A1 ... An J 3"],
      translation: jt.Float[jnp.ndarray, "A1 ... An 3"],
  ) -> jt.Float[jnp.ndarray, "A1 ... An J 4 4"]:
    """Computes the local-to-world transformation for every joint."""
    with jax.default_matmul_precision("float32"):
      return super().get_posed_joint_transforms(
          identity=identity, rotations=rotations, translation=translation
      )

  def compute_vertex_normals(
      self,
      vertices: jt.Float[jnp.ndarray, "... V 3"],
  ) -> jt.Float[jnp.ndarray, "... V 3"]:
    """Compute vertex normals for GNM mesh.

    Args:
      vertices: Vertex positions with shape `(..., V, 3)`.

    Returns:
      Vertex normals with shape `(..., V, 3)`.
    """
    batch_dims = vertices.shape[:-2]
    num_vertices = vertices.shape[-2]
    vertices_flat = vertices.reshape(-1, num_vertices, 3)

    face_vertices = vertices_flat[:, self.triangles, :]
    v0 = face_vertices[..., 0, :]
    v1 = face_vertices[..., 1, :]
    v2 = face_vertices[..., 2, :]
    face_normals_area = jnp.cross(v1 - v0, v2 - v0, axis=-1)

    vertex_normals = jnp.zeros_like(vertices_flat)
    vertex_normals = vertex_normals.at[:, self.triangles, :].add(
        face_normals_area[:, :, None, :]
    )

    normal_magnitudes = jnp.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    if jnp.any(jnp.isclose(normal_magnitudes, 0.0)):
      logging.warning(
          "Some vertex normals have zero magnitude. This is unexpected and"
          " likely indicates triangle collapse."
      )
    vertex_normals = vertex_normals / jnp.maximum(normal_magnitudes, 1e-8)
    return vertex_normals.reshape(batch_dims + (num_vertices, 3))


def axis_angle_to_rotation_matrix(
    axis_angle: jt.Float[jnp.ndarray, "... 3"],
    epsilon: float = 1e-8,
) -> jt.Float[jnp.ndarray, "... 3 3"]:
  """Builds a 3x3 rotation matrix from an axis-angle vector."""
  with jax.default_matmul_precision("float32"):
    return gnm_common.axis_angle_to_rotation_matrix(axis_angle, epsilon=epsilon)

