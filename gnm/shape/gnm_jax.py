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
  gnm = gnm_jax.from_local(version=GNMVersion.V3, variant=GNMVariant.HEAD)

  # Generate batches of parameters.
  n_batch = 5
  identity = np.random.uniform(shape=(n_batch, gnm.identity_dim))
  expression = np.random.uniform(shape=(n_batch, gnm.expression_dim))
  rotations = np.random.uniform(shape=[n_batch, gnm.num_joints, 3])
  translation = np.random.uniform(shape=(n_batch, 3))
  vertices = gnm(identity, expression, rotations, translation)
  ```

  # This module is differentiable.
  grad_func = jax.grad(
     lambda *args: jnp.square(gnm(*args)).mean(),
     argnums=np.Array([0, 1, 2, 3]))
  grads = grad_func(identity, expression, rotations, translation)
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

from absl import logging
from gnm.shape import gnm_landmarks
from gnm.shape import gnm_xnp
from gnm.shape.data.versions import gnm_specs
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

  def compute_vertex_normals(
      self,
      vertices: jt.Float[jnp.ndarray, "... V 3"],
  ) -> jt.Float[jnp.ndarray, "... V 3"]:
    """Compute vertex normals for GNM mesh."""
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
