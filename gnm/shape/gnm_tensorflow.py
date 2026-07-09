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

"""TensorFlow implementation of the GNM model.

Usage:
  ```
  gnm = gnm_tensorflow.from_local(
      version=GNMVersion.V3,
      variant=GNMVariant.HEAD
  )

  # Generate batches of parameters.
  n_batch = 5
  identity = tf.random.uniform(shape=(n_batch, gnm.identity_dim))
  expression = tf.random.uniform(shape=(n_batch, gnm.expression_dim))
  rotations = tf.random.uniform(shape=[n_batch, gnm.num_joints, 3])
  translation = tf.random.uniform(shape=(n_batch, 3))

  vertices = gnm(identity, expression, rotations, translation)
  ```
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

from absl import logging
from etils import enp
from gnm.shape import gnm_landmarks
from gnm.shape import gnm_xnp
from gnm.shape.data.versions import gnm_specs
import tensorflow as tf

GNMVersion = gnm_specs.GNMVersion
GNMMajorVersion = gnm_specs.GNMMajorVersion
GNMVariant = gnm_specs.GNMVariant
GNMBodyPart = gnm_specs.GNMBodyPart
GNMLandmarksType = gnm_landmarks.GNMLandmarksType


@dataclasses.dataclass(frozen=False, kw_only=True, init=False)
class GNM(gnm_xnp.GNM):
  """TensorFlow batched implementation of the GNM parametric model.

  GNM is a mesh-generating function. Given identity, expression, joint
  rotation, and translation parameters, it produces vertices of a mesh.

  This TensorFlow implementation evaluates a batch of N parameters, and produces
  a batch of vertex positions (N, V, 3).

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
    joint_parent_indices: Parent's index for each joint in the skeleton, (J).
    template_vertex_positions: Vertex positions in the template mesh, (V, 3).
    template_joint_positions: Joint positions in the template GNM, (J, 3).
    identity_names: The name of each identity in the identity basis.
    vertex_identity_basis: The vertex identity basis (I, V, 3).
    joint_identity_basis: The joint identity basis of the model, (I, J, 3).
    expression_names: The name of each expression in the expression basis.
    expression_basis: The vertex expression basis, aka blend-shapes, (E, V, 3).
    joint_names: The name of each joint in the skeleton.
    skinning_weights: The model's skinning weights, (J, V).
    quads: The mesh topology as quads, (Q, 4).
    triangles: The mesh topology as triangles, (T, 3).
    version: The version of the loaded GNM model.
    pose_correctives_regressor: Matrix for pose correctives, (9*J, 3*V).
    joint_regressor: Mapping from vertices to joints, (J, V).
    bone_aligned_orientations: The bone-aligned rotations for each joint, (J, 3,
      3). If they do not exist in the GNM npz, they are set to the identity
      matrix. Note that these are not used to compute the GNM joint and vertex
      positions.
    num_vertices: The number of vertices in the mesh V.
    num_joints: The number of joints in the skeleton J.
    identity_dim: The dimensionality of the linear identity basis I.
    expression_dim: The dimensionality of the linear expression basis E.
  """

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> GNM:
    """Creates a TensorFlow GNM instance from model data."""
    return cls._from_model_data_with_xnp(model_data, xnp=enp.lazy.tnp)

  def compute_vertex_normals(
      self,
      vertices: tf.Tensor,
  ) -> tf.Tensor:
    """Compute vertex normals for GNM mesh."""
    batch_dims = tf.shape(vertices)[:-2]
    num_vertices = tf.shape(vertices)[-2]
    vertices_flat = tf.reshape(vertices, (-1, num_vertices, 3))
    batch_size = tf.shape(vertices_flat)[0]

    face_vertices = tf.gather(vertices_flat, self.triangles, axis=1)
    v0 = face_vertices[..., 0, :]
    v1 = face_vertices[..., 1, :]
    v2 = face_vertices[..., 2, :]
    face_normals_area = tf.experimental.numpy.cross(v1 - v0, v2 - v0, axis=-1)

    data = tf.reshape(face_normals_area[:, :, None, :], (-1, 3))
    segment_ids = tf.reshape(self.triangles, (-1,))
    segment_ids = tf.tile(segment_ids[None, :], [batch_size, 1])

    batch_offset = (
        tf.range(batch_size, dtype=segment_ids.dtype)[:, None] * num_vertices
    )
    flat_segment_ids = segment_ids + batch_offset
    flat_normals = tf.math.unsorted_segment_sum(
        data, flat_segment_ids, num_segments=batch_size * num_vertices
    )
    vertex_normals = tf.reshape(flat_normals, (-1, num_vertices, 3))

    normal_magnitudes = tf.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    if tf.reduce_any(tf.experimental.numpy.isclose(normal_magnitudes, 0.0)):
      logging.warning(
          'Some vertex normals have zero magnitude. This is unexpected and'
          ' likely indicates triangle collapse.'
      )
    vertex_normals = vertex_normals / tf.maximum(normal_magnitudes, 1e-8)
    return tf.reshape(
        vertex_normals, tf.concat([batch_dims, [num_vertices, 3]], axis=0)
    )
