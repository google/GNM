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

"""NumPy implementation of the GNM model .

Example usage:
  ```
  gnm = gnm_numpy.from_local(version=GNMVersion.V3, variant=GNMVariant.HEAD)

  # Generate random identity, expression, rotations, translation parameters
  identity = np.random.normal(size=gnm.identity_dim)
  expression = np.zeros(gnm.expression_dim)
  rotations = np.random.uniform(-1, 1, size=(gnm.num_joints, 3)) * 0.15
  translation = np.random.uniform(-1, 1, size=(3,)) * 0.15

  vertices = gnm(identity, expression, rotations, translation)
  ```
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
import numpy as np
import numpy.typing as npt

GNMVersion = gnm_specs.GNMVersion
GNMMajorVersion = gnm_specs.GNMMajorVersion
GNMVariant = gnm_specs.GNMVariant
GNMBodyPart = gnm_specs.GNMBodyPart
GNMLandmarksType = gnm_landmarks.GNMLandmarksType

_rotation_matrix = gnm_common.axis_angle_to_rotation_matrix


@dataclasses.dataclass(frozen=False, kw_only=True, init=False)
class GNM(gnm_xnp.GNM):
  """NumPy implementation of the GNM parametric model.

  GNM is a mesh-generating function. Given identity, expression, joint
  rotation, and translation parameters, it produces vertices of a mesh.

  The GNM class also surfaces useful data for down-stream users, e.g. the
  names of each expression dimension, and the topology of the mesh.

  Shape dimensions are denoted:
  * V: Number of vertices.
  * J: Number of joints.
  * I: Identity basis dimensionality.
  * E: Expression basis dimensionality.
  * Q: The number of quads in the mesh topology.
  * T: The number of triangles, in a triangulated version of the mesh topology.
  * G: Number of vertex groups.
  * P: Number of separate parts of the mesh.

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
    joint_connections: The joint connections of the GNM rig.
    joint_children_indices: A dictionary that contains the joint indices of the
      children of each joint.
    skinning_segmentation: A decomposition of the vertices into separate parts
      based on skinning weights.
    num_vertices: The number of vertices in the mesh V.
    num_joints: The number of joints in the skeleton J.
    identity_dim: The dimensionality of the linear identity basis I.
    expression_dim: The dimensionality of the linear expression basis E.
    edge_list: The quad topology represented as a list of directed edges (E, 2).
    vertex_uvs: Per-vertex UV texture coordinates shaped (V, 2).
  """

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> GNM:
    """Creates a GNM instance from model data."""
    return cls._from_model_data_with_xnp(model_data, xnp=np)

  def compute_vertex_normals(
      self,
      vertices: npt.NDArray[np.floating],
  ) -> npt.NDArray[np.floating]:
    """Compute vertex normals for GNM mesh."""
    batch_dims = vertices.shape[:-2]
    num_vertices = vertices.shape[-2]
    vertices_flat = vertices.reshape(-1, num_vertices, 3)

    face_vertices = vertices_flat[:, self.triangles, :]
    v0 = face_vertices[..., 0, :]
    v1 = face_vertices[..., 1, :]
    v2 = face_vertices[..., 2, :]
    face_normals_area = np.cross(v1 - v0, v2 - v0, axis=-1)

    vertex_normals = np.zeros_like(vertices_flat)
    np.add.at(
        vertex_normals,
        (slice(None), self.triangles, slice(None)),
        face_normals_area[:, :, None, :],
    )

    normal_magnitudes = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    if np.any(np.isclose(normal_magnitudes, 0.0)):
      logging.warning(
          'Some vertex normals have zero magnitude. This is unexpected and'
          ' likely indicates triangle collapse.'
      )
    vertex_normals = vertex_normals / np.maximum(normal_magnitudes, 1e-8)
    return vertex_normals.reshape(batch_dims + (num_vertices, 3))
