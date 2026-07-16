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

"""Unified backend-agnostic implementation of the GNM model using etils.enp."""

from __future__ import annotations

import abc
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
from typing import Any, Self

from absl import logging
from etils import enp
from gnm.shape import gnm_base
from gnm.shape import gnm_common
from gnm.shape import gnm_landmarks
from gnm.shape.data.versions import gnm_specs
import numpy as np
import numpy.typing as npt

enpt = enp.typing

_NONZERO_THRESHOLD = 1e-4
_EPSILON = 1e-8


@dataclasses.dataclass(frozen=False, kw_only=True, init=False)
class GNM(gnm_base.GNMBase):
  """Backend-agnostic implementation of the GNM parametric model.

  This class is abstract. Use a concrete backend subclass (gnm_numpy.GNM,
  gnm_pytorch.GNM, or gnm_jax.GNM).

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
    cl_number: The CL used to create this model.
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

  _shape_error_type = ValueError

  version: gnm_specs.GNMVersion
  variant: gnm_specs.GNMVariant
  template_vertex_positions: enpt.FloatArray
  template_joint_positions: enpt.FloatArray
  vertex_identity_basis: enpt.FloatArray
  joint_identity_basis: enpt.FloatArray
  expression_basis: enpt.FloatArray
  identity_names: Sequence[str]
  joint_names: Sequence[str]
  expression_names: Sequence[str]
  joint_parent_indices: Sequence[int]
  skinning_weights: enpt.FloatArray
  quads: enpt.IntArray
  triangles: enpt.IntArray
  quad_uvs: enpt.FloatArray
  triangle_uvs: enpt.FloatArray
  mesh_component_names: Sequence[str]
  mirror_indices: enpt.IntArray
  joint_regressor: enpt.FloatArray
  pose_correctives_regressor: enpt.FloatArray
  bone_aligned_template_joint_orientations: enpt.FloatArray
  vertex_groups: enpt.FloatArray
  vertex_group_names: Sequence[str]

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise TypeError(
        f'{self.__class__.__name__} cannot be instantiated directly via'
        ' constructor. Please use a class factory method such as from_local()'
        ' or from_model_data().'
    )

  def __post_init__(self):
    self._vertex_group_names_lookup: dict[str, int] = {
        name: i for i, name in enumerate(self.vertex_group_names)
    }
    self._xnp = enp.get_np_module(self.template_vertex_positions)
    self._landmarks: dict[
        gnm_landmarks.GNMLandmarksType, tuple[enpt.IntArray, enpt.FloatArray]
    ] = {}

  @property
  def xnp(self) -> enp.NpModule:
    """Returns the backend module dynamically from G-Nome array state."""
    return self._xnp

  def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('_xnp', None)
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._xnp = enp.get_np_module(self.template_vertex_positions)

  @classmethod
  def _prepare_init_kwargs(
      cls,
      model_data: Mapping[str, Any],
      xnp: enp.NpModule,
  ) -> dict[str, Any]:
    """Prepares and casts GNM initialization arguments using xnp."""
    init_kwargs = {}

    # Identify type convert functions.
    def as_float_array(val):
      return xnp.asarray(val, dtype=xnp.float32) if val is not None else None

    def as_int_array(val):
      return xnp.asarray(val, dtype=xnp.int32) if val is not None else None

    def as_original(val):
      return val

    field_converters = {
        'version': as_original,
        'variant': as_original,
        'cl_number': as_original,
        'template_vertex_positions': as_float_array,
        'template_joint_positions': as_float_array,
        'vertex_identity_basis': as_float_array,
        'joint_identity_basis': as_float_array,
        'expression_basis': as_float_array,
        'identity_names': as_original,
        'joint_names': as_original,
        'expression_names': as_original,
        'joint_parent_indices': as_original,
        'skinning_weights': as_float_array,
        'quads': as_int_array,
        'triangles': as_int_array,
        'quad_uvs': as_float_array,
        'triangle_uvs': as_float_array,
        'mesh_component_names': as_original,
        'mirror_indices': as_int_array,
        'joint_regressor': as_float_array,
        'pose_correctives_regressor': as_float_array,
        'bone_aligned_template_joint_orientations': as_float_array,
        'vertex_groups': as_float_array,
        'vertex_group_names': as_original,
    }

    for field_name, converter in field_converters.items():
      val = model_data.get(field_name, None)
      init_kwargs[field_name] = converter(val)

    return init_kwargs

  @classmethod
  def _from_model_data_with_xnp(
      cls,
      model_data: Mapping[str, Any],
      xnp: enp.NpModule,
  ) -> Self:
    """Creates a GNM instance from a model data dictionary and array module."""    
    init_kwargs = cls._prepare_init_kwargs(model_data, xnp)
    # pylint: disable=no-value-for-parameter
    instance = super(GNM, cls).__new__(cls)
    # pylint: enable=no-value-for-parameter
    for k, v in init_kwargs.items():
      object.__setattr__(instance, k, v)
    instance.__post_init__()
    return instance

  @classmethod
  @abc.abstractmethod
  def _from_model_data(
      cls,
      data_dict: Mapping[str, Any],
  ) -> Self:
    """Creates a GNM instance from a model data dictionary."""
    raise NotImplementedError(
        f'{cls.__name__} is an abstract backend-agnostic base class and'
        ' cannot be loaded directly. Use a concrete backend subclass (e.g.,'
        ' gnm_numpy.GNM, gnm_pytorch.GNM).'
    )

  def to_numpy_data_dict(self) -> dict[str, Any]:
    """Returns a dictionary of the GNM data represented as NumPy arrays."""
    data_dict = {}
    for field in dataclasses.fields(self):
      val = getattr(self, field.name)
      if enp.lazy.is_array(val):
        # Convert JAX, PyTorch or TF tensors to numpy.
        if enp.lazy.is_tf(val):
          val = val.numpy()
        elif enp.lazy.is_jax(val):
          val = np.array(val)
        elif enp.lazy.is_torch(val):
          val = val.detach().cpu().numpy()
      data_dict[field.name] = val
    return data_dict

  def _check_parameter_shapes(
      self,
      identity: enpt.FloatArray['...'] | None = None,
      expression: enpt.FloatArray['...'] | None = None,
      rotations: enpt.FloatArray['...'] | None = None,
      translation: enpt.FloatArray['...'] | None = None,
  ) -> None:
    error_type = self._shape_error_type
    if identity is not None:
      if identity.ndim < 1 or identity.shape[-1] != self.identity_dim:
        raise error_type(
            f'identity shape mismatch: expected last dim {self.identity_dim},'
            f' got {identity.shape}'
        )
    if expression is not None:
      if expression.ndim < 1 or expression.shape[-1] != self.expression_dim:
        raise error_type(
            'expression shape mismatch: expected last dim'
            f' {self.expression_dim}, got {expression.shape}'
        )
    if rotations is not None:
      if (
          rotations.ndim < 2
          or rotations.shape[-2] != self.num_joints
          or rotations.shape[-1] != 3
      ):
        raise error_type(
            'rotations shape mismatch: expected last dims'
            f' {(self.num_joints, 3)}, got {rotations.shape}'
        )
    if translation is not None:
      if translation.ndim < 1 or translation.shape[-1] != 3:
        raise error_type(
            'translation shape mismatch: expected last dim 3, got'
            f' {translation.shape}'
        )

  @enp.check_and_normalize_arrays(strict=False)
  def __call__(
      self,
      identity: enpt.FloatArray['A1 ... An I'] | None = None,
      expression: enpt.FloatArray['A1 ... An E'] | None = None,
      rotations: enpt.FloatArray['A1 ... An J 3'] | None = None,
      translation: enpt.FloatArray['A1 ... An 3'] | None = None,
  ) -> enpt.FloatArray['A1 ... An V 3']:
    """Evaluates the GNM mesh-generating function."""
    self._check_parameter_shapes(identity, expression, rotations, translation)
    batch_shape = _check_batch_dims(
        identity, expression, rotations, translation
    )

    # Fill in missing parameters.
    if identity is None:
      identity = gnm_common.zeros_with_batch_dims(
          self.template_vertex_positions,
          2,
          (*batch_shape, self.identity_dim),
          dtype=self.template_vertex_positions.dtype,
      )
    if expression is None:
      expression = gnm_common.zeros_with_batch_dims(
          self.template_vertex_positions,
          2,
          (*batch_shape, self.expression_dim),
          dtype=self.template_vertex_positions.dtype,
      )
    if rotations is None:
      rotations = gnm_common.zeros_with_batch_dims(
          self.template_vertex_positions,
          2,
          (*batch_shape, self.num_joints, 3),
          dtype=self.template_vertex_positions.dtype,
      )
    if translation is None:
      translation = gnm_common.zeros_with_batch_dims(
          self.template_vertex_positions,
          2,
          (*batch_shape, 3),
          dtype=self.template_vertex_positions.dtype,
      )

    # Bind pose vertex positions.
    vertices = self.vertex_positions_bind_pose(identity, expression)

    # Bind pose joint positions.
    joints = self.joint_positions_bind_pose(identity)

    # Pose correctives.
    pose_correctives = self.compute_pose_correctives(rotations)
    vertices = vertices + pose_correctives

    return self.vertex_positions_world(vertices, joints, rotations, translation)

  @enp.check_and_normalize_arrays(strict=False)
  def vertices_and_landmarks(
      self,
      landmarks_type: gnm_landmarks.GNMLandmarksType,
      identity: enpt.FloatArray['A1 ... An I'] | None = None,
      expression: enpt.FloatArray['A1 ... An E'] | None = None,
      rotations: enpt.FloatArray['A1 ... An J 3'] | None = None,
      translation: enpt.FloatArray['A1 ... An 3'] | None = None,
  ) -> tuple[
      enpt.FloatArray['A1 ... An V 3'],
      enpt.FloatArray['A1 ... An L 3'],
  ]:
    """Evaluates the GNM mesh function and extracts 3D landmarks."""
    gnm_landmarks.check_body_part_compatibility(landmarks_type, self.body_part)
    if landmarks_type not in self._landmarks:
      config = gnm_landmarks.load_landmarks(landmarks_type)
      self._landmarks[landmarks_type] = (
          self.xnp.asarray(config.indices, dtype=self.xnp.int32),
          self.xnp.asarray(config.weights, dtype=self.xnp.float32),
      )
    indices, weights = self._landmarks[landmarks_type]

    vertices = self(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
    weights = enp.compat.astype(weights, vertices.dtype)

    face_vertices = gnm_common.take(vertices, indices, axis=-2, xnp=self.xnp)
    landmarks = self.xnp.sum(face_vertices * weights[..., None], axis=-2)
    return vertices, landmarks

  def vertex_positions_world(
      self,
      vertices: enpt.FloatArray['A1 ... An V 3'],
      joints: enpt.FloatArray['A1 ... An J 3'],
      rotations: enpt.FloatArray['A1 ... An J 3'],
      translation: enpt.FloatArray['A1 ... An 3'],
  ) -> enpt.FloatArray['A1 ... An V 3']:
    """Applies linear blend skinning to GNM vertices."""
    return gnm_common.linear_blend_skinning(
        local_vertices=vertices,
        joint_positions=joints,
        rotations=rotations,
        translation=translation,
        skinning_weights=self.skinning_weights,
        joint_parent_indices=self.joint_parent_indices,
    )

  apply_linear_blend_skinning = vertex_positions_world

  @property
  def num_vertices(self) -> int:
    return self.template_vertex_positions.shape[0]

  @property
  def num_joints(self) -> int:
    return self.skinning_weights.shape[0]

  @property
  def identity_dim(self) -> int:
    return self.vertex_identity_basis.shape[0]

  @property
  def expression_dim(self) -> int:
    return self.expression_basis.shape[0]

  @functools.cached_property
  def edge_list(self) -> npt.NDArray[np.integer]:
    """The quad topology represented as a list of directed edges."""
    quads = np.array(self.quads)
    e1 = quads.ravel()
    e2 = np.roll(quads, -1, axis=1).ravel()
    edges = np.stack([np.minimum(e1, e2), np.maximum(e1, e2)], axis=1)
    edge_keys = e1 + self.num_vertices * e2
    _, unique_indices = np.unique(edge_keys, return_index=True)
    unique_undirected_edges = edges[unique_indices]
    return np.vstack(
        [unique_undirected_edges, np.fliplr(unique_undirected_edges)]
    )

  @functools.cached_property
  def joint_connections(self) -> npt.NDArray[np.int32]:
    """The bones of the GNM rig."""
    return np.stack(
        [
            np.array(self.joint_parent_indices),
            np.arange(self.num_joints, dtype=np.int32),
        ],
        axis=1,
    )[1:]

  @functools.cached_property
  def joint_children_indices(self) -> Mapping[int, Sequence[int]]:
    children = collections.defaultdict(list)
    for joint, parent in enumerate(self.joint_parent_indices):
      if joint == 0:
        continue
      children[parent].append(joint)
    return dict(children)

  @functools.cached_property
  def vertex_uvs(self) -> npt.NDArray[np.floating]:
    """Per-vertex UV texture coordinates."""
    logging.warning(
        'These do not allow correct texturing across UV-seams. Please use '
        'per-quad or per-triangle UV coordinates when possible.'
    )
    quads = np.array(self.quads)
    quad_uvs = np.array(self.quad_uvs)
    indices = quads.ravel()
    uvs = quad_uvs.reshape(-1, 2)
    _, last_indices_reversed = np.unique(indices[::-1], return_index=True)
    last_indices = len(indices) - 1 - last_indices_reversed
    vertex_uvs = np.zeros((self.num_vertices, 2), dtype=uvs.dtype)
    vertex_uvs[indices[last_indices]] = uvs[last_indices]
    return vertex_uvs

  def add_vertex_group(self, name: str, value: enpt.FloatArray['V']) -> None:
    """Adds a new vertex group.

    Args:
      name: The name of the vertex group.
      value: The vertex group values, shape (V,).
    """
    xnp = self.xnp
    if name in self.vertex_group_names:
      raise ValueError(f'Vertex group {name} already exists.')
    if value.ndim != 1 or value.shape[0] != self.num_vertices:
      raise ValueError(
          f'Vertex group must be 1D array of length {self.num_vertices}.'
      )

    self.vertex_groups = xnp.concatenate(
        [self.vertex_groups, value[None]], axis=0
    )
    self.vertex_group_names = list(self.vertex_group_names) + [name]
    self._vertex_group_names_lookup[name] = len(self.vertex_group_names) - 1

  @property
  def skinning_segmentation(self) -> npt.NDArray[np.int32]:
    return np.array(self.skinning_weights).argmax(axis=0)

  @enp.check_and_normalize_arrays(strict=False)
  def vertex_positions_bind_pose(
      self,
      identity: enpt.FloatArray['A1 ... An I'],
      expression: enpt.FloatArray['A1 ... An E'],
  ) -> enpt.FloatArray['A1 ... An V 3']:
    return gnm_common.vertex_positions_bind_pose(
        identity,
        expression,
        self.template_vertex_positions,
        self.vertex_identity_basis,
        self.expression_basis,
    )

  @enp.check_and_normalize_arrays(strict=False)
  def joint_positions_bind_pose(
      self, identity: enpt.FloatArray['A1 ... An I']
  ) -> enpt.FloatArray['A1 ... An J 3']:
    return gnm_common.joint_positions_bind_pose(
        identity,
        self.template_joint_positions,
        self.joint_identity_basis,
    )

  @enp.check_and_normalize_arrays(strict=False)
  def compute_pose_correctives(
      self, rotations: enpt.FloatArray['A1 ... An J 3']
  ) -> enpt.FloatArray['A1 ... An V 3']:
    return gnm_common.compute_pose_correctives(
        rotations,
        self.pose_correctives_regressor,
        self.template_vertex_positions,
        self.num_joints,
        self.num_vertices,
    )

  @enp.check_and_normalize_arrays(strict=False)
  def joint_transforms_world(
      self,
      joints: enpt.FloatArray['A1 ... An J 3'],
      rotations: enpt.FloatArray['A1 ... An J 3'],
      translation: enpt.FloatArray['A1 ... An 3'],
  ) -> enpt.FloatArray['A1 ... An J 4 4']:
    return gnm_common.joint_transforms_world(
        joints, rotations, translation, self.joint_parent_indices
    )

  @enp.check_and_normalize_arrays(strict=False)
  def get_posed_joint_transforms(
      self,
      identity: enpt.FloatArray['A1 ... An I'],
      rotations: enpt.FloatArray['A1 ... An J 3'],
      translation: enpt.FloatArray['A1 ... An 3'],
  ) -> enpt.FloatArray['A1 ... An J 4 4']:
    self._check_parameter_shapes(
        identity=identity, rotations=rotations, translation=translation
    )
    _check_batch_dims(
        identity=identity, rotations=rotations, translation=translation
    )
    return self.joint_transforms_world(
        joints=self.joint_positions_bind_pose(identity),
        rotations=rotations,
        translation=translation,
    )

  def vertex_group(self, name: str) -> npt.NDArray[np.floating]:
    try:
      return np.array(self.vertex_groups[self._vertex_group_names_lookup[name]])
    except KeyError as exc:
      raise KeyError(
          f'Vertex group {name} not found in {self.vertex_group_names}.'
      ) from exc

  def vertex_group_mask(
      self, *names: str, threshold: float = _NONZERO_THRESHOLD
  ) -> npt.NDArray[bool]:
    result_mask = np.zeros(self.num_vertices, dtype=bool)
    for name in names:
      operator, inverse = '|', False
      if name[0] in '|&-':
        operator, name = name[0], name[1:]
      if name[0] == '~':
        inverse, name = True, name[1:]
      group_mask = self.vertex_group(name) > threshold
      if inverse:
        group_mask = ~group_mask
      match operator:
        case '|':
          result_mask |= group_mask
        case '&':
          result_mask &= group_mask
        case '-':
          result_mask &= ~group_mask
    return result_mask

  def vertex_group_indices(
      self, *names: str, threshold: float = _NONZERO_THRESHOLD
  ) -> npt.NDArray[np.integer]:
    return np.where(self.vertex_group_mask(*names, threshold=threshold))[0]

  def quad_indices_for_group(self, *names: str) -> npt.NDArray[np.integer]:
    vertex_indices = self.vertex_group_indices(*names)
    quads = np.array(self.quads)
    return np.where(np.all(np.isin(quads, vertex_indices), axis=-1))[0]

  def triangle_indices_for_group(self, *names: str) -> npt.NDArray[np.integer]:
    vertex_indices = self.vertex_group_indices(*names)
    triangles = np.array(self.triangles)
    return np.where(np.all(np.isin(triangles, vertex_indices), axis=-1))[0]

  def quads_group(self, *names: str) -> npt.NDArray[np.integer]:
    return np.array(self.quads)[self.quad_indices_for_group(*names)]

  def triangles_group(self, *names: str) -> npt.NDArray[np.integer]:
    return np.array(self.triangles)[self.triangle_indices_for_group(*names)]

  def quad_uvs_group(self, *names: str) -> npt.NDArray[np.floating]:
    return np.array(self.quad_uvs)[self.quad_indices_for_group(*names)]

  def triangle_uvs_group(self, *names: str) -> npt.NDArray[np.floating]:
    return np.array(self.triangle_uvs)[self.triangle_indices_for_group(*names)]

  def vertex_uvs_group(self, *names: str) -> npt.NDArray[np.floating]:
    return self.vertex_uvs[self.vertex_group_indices(*names)]

  def compute_vertex_normals(
      self, vertices: enpt.FloatArray['A1 ... An V 3']
  ) -> enpt.FloatArray['A1 ... An V 3']:
    """Computes vertex normals. Must be overridden by subclasses."""
    raise NotImplementedError(
        'Subclasses must implement compute_vertex_normals.'
    )

  def prune_vertices(self, keep_vertices: enpt.IntArray['V_pruned']) -> None:
    """Prunes model vertices in-place."""
    xnp = self.xnp
    num_vertices = self.num_vertices
    keep_vertices = xnp.asarray(keep_vertices, dtype=xnp.int32)

    self.template_vertex_positions = gnm_common.take(
        self.template_vertex_positions, keep_vertices, axis=0, xnp=xnp
    )
    self.vertex_identity_basis = gnm_common.take(
        self.vertex_identity_basis, keep_vertices, axis=1, xnp=xnp
    )
    self.expression_basis = gnm_common.take(
        self.expression_basis, keep_vertices, axis=1, xnp=xnp
    )
    self.skinning_weights = gnm_common.take(
        self.skinning_weights, keep_vertices, axis=1, xnp=xnp
    )

    mapper = _scatter_indices(keep_vertices, num_vertices, xnp)

    quads = gnm_common.take(mapper, self.quads, xnp=xnp)
    triangles = gnm_common.take(mapper, self.triangles, xnp=xnp)

    # Remove quads/triangles containing negative indices.
    quad_mask = xnp.all(quads >= 0, axis=-1)
    triangle_mask = xnp.all(triangles >= 0, axis=-1)

    quad_indices = xnp.where(quad_mask)[0]
    triangle_indices = xnp.where(triangle_mask)[0]

    self.quads = gnm_common.take(quads, quad_indices, axis=0, xnp=xnp)
    self.triangles = gnm_common.take(
        triangles, triangle_indices, axis=0, xnp=xnp
    )

    if self.pose_correctives_regressor is not None:
      pose_correctives = xnp.reshape(
          self.pose_correctives_regressor,
          (self.num_joints * 9, num_vertices, 3),
      )
      pose_correctives = gnm_common.take(
          pose_correctives, keep_vertices, axis=1, xnp=xnp
      )
      self.pose_correctives_regressor = xnp.reshape(
          pose_correctives, (-1, keep_vertices.shape[0] * 3)
      )


def _check_batch_dims(
    identity: enpt.FloatArray['... I'] | None = None,
    expression: enpt.FloatArray['... E'] | None = None,
    rotations: enpt.FloatArray['... J 3'] | None = None,
    translation: enpt.FloatArray['... 3'] | None = None,
) -> tuple[int, ...]:
  """Ensures that the leading batch dimensions of all inputs are the same."""
  shapes = {}
  if identity is not None:
    shapes['identity'] = identity.shape[:-1]
  if expression is not None:
    shapes['expression'] = expression.shape[:-1]
  if rotations is not None:
    shapes['rotations'] = rotations.shape[:-2]
  if translation is not None:
    shapes['translation'] = translation.shape[:-1]

  if not shapes:
    return ()

  first_name, first_shape = next(iter(shapes.items()))
  for name, shape in shapes.items():
    if shape != first_shape:
      raise ValueError(
          f'Mismatched batch dimensions: {first_name} has {first_shape}, '
          f'but {name} has {shape}.'
      )
  return first_shape


def _scatter_indices(keep_vertices, num_vertices, xnp) -> Any:
  """Computes a mapping array to translate pruned vertex indices."""
  if enp.lazy.is_tf_xnp(xnp):
    tf = enp.lazy.tf
    return (
        tf.scatter_nd(
            indices=tf.expand_dims(keep_vertices, axis=-1),
            updates=tf.range(tf.shape(keep_vertices)[0], dtype=tf.int32) + 1,
            shape=(num_vertices,),
        )
        - 1
    )
  elif enp.lazy.is_jax_xnp(xnp):
    # pytype: disable=import-error
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top,import-outside-toplevel
    # pytype: enable=import-error
    mapper = jnp.full((num_vertices,), -1, dtype=jnp.int32)
    return mapper.at[keep_vertices].set(
        jnp.arange(len(keep_vertices), dtype=jnp.int32)
    )
  elif enp.lazy.is_torch_xnp(xnp):
    mapper = xnp.full((num_vertices,), -1, dtype=xnp.int32)
    mapper[keep_vertices] = xnp.arange(len(keep_vertices), dtype=xnp.int32)
    return mapper
  else:
    mapper = np.full((num_vertices,), -1, dtype=np.int32)
    mapper[keep_vertices] = np.arange(len(keep_vertices), dtype=np.int32)
    return mapper
