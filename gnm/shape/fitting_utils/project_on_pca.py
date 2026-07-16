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

"""Projection on a linear vertex basis."""

import dataclasses

from gnm.shape.fitting_utils import regularized_least_squares
import numpy as np

_EPSILON = 1e-16


@dataclasses.dataclass(frozen=True)
class PCAProjectionResult:
  """The result of projecting a mesh on a linear vertex basis.

  Attributes:
    coefficients: The coefficients of the vertex basis. Note that if
      `num_components` is given, then the coefficients will be truncated to that
      number of components.
    reconstruction: The reconstructed vertices.
  """

  coefficients: np.ndarray
  reconstruction: np.ndarray | None = None


def project_on_linear_vertex_basis(
    vertices: np.ndarray,
    mean_vertex_positions: np.ndarray,
    vertex_basis: np.ndarray,
    indices: np.ndarray | None = None,
    regularization: float = 0.0,
    num_components: int | None = None,
    compute_reconstruction: bool = True,
) -> PCAProjectionResult:
  """Finds the coefficients of the vertex basis that best fit the vertices.

  Solves a least-squares problem to find the coefficients of the vertex basis
  that best fit the vertices. The formulation is:

  || mu + B @ c - v ||_2^2 + lambda * ||c||_2^2
  || B @ c - (v - mu) ||_2^2 + lambda * ||c||_2^2

  where mu contains the mean vertex positions, (V, 3), B is the vertex basis we
  are projecting on, (I, V, 3), c contains the desired coefficients, (I,), and v
  are the target vertices, (V, 3).

  Args:
    vertices: The vertices to project, ([A1, ..., An], V, 3).
    mean_vertex_positions: The mean vertex positions, (V, 3).
    vertex_basis: The vertex basis, (I, V, 3).
    indices: The indices of the vertices to project, (V,). If None, all vertices
      are used to compute the least-squares error.
    regularization: The regularization weight for the least-squares problem.
    num_components: The number of components to use for the projection. If None,
      all components are used.
    compute_reconstruction: If True, compute the reconstructed vertices using
      the estimated coefficients and given vertex basis.

  Returns:
    An instance of `PCAProjectionResult` containing the coefficients and the
    reconstructed vertices. Note that the coefficients and reconstructions in
    `PCAProjectionResult` are *always* batched.
  """
  if indices is None:
    indices = np.arange(len(mean_vertex_positions))

  # If the number of components is not given, use all components.
  if num_components is None:
    num_components = vertex_basis.shape[0]

  # Add a batch dimension if the vertices are not batched.
  if len(vertices.shape) == 2:
    vertices = np.expand_dims(vertices, axis=0)

  # Compute the offset from the mean vertex positions, ([A1, ..., An], V, 3]).
  offsets_from_mean = vertices - mean_vertex_positions
  offsets_from_mean = np.take(offsets_from_mean, indices, axis=-2)

  # Flatten the shape offsets to (prod(A1, ..., An), V * 3).
  num_batch_elements = np.prod(offsets_from_mean.shape[:-2])
  offsets_from_mean = np.reshape(
      offsets_from_mean, [num_batch_elements, len(indices) * 3]
  )

  # Transpose the shape offsets to (V * 3, prod(A1, ..., An)).
  offsets_from_mean = np.swapaxes(offsets_from_mean, 0, 1)

  # Index the basis using the selected vertex indices and the desired number of
  # components, flatten the vertex and coordinate axis, and transpose to
  # (V*3, I).
  vertex_basis_subset = vertex_basis[:num_components]
  basis_for_fitting = np.reshape(
      vertex_basis_subset[:, indices], [num_components, len(indices) * 3]
  )
  basis_for_fitting = np.swapaxes(basis_for_fitting, 0, 1)

  coefficients, *_ = regularized_least_squares.regularized_least_squares(
      basis_for_fitting, offsets_from_mean, regularization
  )

  # Reshape the coefficients to ([A1, ..., An], I).
  coefficients = np.swapaxes(coefficients, 0, 1)
  coefficients = np.reshape(
      coefficients, [*vertices.shape[:-2], num_components]
  )

  reconstruction = None
  if compute_reconstruction:
    reconstruction = mean_vertex_positions + np.einsum(
        '...i,ivm->...vm', coefficients, vertex_basis_subset
    )

  return PCAProjectionResult(
      coefficients=coefficients, reconstruction=reconstruction
  )


class PCABasisProjection:
  """Projects vertices on a linear vertex basis.

  This class pre-computes the matrix used to project vertices on a linear vertex
  basis for repeated solves.
  """

  def __init__(
      self,
      mean_vertex_positions: np.ndarray,
      vertex_basis: np.ndarray,
      vertex_indices: np.ndarray | None = None,
      regularization: float = 0.0,
      num_components: int | None = None,
      compute_reconstruction: bool = True,
  ):
    """Builds the instance and pre-computes the projection matrix.

    Args:
      mean_vertex_positions: The mean vertex positions for the basis, (V, 3).
      vertex_basis: The vertex basis, (I, V, 3).
      vertex_indices: The indices of the vertices to project, (V,). If None, all
        vertices are used to compute the least-squares error.
      regularization: The regularization weight for the least-squares problem.
      num_components: The number of components to use for the projection. If
        None, all components are used.
      compute_reconstruction: If True, compute the reconstructed vertices using
        the estimated coefficients and given vertex basis.
    """

    self._compute_reconstruction = compute_reconstruction

    if vertex_indices is None:
      vertex_indices = np.arange(len(mean_vertex_positions))

    # If the number of components is not given, use all components.
    if num_components is None:
      num_components = vertex_basis.shape[0]

    self._num_components = num_components
    self._vertex_indices = vertex_indices
    self._mean_vertex_positions = mean_vertex_positions.astype(np.float32)

    regularization_sqrt = np.sqrt(regularization + _EPSILON)
    self._regularization_sqrt = regularization_sqrt

    # Index the basis using the selected vertex indices and the desired number
    # of components, flatten the vertex and coordinate axis, and transpose to
    # (V*3, I).
    vertex_basis_subset = vertex_basis[:num_components]
    self._vertex_basis_subset = vertex_basis_subset

    basis_for_fitting = np.reshape(
        vertex_basis_subset[:, vertex_indices],
        [num_components, len(vertex_indices) * 3],
    )
    basis_for_fitting = np.swapaxes(basis_for_fitting, 0, 1)

    # Append a diagonal array multiplied with the regularization weight.
    left_hand_side_regularization_term = regularization_sqrt * np.eye(
        basis_for_fitting.shape[0],
        basis_for_fitting.shape[1],
        dtype=basis_for_fitting.dtype,
    )
    left_hand_side = np.concatenate(
        [basis_for_fitting, left_hand_side_regularization_term]
    ).astype(np.float32)

    # Contains the array used to project the vertices on the basis.
    self._basis_projection_matrix = np.linalg.pinv(left_hand_side).astype(
        np.float32
    )

  def __call__(
      self,
      vertices: np.ndarray,
  ) -> PCAProjectionResult:
    """Finds the coefficients of the vertex basis that best fit the vertices.

    Args:
      vertices: The vertices to project, ([A1, ..., An], V, 3).

    Returns:
      An instance of `PCAProjectionResult` containing the coefficients and the
      reconstructed vertices.
    """
    # Add a batch dimension if the vertices are not batched.
    if len(vertices.shape) == 2:
      vertices = np.expand_dims(vertices, axis=0)

    # Compute the offset from the mean vertex positions, ([A1, ..., An], V, 3]).
    shape_offsets = (vertices - self._mean_vertex_positions).astype(np.float32)
    shape_offsets = np.take(shape_offsets, self._vertex_indices, axis=-2)

    # Flatten the shape offsets to (prod(A1, ..., An), V * 3).
    num_batch_elements = np.prod(shape_offsets.shape[:-2])
    shape_offsets = np.reshape(
        shape_offsets, [num_batch_elements, len(self._vertex_indices) * 3]
    )

    # Transpose the shape offsets to (V * 3, prod(A1, ..., An)).
    shape_offsets = np.swapaxes(shape_offsets, 0, 1)

    # Append an array full of zeros to the right-hand side for the
    # regularization term.
    right_hand_side_regularization_term = np.zeros_like(shape_offsets)
    right_hand_side = np.concatenate(
        [shape_offsets, right_hand_side_regularization_term]
    )

    # Compute the coefficients, (I, [A1, ..., An]]).
    coefficients = np.einsum(
        'iv,v...->i...',
        self._basis_projection_matrix,
        right_hand_side,
    )
    coefficients = np.swapaxes(coefficients, 0, 1)
    coefficients = np.reshape(
        coefficients, [*vertices.shape[:-2], self._num_components]
    )

    reconstruction = None
    if self._compute_reconstruction:
      reconstruction = self._mean_vertex_positions + np.einsum(
          '...i,ivm->...vm', coefficients, self._vertex_basis_subset
      )

    return PCAProjectionResult(
        coefficients=coefficients, reconstruction=reconstruction
    )
