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

"""A wrapper of `np.linalg.lstsq` that supports regularization."""

import numpy as np
import numpy.typing as npt

_EPSILON = 1e-16


def regularized_least_squares(
    left_hand_side_data_term: npt.NDArray[np.floating],
    right_hand_side_data_term: npt.NDArray[np.floating],
    regularization: float = 0.0,
    rcond: float | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    int,
    npt.NDArray[np.floating],
]:
  """Solves a regularized least-squares problem.

  Returns the solution of a problem of the form:

  min_x || A x - b ||_2^2 + gamma * ||x||_2^2.

  To solve this problem using a standard least-squares function, we convert it
  to:

  min_x || [A; sqrt(gamma) * I] x - [b; 0] ||_2^2

  where [;] means that we concatenate the two arrays along the rows.

  Args:
    left_hand_side_data_term: The left hand side array,  A, (N, M).
    right_hand_side_data_term: The right hand side term,  b, with shape (N,) or
      (N, K).
    regularization: The scalar regularization that will be used.
    rcond: Cut-off ratio for small singular values of A.

  Returns:
    A tuple that contains the least-squares solution, the residuals for each
    data term, the rank of the left hand side matrix and its singular values.
  """

  regularization_sqrt = np.sqrt(regularization + _EPSILON)

  # Append a diagonal array multiplied with the regularization weight.
  left_hand_side_regularization_term = regularization_sqrt * np.eye(
      left_hand_side_data_term.shape[0],
      left_hand_side_data_term.shape[1],
      dtype=left_hand_side_data_term.dtype,
  )
  left_hand_side = np.concatenate(
      [left_hand_side_data_term, left_hand_side_regularization_term]
  )

  # Append an array full of zeros to the right-hand side for the regularization
  # term.
  right_hand_side_regularization_term = np.zeros_like(right_hand_side_data_term)
  right_hand_side = np.concatenate(
      [right_hand_side_data_term, right_hand_side_regularization_term]
  )

  return np.linalg.lstsq(left_hand_side, right_hand_side, rcond)


class RegularizedLeastSquares:
  """A class for repeated solves using the same left-hand side."""

  def __init__(
      self,
      left_hand_side_data_term: npt.NDArray[np.floating],
      regularization: float = 0.0,
  ):
    """Pre-computes the left-hand side of the regularized least-squares problem.

    Args:
      left_hand_side_data_term: The left hand side array,  A, (N, M).
      regularization: The scalar regularization that will be used.
    """
    regularization_sqrt = np.sqrt(regularization + _EPSILON)

    # Append a diagonal array multiplied with the regularization weight.
    left_hand_side_regularization_term = regularization_sqrt * np.eye(
        left_hand_side_data_term.shape[0],
        left_hand_side_data_term.shape[1],
        dtype=left_hand_side_data_term.dtype,
    )
    left_hand_side = np.concatenate(
        [left_hand_side_data_term, left_hand_side_regularization_term]
    )

    # Compute the pseudo-inverse of the left-hand side.
    self._left_hand_side = (
        np.linalg.pinv(left_hand_side.T @ left_hand_side) @ left_hand_side.T
    )

  def solve(
      self, right_hand_side_data_term: npt.NDArray[np.floating]
  ) -> npt.NDArray[np.floating]:
    """Solves the least-squares problem for a given target.

    Args:
      right_hand_side_data_term: The right hand side term,  b, with shape (N,)
        or (N, K).

    Returns:
      The least-squares solution, x.
    """
    right_hand_side_regularization_term = np.zeros_like(
        right_hand_side_data_term
    )
    right_hand_side = np.concatenate(
        [right_hand_side_data_term, right_hand_side_regularization_term],
        axis=0,
    )
    return self._left_hand_side @ right_hand_side
