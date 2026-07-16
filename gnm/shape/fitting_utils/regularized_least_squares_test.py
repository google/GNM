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

"""Tests for regularized_least_squares."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape.fitting_utils import regularized_least_squares
import numpy as np

_regularized_least_squares = regularized_least_squares.regularized_least_squares

_DIMENSIONS = [1, 2, 8, 16, 128]
_REGULARIZATION_WEIGHTS = [0.0, 1.0, 10, 100]


class RegularizedLeastSquaresTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)

  @parameterized.parameters(
      list(itertools.product(_DIMENSIONS, _REGULARIZATION_WEIGHTS))
  )
  def test_correct_solution(self, dim: int, weight: float):
    """Tests that the correct solution is returned."""
    array = np.diag(np.arange(1, dim + 1))
    target = self.rng.normal(size=(dim,))

    actual_solution = _regularized_least_squares(
        array, target, weight  # pyrefly: ignore[bad-argument-type]
    )[0]

    # Compute the expected solution using the analytical solution:
    # (A^T @ A + weight * I)^(-1) @ A^T @ b
    expected_solution = (
        np.linalg.inv(array.T @ array + weight * np.eye(dim)) @ array.T @ target
    )
    np.testing.assert_allclose(actual_solution, expected_solution)

  @parameterized.parameters(
      list(itertools.product(_DIMENSIONS, _REGULARIZATION_WEIGHTS))
  )
  def test_solver(self, dim: int, weight: float):
    """Tests that the correct solution is returned."""
    array = np.diag(np.arange(1, dim + 1))
    target = self.rng.normal(size=(dim,))

    solver = regularized_least_squares.RegularizedLeastSquares(
        array, weight  # pyrefly: ignore[bad-argument-type]
    )
    actual_solution = solver.solve(target)

    # Compute the expected solution using the analytical solution:
    # (A^T @ A + weight * I)^(-1) @ A^T @ b
    expected_solution = (
        np.linalg.pinv(array.T @ array + weight * np.eye(dim))
        @ array.T
        @ target
    )
    np.testing.assert_allclose(actual_solution, expected_solution)


if __name__ == "__main__":
  absltest.main()
