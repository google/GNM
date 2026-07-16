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

"""Test utilities for GNM, primarily for mocking TFHub access."""

from gnm.shape import gnm_numpy
import numpy as np
import numpy.typing as npt

DTypeLike = npt.DTypeLike


def default_gnm_parameters(
    gnm: gnm_numpy.GNM,
    batch_shape: tuple[int, ...] | None = None,
    dtype: DTypeLike = np.float32,
) -> dict[str, npt.NDArray[np.floating]]:
  """Returns default GNM parameters.

  Args:
    gnm: The GNM model to use.
    batch_shape: The batch shape to use for the parameters.
    dtype: The dtype to use for the parameters.

  Returns:
    A dictionary of GNM parameters.
  """
  if batch_shape is None:
    batch_shape = tuple()

  parameters = {
      'identity': np.zeros(batch_shape + (gnm.identity_dim,)),
      'expression': np.zeros(batch_shape + (gnm.expression_dim,)),
      'rotations': np.zeros(batch_shape + (gnm.num_joints, 3)),
      'translation': np.zeros(batch_shape + (3,)),
  }

  return {k: v.astype(dtype) for k, v in parameters.items()}


def random_gnm_parameters(
    gnm: gnm_numpy.GNM,
    batch_shape: tuple[int, ...] | None = None,
    seed: int | np.random.Generator | None = None,
    identity_range: tuple[float, float] = (-1.0, 1.0),
    expression_range: tuple[float, float] = (-1.0, 1.0),
    rotation_range_degrees: tuple[float, float] = (-30.0, 30.0),
    translation_range_m: tuple[float, float] = (-0.01, 0.01),
    dtype: DTypeLike = np.float32,
) -> dict[str, npt.NDArray[np.floating]]:
  """Returns random GNM parameters sampled from a uniform distribution.

  Args:
    gnm: The GNM model to use.
    batch_shape: The batch shape to use for the parameters.
    seed: The random seed to use. If None, a default seed is used.
    identity_range: The range of values to sample identity from.
    expression_range: The range of values to sample expression from.
    rotation_range_degrees: The range of values to sample rotation from (in
      degrees).
    translation_range_m: The range of values to sample translation from (in
      meters).
    dtype: The dtype to use for the parameters.

  Returns:
    A dictionary of GNM parameters.
  """
  if seed is None:
    seed = np.random.default_rng(0)
  if batch_shape is None:
    batch_shape = tuple()

  rng = np.random.default_rng(seed)

  rotation_range_radians = np.deg2rad(rotation_range_degrees)

  parameters = {
      'identity': rng.uniform(
          *identity_range, size=batch_shape + (gnm.identity_dim,)
      ),
      'expression': rng.uniform(
          *expression_range, size=batch_shape + (gnm.expression_dim,)
      ),
      'rotations': rng.uniform(
          *rotation_range_radians, size=batch_shape + (gnm.num_joints, 3)
      ),
      'translation': rng.uniform(*translation_range_m, size=batch_shape + (3,)),
  }

  return {k: v.astype(dtype) for k, v in parameters.items()}
