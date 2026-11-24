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

"""GNM landmarks definitions and loaders."""

import dataclasses
import enum

from etils import epath
from gnm.shape.data.versions import gnm_specs
import jaxtyping as jt
import numpy as np

_pkg = __package__ or 'gnm.shape'
_LANDMARKS_DIR = epath.resource_path(_pkg) / 'data' / 'landmarks'


class GNMLandmarksType(enum.StrEnum):
  """Available GNM landmarks types."""

  HEAD_SPARSE_68 = 'head_sparse_68'
  HEAD_DENSE_599 = 'head_dense_599'


_LANDMARKS_TYPE_TO_BODY_PART_MAP = {
    GNMLandmarksType.HEAD_SPARSE_68: gnm_specs.GNMBodyPart.HEAD,
    GNMLandmarksType.HEAD_DENSE_599: gnm_specs.GNMBodyPart.HEAD,
}

_LANDMARKS_CONNECTIVITY_FILES: dict[GNMLandmarksType, str] = {
    GNMLandmarksType.HEAD_DENSE_599: 'head_dense_599_connectivity.txt',
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class LandmarksConfiguration:
  """Configuration holding landmark indices, weights, and connectivity.

  Attributes:
    indices: Mesh vertex indices of shape `(L, K)` with dtype `int32`.
    weights: Barycentric interpolation weights of shape `(L, K)` with dtype
      `float32`.
    connectivity: Optional landmark graph edge indices of shape `(E, 2)` with
      dtype `int32`, or `None` when no edge topology is defined.
  """

  indices: jt.Int32[np.ndarray, 'L K']
  weights: jt.Float32[np.ndarray, 'L K']
  connectivity: jt.Int32[np.ndarray, 'E 2'] | None = None


class GNMLandmarksDataNotLinkedError(Exception):
  """Raised when GNM landmark definition data is not linked into binary."""


def check_body_part_compatibility(
    landmarks_type: GNMLandmarksType,
    body_part: gnm_specs.GNMBodyPart,
) -> None:
  """Checks if the landmark type is compatible with the model body part.

  Args:
    landmarks_type: Landmark definition identifier.
    body_part: Target GNM model body part.

  Raises:
    ValueError: If `body_part` is incompatible with `landmarks_type`.
  """
  expected_body_part = _LANDMARKS_TYPE_TO_BODY_PART_MAP[landmarks_type]
  if body_part != expected_body_part:
    raise ValueError(
        f'Landmark type {landmarks_type} is only compatible with GNM models'
        f' of body part {expected_body_part}, but got'
        f' body part {body_part}.'
    )


def load_landmarks(
    landmarks_type: GNMLandmarksType,
) -> LandmarksConfiguration:
  """Loads landmark indices, weights, and optional connectivity edges.

  Args:
    landmarks_type: The type of landmarks to load.

  Returns:
    A `LandmarksConfiguration` containing `indices`, `weights`, and optional
    `connectivity` arrays.

  Raises:
    GNMLandmarksDataNotLinkedError: In Google3 builds if the landmark or
      connectivity data file is not linked into Blaze `data`.
  """
  landmark_file = _get_landmark_path(landmarks_type)
  with landmark_file.open('r') as f:
    raw_data = np.loadtxt(f)
    if raw_data.ndim == 1:
      raw_data = raw_data[np.newaxis, :]
    indices = raw_data[:, ::2].astype(np.int32)
    weights = raw_data[:, 1::2].astype(np.float32)

  connectivity = None
  if (connectivity_file := _get_connectivity_path(landmarks_type)) is not None:
    with connectivity_file.open('r') as f:
      connectivity = np.loadtxt(f, dtype=np.int32)
      if connectivity.ndim == 1:
        connectivity = connectivity[np.newaxis, :]

  return LandmarksConfiguration(
      indices=indices,
      weights=weights,
      connectivity=connectivity,
  )


def _get_landmark_path(landmarks_type: GNMLandmarksType) -> epath.Path:
  """Returns the path to the landmark definition file."""
  file_name = f'{landmarks_type}.txt'
  return _LANDMARKS_DIR / file_name


def _get_connectivity_path(
    landmarks_type: GNMLandmarksType,
) -> epath.Path | None:
  """Returns the path to the optional connectivity file, or `None`."""
  if landmarks_type not in _LANDMARKS_CONNECTIVITY_FILES:
    return None
  file_name = _LANDMARKS_CONNECTIVITY_FILES[landmarks_type]
  return _LANDMARKS_DIR / file_name


def _check_landmarks_data_linked(landmarks_type: GNMLandmarksType) -> None:
  """Validates that landmark and connectivity files exist in runfiles."""
  landmark_file = _get_landmark_path(landmarks_type)
  if not landmark_file.exists():
    file_name = f'{landmarks_type}.txt'
    build_target = f'//third_party/py/gnm/shape/data/landmarks:{file_name}'
    raise GNMLandmarksDataNotLinkedError(
        f'The GNM landmark data for {landmarks_type} is not linked into the'
        ' binary package. Please check the BUILD file and make sure to include'
        f' \'data = ["{build_target}"]\'.'
    )
  connectivity_file = _get_connectivity_path(landmarks_type)
  if connectivity_file is not None and not connectivity_file.exists():
    file_name = _LANDMARKS_CONNECTIVITY_FILES[landmarks_type]
    build_target = f'//third_party/py/gnm/shape/data/landmarks:{file_name}'
    raise GNMLandmarksDataNotLinkedError(
        f'The GNM landmark connectivity data for {landmarks_type} is not'
        ' linked into the binary package. Please check the BUILD file and make'
        f' sure to include \'data = ["{build_target}"]\'.'
    )
