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
import numpy as np

_pkg = __package__ or 'gnm.shape'
_LANDMARKS_DIR = epath.resource_path(_pkg) / 'data' / 'landmarks'


class GNMLandmarksType(enum.StrEnum):
  """Available GNM landmarks types."""

  HEAD_SPARSE_68 = 'head_sparse_68'


_LANDMARKS_TYPE_TO_BODY_PART_MAP = {
    GNMLandmarksType.HEAD_SPARSE_68: gnm_specs.GNMBodyPart.HEAD,
}


@dataclasses.dataclass(frozen=True)
class LandmarksConfiguration:
  """Configuration holding landmark definition indices and barycentric weights.
  """

  indices: np.ndarray
  weights: np.ndarray


class GNMLandmarksDataNotLinkedError(Exception):
  """Raised when GNM landmark definition data is not linked into the binary."""

  pass


def _check_landmarks_data_linked(landmarks_type: GNMLandmarksType) -> None:
  """Validates that the GNM landmarks file exists in the binary package."""
  landmark_file = _get_landmark_path(landmarks_type)
  if not landmark_file.exists():
    file_name = f'{landmarks_type}.txt'
    build_target = f'//third_party/py/gnm/shape/data/landmarks:{file_name}'
    raise GNMLandmarksDataNotLinkedError(
        f'The GNM landmark data for {landmarks_type} is not linked into the'
        ' binary package. Please check the BUILD file and make sure to include'
        f' \'data = ["{build_target}"]\'.'
    )


def _get_landmark_path(landmarks_type: GNMLandmarksType) -> epath.Path:
  """Returns the path to the landmark definition file."""
  file_name = f'{landmarks_type}.txt'
  return _LANDMARKS_DIR / file_name


def check_body_part_compatibility(
    landmarks_type: GNMLandmarksType,
    body_part: gnm_specs.GNMBodyPart,
) -> None:
  """Checks if the landmark type is compatible with the model body part."""
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
  """Loads landmark indices and weights for the given landmark type.

  Args:
    landmarks_type: The type of landmarks to load.

  Returns:
    A LandmarksConfiguration instance containing indices and weights arrays.
  """
  landmark_file = _get_landmark_path(landmarks_type)
  with landmark_file.open('r') as f:
    data = np.loadtxt(f)
    indices = data[:, ::2].astype(np.int32)
    weights = data[:, 1::2].astype(np.float32)
  return LandmarksConfiguration(indices=indices, weights=weights)
