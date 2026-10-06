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

"""GNM data schema."""

from collections.abc import Mapping, MutableMapping
from typing import Any

GNM_DATA_ATTRIBUTES = [
    'version',
    'variant',
    'template_vertex_positions',
    'template_joint_positions',
    'vertex_identity_basis',
    'joint_identity_basis',
    'expression_basis',
    'identity_names',
    'joint_names',
    'expression_names',
    'joint_parent_indices',
    'skinning_weights',
    'quads',
    'triangles',
    'quad_uvs',
    'triangle_uvs',
    'mesh_component_names',
    'mirror_indices',
    'joint_regressor',
    'pose_correctives_regressor',
    'bone_aligned_template_joint_orientations',
    'vertex_groups',
    'vertex_group_names',
]


def get_extra_fields(model_data: Mapping[str, Any]) -> list[str]:
  """Lists extra fields not in GNM_DATA_ATTRIBUTES.

  Args:
    model_data: The GNM model data dictionary.

  Returns:
    A list of extra field names.
  """
  expected = set(GNM_DATA_ATTRIBUTES)
  return [k for k in model_data if k not in expected]


def strip_extra_fields(model_data: MutableMapping[str, Any]) -> None:
  """Strips extra fields not in GNM_DATA_ATTRIBUTES in place.

  Args:
    model_data: The GNM model data dictionary.
  """
  for extra in get_extra_fields(model_data):
    del model_data[extra]


def get_missing_fields(model_data: Mapping[str, Any]) -> list[str]:
  """Returns a list of required fields missing from the GNM model data."""
  expected_fields = set(GNM_DATA_ATTRIBUTES)
  return [f for f in expected_fields if f not in model_data]


def is_model_data_valid(model_data: Mapping[str, Any]) -> bool:
  """Returns True if model_data has no missing and no extra fields."""
  return not get_extra_fields(model_data) and not get_missing_fields(model_data)
