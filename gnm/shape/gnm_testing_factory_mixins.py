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

"""Testing factory methods mixin for GNM classes."""

from __future__ import annotations

from collections.abc import Mapping
import typing
from typing import Any, Self

from gnm.shape import gnm_data_loader
from gnm.shape.data.versions import gnm_specs
from gnm.shape.data.versions import gnm_test_catalog
import numpy as np


class GNMTestingFactoryMethodsMixin:
  """Mixin providing testing factory classmethods for GNM classes.

  Intended to be mixed into a class that implements `_from_model_data` (such as
  `GNMBase` subclasses) for unit testing.
  """

  if typing.TYPE_CHECKING:

    @classmethod
    def _from_model_data(
        cls,
        model_data: Mapping[str, Any],
    ) -> Self:
      """Creates a GNM instance from model data."""

  @classmethod
  def from_placeholder(
      cls,
      **kwargs: Any,
  ) -> Self:
    """Creates a GNM instance populated with minimal placeholder model data.

    Args:
      **kwargs: Optional field overrides for the model data dictionary.

    Returns:
      A GNM instance initialized with placeholder data and any overrides.
    """
    latest_major_version = gnm_specs.GNMMajorVersion(
        gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[-1].removeprefix('v')
    )
    model_data: dict[str, Any] = {
        'version': gnm_data_loader.major_to_newest_full_version(
            latest_major_version
        ),
        'variant': gnm_specs.GNMVariant.HEAD,
        'template_vertex_positions': np.zeros((1, 3), dtype=np.float32),
        'template_joint_positions': np.zeros((1, 3), dtype=np.float32),
        'vertex_identity_basis': np.zeros((1, 1, 3), dtype=np.float32),
        'joint_identity_basis': np.zeros((1, 1, 3), dtype=np.float32),
        'expression_basis': np.zeros((1, 1, 3), dtype=np.float32),
        'identity_names': ['id1'],
        'joint_names': ['joint1'],
        'expression_names': ['exp1'],
        'joint_parent_indices': np.array([0], dtype=np.int32),
        'skinning_weights': np.zeros((1, 1), dtype=np.float32),
        'quads': np.zeros((1, 4), dtype=np.int32),
        'triangles': np.zeros((1, 3), dtype=np.int32),
        'quad_uvs': np.zeros((1, 4, 2), dtype=np.float32),
        'triangle_uvs': np.zeros((1, 3, 2), dtype=np.float32),
        'mesh_component_names': ['part1'],
        'mirror_indices': np.array([0], dtype=np.int32),
        'joint_regressor': np.zeros((1, 1), dtype=np.float32),
        'pose_correctives_regressor': np.zeros((9, 3), dtype=np.float32),
        'bone_aligned_template_joint_orientations': np.zeros(
            (1, 3, 3), dtype=np.float32
        ),
        'vertex_groups': np.zeros((1, 1), dtype=np.float32),
        'vertex_group_names': ['group1'],
    }
    model_data.update(kwargs)
    return cls._from_model_data(model_data)
