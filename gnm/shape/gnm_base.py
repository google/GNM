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

"""Base GNM class."""

from __future__ import annotations

import abc
from collections.abc import Mapping
import dataclasses
from typing import Any, Self

from gnm.shape import gnm_data_loader
from gnm.shape.data.versions import gnm_specs


@dataclasses.dataclass(init=False)
class GNMBase(abc.ABC):
  """Base GNM class."""

  version: gnm_specs.GNMVersion
  variant: gnm_specs.GNMVariant

  @classmethod
  def from_local(
      cls,
      version: gnm_specs.GNMMajorVersion,
      variant: gnm_specs.GNMVariant,
  ) -> Self:
    """Creates a GNM instance from a local model file."""
    data_dict = gnm_data_loader.load_model_from_runfile(version, variant)
    return cls._from_model_data(data_dict)

  @classmethod
  def from_gnm(cls, gnm: GNMBase) -> Self:
    """Creates a GNM instance from another GNM instance."""
    data_dict = gnm.to_numpy_data_dict()
    return cls._from_model_data(data_dict)

  @abc.abstractmethod
  def to_numpy_data_dict(self) -> dict[str, Any]:
    """Returns a dictionary of the GNM data represented as NumPy arrays."""
    raise NotImplementedError()

  @classmethod
  @abc.abstractmethod
  def _from_model_data(
      cls,
      data_dict: Mapping[str, Any],
  ) -> Self:
    """Creates a GNM instance from a model data."""
    pass

  @property
  def major_version(self) -> str:
    """Returns the major version of the model."""
    return gnm_data_loader.full_version_to_major(self.version)

  @property
  def body_part(self) -> gnm_specs.GNMBodyPart:
    """Returns the body part of the GNM model."""
    return gnm_specs.GNM_VARIANT_TO_BODY_PART_MAP[self.variant]
