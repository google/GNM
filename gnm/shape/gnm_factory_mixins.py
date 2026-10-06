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

"""Factory methods mixin for GNM classes."""

from __future__ import annotations

from collections.abc import Mapping
import typing
from typing import Any, Self

from etils import epath
from gnm.shape import gnm_data_loader


class GNMFactoryMethodsMixin:
  """Mixin providing factory classmethods for GNM classes."""

  if typing.TYPE_CHECKING:

    @classmethod
    def _from_model_data(
        cls,
        model_data: Mapping[str, Any],
    ) -> Self:
      """Creates a GNM instance from a model data."""

  @classmethod
  def from_custom_file(
      cls,
      model_file: epath.PathLike,
  ) -> Self:
    """Creates a GNM instance from a custom model file.

    The model file must be an .npz archive containing all the expected GNM
    attributes. Extra attributes are ignored.

    Args:
      model_file: Path to the GNM model file (.npz) as Path or str.

    Returns:
      A GNM instance loaded with the model weights.
    """
    model_data = gnm_data_loader.load_model_from_custom_file(model_file)
    return cls._from_model_data(model_data)
