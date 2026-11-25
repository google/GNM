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

"""GNM shape specifications."""

import enum


class GNMShapeVersion(enum.StrEnum):
  """Specifies full GNM shape version.

  Use ONLY to identify serialized data and for logging.
  """
  V3_0 = '3.0'


class GNMShapeVariant(enum.StrEnum):
  """Specifies the GNM shape variant."""
  HEAD = 'head'


class GNMBodyPart(enum.StrEnum):
  """Specifies the GNM body part."""
  HEAD = 'head'


GNM_SHAPE_VARIANT_TO_BODY_PART_MAP = {
    GNMShapeVariant.HEAD: GNMBodyPart.HEAD,
}


class GNMRemoteSource(enum.StrEnum):
  """Specifies the remote repository source for GNM shape models."""
  HTTP = 'http'
  HUGGING_FACE = 'huggingface'
  KAGGLE = 'kaggle'
