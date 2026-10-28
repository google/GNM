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

"""GNM ecosystem specifications."""

import enum


class GNMMajorVersion(enum.StrEnum):
  """Specifies GNM major version. Use for version-dependent behavior."""
  V3 = '3'


class GNMVersion(enum.StrEnum):
  """Specifies full GNM version.

  Use ONLY to identify serialized data and for logging.
  """
  V3_0 = '3.0'


class GNMVariant(enum.StrEnum):
  """Specifies the GNM variant."""
  HEAD = 'head'


class GNMBodyPart(enum.StrEnum):
  """Specifies the GNM body part."""
  HEAD = 'head'


GNM_VARIANT_TO_BODY_PART_MAP = {
    GNMVariant.HEAD: GNMBodyPart.HEAD,
}
