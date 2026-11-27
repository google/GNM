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

"""Version conversion utilities for GNM Shape specifications."""

from gnm import gnm_specs
from gnm.shape import gnm_shape_specs


def full_version_to_major(
    version: gnm_shape_specs.GNMShapeVersion,
) -> gnm_specs.GNMMajorVersion:
  """Returns the major version corresponding to a full `GNMShapeVersion`."""
  return gnm_specs.GNMMajorVersion(version.value.split('.')[0])


def major_to_newest_full_version(
    major: gnm_specs.GNMMajorVersion,
) -> gnm_shape_specs.GNMShapeVersion:
  """Returns the newest `GNMShapeVersion` for a given `GNMMajorVersion`."""
  minors = [
      entry
      for entry in gnm_shape_specs.GNMShapeVersion
      if entry.value.split('.')[0] == major.value
  ]
  return max(minors, key=lambda entry: int(entry.value.split('.')[1]))
