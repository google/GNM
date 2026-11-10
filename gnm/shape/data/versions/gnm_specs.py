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

"""Transitional re-export shim for GNM specifications."""

from gnm import gnm_specs as _top_gnm_specs
from gnm.shape.data.versions import gnm_shape_specs as _gnm_shape_specs

# pylint: disable=invalid-name
GNMVersion = _top_gnm_specs.GNMVersion
GNMMajorVersion = _top_gnm_specs.GNMMajorVersion
GNMVariant = _top_gnm_specs.GNMVariant
GNMBodyPart = _top_gnm_specs.GNMBodyPart
GNM_VARIANT_TO_BODY_PART_MAP = _top_gnm_specs.GNM_VARIANT_TO_BODY_PART_MAP
GNMRemoteSource = _gnm_shape_specs.GNMRemoteSource
# pylint: enable=invalid-name
