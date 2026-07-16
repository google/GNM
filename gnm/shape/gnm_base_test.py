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

"""Tests for gnm_base."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from absl.testing import absltest
from gnm.shape import gnm_base
from gnm.shape import gnm_data_loader
from gnm.shape.data.versions import gnm_catalog
from gnm.shape.data.versions import gnm_specs

_TEST_MAJOR_VERSION_STR = gnm_catalog.MAINTAINED_MAJOR_VERSIONS[0]
_TEST_MAJOR_VERSION = gnm_specs.GNMMajorVersion(_TEST_MAJOR_VERSION_STR[1:])
_TEST_FULL_VERSION = gnm_data_loader.major_to_newest_full_version(
    _TEST_MAJOR_VERSION
)
_TEST_VARIANT = gnm_specs.GNMVariant(
    gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[_TEST_MAJOR_VERSION_STR][0]
)
_TEST_BODY_PART = gnm_specs.GNMBodyPart(
    gnm_specs.GNM_VARIANT_TO_BODY_PART_MAP[_TEST_VARIANT]
)


class DummyGNM(gnm_base.GNMBase):
  """Dummy GNM subclass for testing."""

  def __init__(
      self,
      version: gnm_specs.GNMVersion,
      variant: gnm_specs.GNMVariant,
  ) -> None:
    self.version = version
    self.variant = variant

  def to_numpy_data_dict(self) -> dict[str, Any]:
    return {"dummy": 1}

  @classmethod
  def _from_model_data(
      cls,
      data_dict: Mapping[str, Any],
  ) -> DummyGNM:
    del data_dict
    return cls(
        version=_TEST_FULL_VERSION,
        variant=_TEST_VARIANT,
    )


class GNMBaseTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.gnm = DummyGNM(
        version=_TEST_FULL_VERSION,
        variant=_TEST_VARIANT,
    )

  def test_properties(self) -> None:
    self.assertEqual(self.gnm.major_version, _TEST_MAJOR_VERSION)
    self.assertEqual(self.gnm.body_part, _TEST_BODY_PART)

  def test_from_gnm(self) -> None:
    new_gnm = DummyGNM.from_gnm(self.gnm)
    self.assertIsInstance(new_gnm, DummyGNM)
    self.assertEqual(new_gnm.version, _TEST_FULL_VERSION)
    self.assertEqual(new_gnm.variant, _TEST_VARIANT)

if __name__ == "__main__":
  absltest.main()
