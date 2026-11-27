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

"""Tests for GNM Shape specification utilities."""

import enum
from unittest import mock

from absl.testing import absltest
from gnm import gnm_specs
from gnm.shape import gnm_shape_specs
from gnm.shape import gnm_shape_specs_utils


class GNMShapeSpecsUtilsTest(absltest.TestCase):

  def test_full_version_to_major_returns_expected_major(self):
    self.assertIs(
        gnm_shape_specs_utils.full_version_to_major(
            gnm_shape_specs.GNMShapeVersion.V3_0
        ),
        gnm_specs.GNMMajorVersion.V3,
    )

  def test_full_version_to_major_covers_every_version(self):
    for full_version in gnm_shape_specs.GNMShapeVersion:
      major_version = gnm_shape_specs_utils.full_version_to_major(full_version)
      self.assertIsInstance(major_version, gnm_specs.GNMMajorVersion)
      self.assertTrue(full_version.value.startswith(f'{major_version.value}.'))

  def test_major_to_newest_full_version_returns_latest_minor(self):
    self.assertIs(
        gnm_shape_specs_utils.major_to_newest_full_version(
            gnm_specs.GNMMajorVersion.V3
        ),
        gnm_shape_specs.GNMShapeVersion.V3_0,
    )

  def test_major_to_newest_full_version_selects_highest_minor_numerically(self):
    class _FakeShapeVersion(enum.StrEnum):
      V3_0 = '3.0'
      V3_10 = '3.10'
      V3_2 = '3.2'
      V99_99 = '99.99'

    with mock.patch.object(
        gnm_shape_specs, 'GNMShapeVersion', _FakeShapeVersion
    ):
      self.assertIs(
          gnm_shape_specs_utils.major_to_newest_full_version(
              gnm_specs.GNMMajorVersion.V3
          ),
          _FakeShapeVersion.V3_10,
      )

  def test_major_to_newest_full_version_round_trips_all_majors(self):
    for major_version in gnm_specs.GNMMajorVersion:
      newest_full = gnm_shape_specs_utils.major_to_newest_full_version(
          major_version
      )
      self.assertIs(
          gnm_shape_specs_utils.full_version_to_major(newest_full),
          major_version,
      )


if __name__ == '__main__':
  absltest.main()
