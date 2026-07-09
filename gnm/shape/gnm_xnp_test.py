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

"""Tests for gnm_xnp."""

from __future__ import annotations

import dataclasses

from absl.testing import absltest
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_numpy
from gnm.shape import gnm_xnp


class GNMXnpTest(absltest.TestCase):

  def test_annotations_match_schema(self) -> None:
    expected_attributes = set(gnm_data_schema.GNM_DATA_ATTRIBUTES)
    actual_attributes = set(gnm_xnp.GNM.__annotations__.keys())
    self.assertEqual(actual_attributes, expected_attributes)

  def test_dataclass_fields_match_schema(self) -> None:
    expected_attributes = set(gnm_data_schema.GNM_DATA_ATTRIBUTES)
    actual_fields = {f.name for f in dataclasses.fields(gnm_xnp.GNM)}
    self.assertEqual(actual_fields, expected_attributes)

  def test_cannot_instantiate_abstract_gnm(self) -> None:
    with self.assertRaises(TypeError):
      gnm_xnp.GNM()  # pytype: disable=not-instantiable,missing-parameter  # pylint: disable=abstract-class-instantiated

  def test_from_model_data_raises_not_implemented(self) -> None:
    with self.assertRaises(NotImplementedError):
      gnm_xnp.GNM._from_model_data({})  # pylint: disable=protected-access

  def test_cannot_instantiate_concrete_gnm_via_constructor(self) -> None:
    with self.assertRaises(TypeError):
      gnm_numpy.GNM()  # pytype: disable=not-instantiable,missing-parameter


if __name__ == "__main__":
  absltest.main()
