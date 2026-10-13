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

"""Unit tests for GNM data schema."""

from typing import Any

from absl.testing import absltest
from gnm.shape import gnm_data_schema


def _create_sample_model_data(**extra_fields: Any) -> dict[str, Any]:
  model_data = {attr: attr for attr in gnm_data_schema.GNM_DATA_ATTRIBUTES}
  model_data.update(extra_fields)
  return model_data


class GNMDataSchemaTest(absltest.TestCase):

  def test_get_extra_fields_lists_extra_fields(self):
    model_data = _create_sample_model_data(
        extra_field_1=1,
        extra_field_2=2,
    )

    extra = gnm_data_schema.get_extra_fields(model_data)

    self.assertCountEqual(['extra_field_1', 'extra_field_2'], extra)

  def test_get_extra_fields_returns_empty_list_when_no_extra_keys(self):
    model_data = _create_sample_model_data()

    extra = gnm_data_schema.get_extra_fields(model_data)

    self.assertEmpty(extra)

  def test_strip_extra_fields_removes_extra_keys_in_place(self):
    model_data = _create_sample_model_data(
        extra_field_1=1,
        extra_field_2=2,
    )

    gnm_data_schema.strip_extra_fields(model_data)

    self.assertNotIn('extra_field_1', model_data)
    self.assertNotIn('extra_field_2', model_data)
    self.assertEqual(model_data, _create_sample_model_data())

  def test_get_missing_fields_returns_missing_keys(self):
    model_data = _create_sample_model_data()
    del model_data['version']
    del model_data['variant']

    missing = gnm_data_schema.get_missing_fields(model_data)

    self.assertCountEqual(missing, ['version', 'variant'])

  def test_is_model_data_valid_returns_true_for_valid_data(self):
    model_data = _create_sample_model_data()

    self.assertTrue(gnm_data_schema.is_model_data_valid(model_data))

  def test_is_model_data_valid_returns_false_for_missing_fields(self):
    model_data = _create_sample_model_data()
    del model_data['version']

    self.assertFalse(gnm_data_schema.is_model_data_valid(model_data))

  def test_is_model_data_valid_returns_false_for_extra_fields(self):
    model_data = _create_sample_model_data(extra_field=1)

    self.assertFalse(gnm_data_schema.is_model_data_valid(model_data))


if __name__ == '__main__':
  absltest.main()
