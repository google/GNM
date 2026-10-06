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

"""Unit tests for public GNM factory methods mixin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self
from unittest import mock

from absl.testing import absltest
from gnm.shape import gnm_data_loader
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_factory_mixins


def _create_placeholder_model_data(**extra_fields: Any) -> dict[str, Any]:
  model_data = {attr: attr for attr in gnm_data_schema.GNM_DATA_ATTRIBUTES}
  model_data.update(extra_fields)
  return model_data


class _PlaceholderGNM(gnm_factory_mixins.GNMFactoryMethodsMixin):
  """Placeholder GNM class for testing factory mixin methods."""

  def __init__(self, model_data: Mapping[str, Any]):
    self.model_data = model_data

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> Self:
    return cls(model_data=model_data)


class GNMFactoryMethodsMixinTest(absltest.TestCase):

  def test_from_custom_file_loads_model_and_delegates_to_from_model_data(self):
    expected_data = _create_placeholder_model_data()
    with mock.patch.object(
        gnm_data_loader,
        'load_model_from_custom_file',
        return_value=dict(expected_data),
    ) as mock_load:
      gnm = _PlaceholderGNM.from_custom_file('/path/to/custom_model.npz')
      self.assertIsInstance(gnm, _PlaceholderGNM)
      self.assertEqual(gnm.model_data, expected_data)
      mock_load.assert_called_once_with('/path/to/custom_model.npz')

  def test_from_custom_file_strips_extra_fields_and_logs_warning(self):
    expected_data = _create_placeholder_model_data()
    loaded_data = _create_placeholder_model_data(extra_field='extra_value')
    with (
        mock.patch.object(
            gnm_data_loader,
            'load_model_from_custom_file',
            return_value=loaded_data,
        ),
        self.assertLogs(level='WARNING') as logs,
    ):
      gnm = _PlaceholderGNM.from_custom_file('/path/to/custom_model.npz')
      self.assertEqual(gnm.model_data, expected_data)
      self.assertNotIn('extra_field', gnm.model_data)
      self.assertLen(logs.output, 1)
      self.assertIn('/path/to/custom_model.npz', logs.output[0])
      self.assertIn('extra_field', logs.output[0])

  def test_from_custom_file_fails_for_missing_fields_without_logging_warning(
      self,
  ):
    loaded_data = _create_placeholder_model_data(extra_field='extra_value')
    del loaded_data['version']
    with (
        mock.patch.object(
            gnm_data_loader,
            'load_model_from_custom_file',
            return_value=loaded_data,
        ),
        mock.patch.object(
            gnm_factory_mixins.logging, 'warning'
        ) as mock_warning,
    ):
      with self.assertRaisesRegex(
          ValueError,
          r'Failed to load the custom GNM model data from'
          r' "/path/to/custom_model\.npz"\. Missing fields:.*\'version\'',
      ):
        _PlaceholderGNM.from_custom_file('/path/to/custom_model.npz')
      mock_warning.assert_not_called()

  def test_from_custom_file_rejects_gfs_user_argument(self):
    with self.assertRaises(TypeError):
      _PlaceholderGNM.from_custom_file(
          '/path/to/custom_model.npz',
          gfs_user='test_user',  # pyrefly: ignore[unexpected-keyword]
      )


if __name__ == '__main__':
  absltest.main()
