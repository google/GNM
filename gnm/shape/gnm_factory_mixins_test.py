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
from gnm.shape import gnm_factory_mixins


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
    with mock.patch.object(
        gnm_data_loader,
        'load_model_from_custom_file',
        return_value={'version': 'version'},
    ) as mock_load:
      gnm = _PlaceholderGNM.from_custom_file('/path/to/custom_model.npz')
      self.assertIsInstance(gnm, _PlaceholderGNM)
      self.assertEqual(gnm.model_data, {'version': 'version'})
      mock_load.assert_called_once_with('/path/to/custom_model.npz')

  def test_from_custom_file_rejects_gfs_user_argument(self):
    with self.assertRaises(TypeError):
      _PlaceholderGNM.from_custom_file(
          '/path/to/custom_model.npz',
          gfs_user='test_user',  # pyrefly: ignore[unexpected-keyword]
      )


if __name__ == '__main__':
  absltest.main()
