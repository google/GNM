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

"""Tests for gnm_data_loader."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from gnm.shape import gnm_data_loader
from gnm.shape.data.versions import gnm_catalog
from gnm.shape.data.versions import gnm_specs

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP


class GNMDataTest(parameterized.TestCase):

  def test_print_gnm_major_versions(self):
    """Prints all available GNMMajorVersion versions."""
    print('\nAvailable GNM Major Versions:')
    for version in gnm_specs.GNMMajorVersion:
      print(f'  {version.name}: {version.value}')

  def test_print_gnm_versions(self):
    """Prints all available GNMVersion versions."""
    print('\nAvailable GNM MajorMinor Versions:')
    for version in gnm_specs.GNMVersion:
      print(f'  {version.name}: {version.value}')


class GNMModelLoadingTest(parameterized.TestCase):
  """Tests for loading GNM model files."""

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=gnm_catalog.ALL_VARIANTS,
  )
  def test_load_model_from_runfile_successful(self, version, variant):
    if variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]:
      # Convert string version/variant to Enums.
      major_version = gnm_specs.GNMMajorVersion(version[1:])
      gnm_variant = gnm_specs.GNMVariant(variant)

      data = gnm_data_loader.load_model_from_runfile(
          major_version, gnm_variant
      )
      self.assertIsInstance(data, dict)
    else:
      self.skipTest(f'Variant {variant} not available in version {version}')

  def test_load_model_from_runfile_fails_when_file_not_found(self):
    with mock.patch.object(
        gnm_data_loader,
        '_get_model_path_from_version_and_variant',
        return_value=epath.Path('/non/existent/model/file.npz'),
    ):
      with self.assertRaises(FileNotFoundError):
        gnm_data_loader.load_model_from_runfile(
            gnm_specs.GNMMajorVersion.V3,
            gnm_specs.GNMVariant.HEAD,
        )


if __name__ == '__main__':
  absltest.main()
