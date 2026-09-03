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

"""Unit tests verifying GNM model data loaders.

Tests loaders across runfiles, TFHub, and remote CDNs.
"""

# pylint: disable=protected-access

import email.message
import io
import os
from typing import Any
from unittest import mock
import urllib.error
import urllib.request

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from gnm.shape import gnm_data_loader
from gnm.shape.data.versions import gnm_specs
from gnm.shape.data.versions import gnm_test_catalog
import numpy as np

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_test_catalog.MAJOR_VERSION_TO_VARIANTS_MAP


def _get_dummy_gnm_data_dict() -> dict[str, Any]:
  """Returns a dummy GNM data dictionary."""
  return {
      'version': '3.0',
      'variant': 'head',
      'template_vertex_positions': np.zeros((1, 3)),
      'template_joint_positions': np.zeros((1, 3)),
      'vertex_identity_basis': np.zeros((1, 1, 3)),
      'joint_identity_basis': np.zeros((1, 1, 3)),
      'expression_basis': np.zeros((1, 1, 3)),
      'identity_names': ['id1'],
      'joint_names': ['joint1'],
      'expression_names': ['exp1'],
      'joint_parent_indices': np.array([0]),
      'skinning_weights': np.zeros((1, 1)),
      'quads': np.zeros((1, 4)),
      'triangles': np.zeros((1, 3)),
      'quad_uvs': np.zeros((1, 4, 2)),
      'triangle_uvs': np.zeros((1, 3, 2)),
      'mesh_component_names': ['part1'],
      'mirror_indices': np.array([0]),
      'joint_regressor': np.zeros((1, 1)),
      'pose_correctives_regressor': np.zeros((9, 3)),
      'bone_aligned_template_joint_orientations': np.zeros((1, 3, 3)),
      'vertex_groups': np.zeros((1, 1)),
      'vertex_group_names': ['group1'],
  }


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
      variant=gnm_test_catalog.ALL_VARIANTS,
  )
  def test_load_model_from_runfile_successful(self, version, variant):
    if variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]:
      # Convert string version/variant to Enums.
      major_version = gnm_specs.GNMMajorVersion(version[1:])
      gnm_variant = gnm_specs.GNMVariant(variant)

      data = gnm_data_loader.load_model_from_runfile(major_version, gnm_variant)
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


class GNMRemoteModelLoadingTest(parameterized.TestCase):
  """Tests for remote model loading and caching."""

  def setUp(self):
    super().setUp()
    self.temp_dir = epath.Path(self.create_tempdir().full_path)
    self.dummy_gnm_data_dict = _get_dummy_gnm_data_dict()

    # Save a dummy npz in temp_dir.
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **self.dummy_gnm_data_dict)
    self.dummy_npz_bytes = buffer.getvalue()

  def test_get_default_gnm_cache_dir(self):
    with mock.patch.dict(os.environ, {'GNM_CACHE_DIR': '/custom/gnm/cache'}):
      self.assertEqual(
          gnm_data_loader.get_default_gnm_cache_dir(),
          epath.Path('/custom/gnm/cache'),
      )

    with mock.patch.dict(
        os.environ,
        {'XDG_CACHE_HOME': '/custom/xdg/cache'},
        clear=True,
    ):
      self.assertEqual(
          gnm_data_loader.get_default_gnm_cache_dir(),
          epath.Path('/custom/xdg/cache/gnm/models'),
      )

  def test_download_file_success(self):
    dest_path = self.temp_dir / 'downloaded_model.npz'
    mock_response = io.BytesIO(self.dummy_npz_bytes)

    with mock.patch.object(
        urllib.request, 'urlopen', return_value=mock_response
    ):
      result_path = gnm_data_loader._download_file(
          'https://huggingface.co/google/gnm/resolve/main/v1/gnm_head.npz',
          dest_path,
      )
      self.assertEqual(result_path, dest_path)
      self.assertTrue(dest_path.exists())
      self.assertEqual(dest_path.read_bytes(), self.dummy_npz_bytes)

  def test_download_file_http_error(self):
    dest_path = self.temp_dir / 'fail_model.npz'
    with mock.patch.object(
        urllib.request,
        'urlopen',
        side_effect=urllib.error.HTTPError(
            'https://example.com/not_found.npz',
            404,
            'Not Found',
            email.message.Message(),
            None,
        ),
    ):
      with self.assertRaises(urllib.error.HTTPError):
        gnm_data_loader._download_file(
            'https://example.com/not_found.npz', dest_path
        )
      self.assertFalse(dest_path.exists())

  def test_load_model_from_remote_with_version_and_caching(self):
    dest_cache_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'

    def _fake_download(url, dest):
      del url
      dest.parent.mkdir(parents=True, exist_ok=True)
      dest.write_bytes(self.dummy_npz_bytes)
      return dest

    with mock.patch.object(
        gnm_data_loader, '_download_file', side_effect=_fake_download
    ) as mock_download:
      # First load: triggers download
      data1 = gnm_data_loader.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data1, dict)
      self.assertEqual(mock_download.call_count, 1)
      self.assertTrue(dest_cache_file.exists())

      # Second load: uses cached file directly, does not re-download
      data2 = gnm_data_loader.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data2, dict)
      self.assertEqual(mock_download.call_count, 1)

      # Force download: re-downloads even if cached
      data3 = gnm_data_loader.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
          force_download=True,
      )
      self.assertIsInstance(data3, dict)
      self.assertEqual(mock_download.call_count, 2)

  def test_load_model_from_huggingface(self):
    dest_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        gnm_data_loader,
        '_resolve_huggingface_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = gnm_data_loader.load_model_from_huggingface(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          repo_id='custom/repo',
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          'custom/repo',
          'main',
          self.temp_dir,
          False,
      )

  def test_load_model_from_kaggle(self):
    dest_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        gnm_data_loader,
        '_resolve_kaggle_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = gnm_data_loader.load_model_from_kaggle(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          handle_prefix='custom/handle',
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          'custom/handle',
          self.temp_dir,
          False,
      )

  def test_load_model_from_remote_with_str_cache_dir(self):
    dest_cache_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'

    def _fake_download(url, dest):
      del url
      dest.parent.mkdir(parents=True, exist_ok=True)
      dest.write_bytes(self.dummy_npz_bytes)
      return dest

    with mock.patch.object(
        gnm_data_loader, '_download_file', side_effect=_fake_download
    ):
      # Pass cache_dir as string instead of epath.Path
      data = gnm_data_loader.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=str(self.temp_dir),
      )
      self.assertIsInstance(data, dict)
      self.assertTrue(dest_cache_file.exists())

  def test_resolve_huggingface_model_file_sdk(self):
    mock_hf = mock.MagicMock()
    mock_hf.hf_hub_download.return_value = '/downloaded/path/gnm_head.npz'
    with mock.patch.object(
        gnm_data_loader.importlib, 'import_module', return_value=mock_hf
    ):
      res = gnm_data_loader._resolve_huggingface_model_file(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          repo_id=gnm_data_loader.DEFAULT_HF_REPO,
          revision='main',
          cache_dir=self.temp_dir,
      )
      self.assertEqual(res, epath.Path('/downloaded/path/gnm_head.npz'))
      mock_hf.hf_hub_download.assert_called_once_with(
          repo_id='google/gnm-v3',
          filename='v3_0/gnm_head.npz',
          revision='main',
          cache_dir=str(self.temp_dir),
          force_download=False,
      )

  def test_resolve_huggingface_model_file_fallback_error(self):
    with mock.patch.object(
        gnm_data_loader.importlib,
        'import_module',
        side_effect=ImportError('No HF'),
    ):
      with mock.patch.object(
          gnm_data_loader,
          '_download_file',
          side_effect=RuntimeError('Network unreachable'),
      ):
        with self.assertRaises(FileNotFoundError):
          gnm_data_loader._resolve_huggingface_model_file(
              gnm_specs.GNMMajorVersion.V3,
              gnm_specs.GNMVariant.HEAD,
              repo_id='google/gnm',
              revision='main',
              cache_dir=self.temp_dir,
          )


if __name__ == '__main__':
  absltest.main()
