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

"""Unit tests verifying OSS GNM model data loaders.

Tests remote file resolution and download logic for Hugging Face and Kaggle.
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
from gnm.shape.data.versions import gnm_specs
from gnm.shape.oss_data_loaders import oss_data_loaders
import numpy as np


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


class OSSDataLoadersTest(parameterized.TestCase):
  """Tests for remote model loading and caching in oss_data_loaders."""

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
          oss_data_loaders.get_default_gnm_cache_dir(),
          epath.Path('/custom/gnm/cache'),
      )

    with mock.patch.dict(
        os.environ,
        {'XDG_CACHE_HOME': '/custom/xdg/cache'},
        clear=True,
    ):
      self.assertEqual(
          oss_data_loaders.get_default_gnm_cache_dir(),
          epath.Path('/custom/xdg/cache/gnm/models'),
      )

  def test_download_file_success(self):
    dest_path = self.temp_dir / 'downloaded_model.npz'
    mock_response = io.BytesIO(self.dummy_npz_bytes)

    with mock.patch.object(
        urllib.request, 'urlopen', return_value=mock_response
    ):
      result_path = oss_data_loaders._download_file(
          'https://huggingface.co/google/gnm-v3/resolve/main/v3_0/gnm_head.npz',
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
        oss_data_loaders._download_file(
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
        oss_data_loaders, '_download_file', side_effect=_fake_download
    ) as mock_download:
      # First load: triggers download
      data1 = oss_data_loaders.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data1, dict)
      self.assertEqual(mock_download.call_count, 1)
      self.assertTrue(dest_cache_file.exists())

      # Second load: uses cached file directly, does not re-download
      data2 = oss_data_loaders.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data2, dict)
      self.assertEqual(mock_download.call_count, 1)

      # Force download: re-downloads even if cached
      data3 = oss_data_loaders.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
          force_download=True,
      )
      self.assertIsInstance(data3, dict)
      self.assertEqual(mock_download.call_count, 2)

  def test_load_model_from_remote_returns_independent_data(self):
    dest_cache_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'
    dest_cache_file.parent.mkdir(parents=True, exist_ok=True)
    dest_cache_file.write_bytes(self.dummy_npz_bytes)

    data1 = oss_data_loaders.load_model_from_remote(
        gnm_specs.GNMMajorVersion.V3,
        gnm_specs.GNMVariant.HEAD,
        cache_dir=self.temp_dir,
    )
    data1['joint_names'].append('leaked_joint')
    data2 = oss_data_loaders.load_model_from_remote(
        gnm_specs.GNMMajorVersion.V3,
        gnm_specs.GNMVariant.HEAD,
        cache_dir=self.temp_dir,
    )

    self.assertIsNot(data1, data2)
    self.assertNotIn('leaked_joint', data2['joint_names'])

  def test_load_model_from_remote_with_str_cache_dir(self):
    dest_cache_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'

    def _fake_download(url, dest):
      del url
      dest.parent.mkdir(parents=True, exist_ok=True)
      dest.write_bytes(self.dummy_npz_bytes)
      return dest

    with mock.patch.object(
        oss_data_loaders, '_download_file', side_effect=_fake_download
    ):
      data = oss_data_loaders.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=str(self.temp_dir),
      )
      self.assertIsInstance(data, dict)
      self.assertTrue(dest_cache_file.exists())

  def test_load_model_from_huggingface(self):
    dest_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        oss_data_loaders,
        '_resolve_huggingface_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = oss_data_loaders.load_model_from_huggingface(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          self.temp_dir,
          False,
      )

  def test_load_model_from_remote_huggingface(self):
    dest_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        oss_data_loaders,
        '_resolve_huggingface_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = oss_data_loaders.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          source=gnm_specs.GNMRemoteSource.HUGGING_FACE,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          self.temp_dir,
          False,
      )

  def test_load_model_from_kaggle(self):
    dest_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        oss_data_loaders,
        '_resolve_kaggle_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = oss_data_loaders.load_model_from_kaggle(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          self.temp_dir,
          False,
      )

  def test_load_model_from_remote_kaggle(self):
    dest_file = self.temp_dir / 'v3_0' / 'gnm_head.npz'
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        oss_data_loaders,
        '_resolve_kaggle_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = oss_data_loaders.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          source=gnm_specs.GNMRemoteSource.KAGGLE,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          self.temp_dir,
          False,
      )

  def test_load_model_from_remote_invalid_source(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported remote source'):
      oss_data_loaders.load_model_from_remote(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          source='invalid_source',  # pyrefly: ignore[bad-argument-type]
          cache_dir=self.temp_dir,
      )

  def test_resolve_huggingface_model_file_sdk(self):
    mock_hf = mock.MagicMock()
    mock_hf.hf_hub_download.return_value = '/downloaded/path/gnm_head.npz'
    with mock.patch.object(
        oss_data_loaders.importlib, 'import_module', return_value=mock_hf
    ):
      res = oss_data_loaders._resolve_huggingface_model_file(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
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
        oss_data_loaders.importlib,
        'import_module',
        side_effect=ImportError('No HF'),
    ):
      with mock.patch.object(
          oss_data_loaders,
          '_download_file',
          side_effect=RuntimeError('Network unreachable'),
      ):
        with self.assertRaises(FileNotFoundError):
          oss_data_loaders._resolve_huggingface_model_file(
              gnm_specs.GNMMajorVersion.V3,
              gnm_specs.GNMVariant.HEAD,
              cache_dir=self.temp_dir,
          )

  def test_resolve_kaggle_model_file(self):
    fake_kagglehub = mock.MagicMock()
    expected_file = self.temp_dir / 'gnm_head.npz'
    fake_kagglehub.model_download.return_value = str(expected_file)
    with mock.patch.dict('sys.modules', {'kagglehub': fake_kagglehub}):
      path = oss_data_loaders._resolve_kaggle_model_file(
          gnm_specs.GNMMajorVersion.V3,
          gnm_specs.GNMVariant.HEAD,
          cache_dir=self.temp_dir,
      )
      fake_kagglehub.model_download.assert_called_once_with(
          'google/gnm-v3/other/gnm_head_v3_0',
          path='gnm_head.npz',
          force_download=False,
      )
      self.assertEqual(path, expected_file)


if __name__ == '__main__':
  absltest.main()
