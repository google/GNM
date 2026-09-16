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

Tests loaders across runfiles and TFHub.
"""

# pylint: disable=protected-access

import io
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from gnm.shape import gnm_data_loader
from gnm.shape.data.versions import gnm_specs
from gnm.shape.data.versions import gnm_test_catalog
from gnm.shape.oss_data_loaders import oss_data_loaders
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

  def setUp(self):
    super().setUp()
    gnm_data_loader.load_model_from_runfile.cache_clear()

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
  """Tests for remote model loading and caching in gnm_data_loader."""

  def setUp(self):
    super().setUp()
    self.temp_dir = epath.Path(self.create_tempdir().full_path)
    self.dummy_gnm_data_dict = _get_dummy_gnm_data_dict()

    # Save a dummy npz in temp_dir.
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **self.dummy_gnm_data_dict)
    self.dummy_npz_bytes = buffer.getvalue()

    # Dynamically determine the latest maintained version and an available
    # variant.
    version_key = gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS[-1]
    self.test_version = gnm_specs.GNMMajorVersion(
        version_key.removeprefix('v')
    )
    variants = gnm_test_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version_key]
    if gnm_specs.GNMVariant.HEAD.value in variants:
      self.test_variant = gnm_specs.GNMVariant.HEAD
    else:
      self.test_variant = gnm_specs.GNMVariant(variants[0])
    version_dir_name = gnm_data_loader._get_version_dir_name(
        self.test_version
    )
    model_file_name = gnm_data_loader._get_model_filename(
        self.test_version, self.test_variant
    )
    self.dest_cache_file = self.temp_dir / version_dir_name / model_file_name

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

  def test_load_model_from_remote(self):
    def _fake_download(url, dest):
      del url
      dest.parent.mkdir(parents=True, exist_ok=True)
      dest.write_bytes(self.dummy_npz_bytes)
      return dest

    with mock.patch.object(
        oss_data_loaders, '_download_file', side_effect=_fake_download
    ) as mock_download:
      # First load: triggers download
      data1 = gnm_data_loader.load_model_from_remote(
          self.test_version,
          self.test_variant,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data1, dict)
      self.assertEqual(mock_download.call_count, 1)
      self.assertTrue(self.dest_cache_file.exists())

      # Second load: uses cached file directly, does not re-download
      data2 = gnm_data_loader.load_model_from_remote(
          self.test_version,
          self.test_variant,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data2, dict)
      self.assertEqual(mock_download.call_count, 1)

      # Force download: re-downloads even if cached
      data3 = gnm_data_loader.load_model_from_remote(
          self.test_version,
          self.test_variant,
          cache_dir=self.temp_dir,
          force_download=True,
      )
      self.assertIsInstance(data3, dict)
      self.assertEqual(mock_download.call_count, 2)

  def test_load_model_from_remote_with_str_cache_dir(self):
    def _fake_download(url, dest):
      del url
      dest.parent.mkdir(parents=True, exist_ok=True)
      dest.write_bytes(self.dummy_npz_bytes)
      return dest

    with mock.patch.object(
        oss_data_loaders, '_download_file', side_effect=_fake_download
    ):
      data = gnm_data_loader.load_model_from_remote(
          self.test_version,
          self.test_variant,
          cache_dir=str(self.temp_dir),
      )
      self.assertIsInstance(data, dict)
      self.assertTrue(self.dest_cache_file.exists())

  def test_load_model_from_remote_huggingface(self):
    dest_file = self.dest_cache_file
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        oss_data_loaders,
        '_resolve_huggingface_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = gnm_data_loader.load_model_from_remote(
          self.test_version,
          self.test_variant,
          source=gnm_specs.GNMRemoteSource.HUGGING_FACE,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          self.test_version,
          self.test_variant,
          self.temp_dir,
          False,
      )

  def test_load_model_from_remote_kaggle(self):
    dest_file = self.dest_cache_file
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(self.dummy_npz_bytes)

    with mock.patch.object(
        oss_data_loaders,
        '_resolve_kaggle_model_file',
        return_value=dest_file,
    ) as mock_resolve:
      data = gnm_data_loader.load_model_from_remote(
          self.test_version,
          self.test_variant,
          source=gnm_specs.GNMRemoteSource.KAGGLE,
          cache_dir=self.temp_dir,
      )
      self.assertIsInstance(data, dict)
      mock_resolve.assert_called_once_with(
          self.test_version,
          self.test_variant,
          self.temp_dir,
          False,
      )


if __name__ == '__main__':
  absltest.main()
