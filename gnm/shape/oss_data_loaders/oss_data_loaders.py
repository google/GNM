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

"""Remote model loaders and download resolution utilities for OSS releases.

This module provides loaders for downloading and caching GNM model weights from
public repositories (Hugging Face Hub, Kaggle Models, and HTTP/HTTPS CDNs).
These loaders are isolated from Google3 clients and intended for open-source
consumption, see https://github.com/google/GNM.
"""

# pylint: disable=protected-access

import importlib
import os
from typing import Any
import urllib.request

from absl import logging
from etils import epath
from gnm.shape import gnm_data_loader
from gnm.shape.data.versions import gnm_specs

DEFAULT_KAGGLE_HANDLE_PREFIX = 'google/gnm-{major}/other'
DEFAULT_HF_REPO = 'google/gnm-{major}'
DEFAULT_HF_CDN_BASE_URL = (
    'https://huggingface.co/google/gnm-{major}/resolve/main'
)


def get_default_gnm_cache_dir() -> epath.Path:
  """Returns the default directory for caching downloaded GNM models."""
  if env_cache := os.getenv('GNM_CACHE_DIR'):
    return epath.Path(env_cache)
  if xdg_cache := os.getenv('XDG_CACHE_HOME'):
    return epath.Path(xdg_cache) / 'gnm' / 'models'
  return epath.Path(os.path.expanduser('~/.cache/gnm/models'))


def _download_file(
    url: str,
    destination: epath.Path,
    timeout: int = 120,
) -> epath.Path:
  """Downloads a remote file via HTTP/HTTPS to a destination path atomically."""
  destination.parent.mkdir(parents=True, exist_ok=True)
  temp_destination = destination.with_suffix(
      f'{destination.suffix}.tmp.{os.getpid()}'
  )
  logging.info('Downloading %s to %s...', url, destination)
  req = urllib.request.Request(
      url,
      headers={'User-Agent': 'gnm-client-python'},
  )
  try:
    with (
        urllib.request.urlopen(req, timeout=timeout) as response,
        temp_destination.open('wb') as out_f,
    ):
      while True:
        chunk = response.read(64 * 1024)
        if not chunk:
          break
        out_f.write(chunk)
  except Exception:
    if temp_destination.exists():
      temp_destination.unlink()
    raise

  temp_destination.replace(destination)
  return destination


def _resolve_remote_model_file(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    cache_dir: epath.Path,
    force_download: bool = False,
) -> epath.Path:
  """Downloads the model file via HTTP/HTTPS from the official CDN."""
  version_dir_name = gnm_data_loader._get_version_dir_name(version)
  major_tag = version_dir_name.split('_', maxsplit=1)[0]
  model_file_name = gnm_data_loader._get_model_filename(version, variant)
  cached_file = cache_dir / version_dir_name / model_file_name
  if cached_file.exists() and not force_download:
    return cached_file

  cdn_base = DEFAULT_HF_CDN_BASE_URL.format(major=major_tag)
  cdn_url = f'{cdn_base}/{version_dir_name}/{model_file_name}'
  try:
    return _download_file(cdn_url, cached_file)
  except Exception as e:
    raise FileNotFoundError(
        f'Could not download GNM model file for version {version} and variant'
        f' {variant} from CDN URL: {cdn_url}.\nError: {e}'
    ) from e


def _resolve_huggingface_model_file(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    cache_dir: epath.Path,
    force_download: bool = False,
) -> epath.Path:
  """Resolves model file from HF Hub via huggingface_hub SDK or CDN fallback."""
  version_dir_name = gnm_data_loader._get_version_dir_name(version)
  major_tag = version_dir_name.split('_', maxsplit=1)[0]
  model_file_name = gnm_data_loader._get_model_filename(version, variant)
  filename = f'{version_dir_name}/{model_file_name}'
  effective_repo_id = f'google/gnm-{major_tag}'
  revision = 'main'

  try:
    huggingface_hub = importlib.import_module('huggingface_hub')
    downloaded_path = huggingface_hub.hf_hub_download(
        repo_id=effective_repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir),
        force_download=force_download,
    )
    return epath.Path(downloaded_path)
  except ImportError:
    cdn_url = (
        f'https://huggingface.co/{effective_repo_id}/resolve/{revision}/'
        f'{filename}'
    )
    cached_file = cache_dir / effective_repo_id.replace('/', '_') / filename
    if cached_file.exists() and not force_download:
      return cached_file
    try:
      return _download_file(cdn_url, cached_file)
    except Exception as e:
      raise FileNotFoundError(
          f'Could not download GNM model file for version {version} and variant'
          f' {variant} from Hugging Face CDN URL: {cdn_url}.\nError: {e}'
      ) from e


def _resolve_kaggle_model_file(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    cache_dir: epath.Path,
    force_download: bool = False,
) -> epath.Path:
  """Resolves model file from Kaggle Models using kagglehub SDK."""
  del cache_dir  # kagglehub manages its own internal cache directory.
  try:
    kagglehub = importlib.import_module('kagglehub')
  except ImportError as e:
    raise ImportError(
        'Loading from Kaggle requires kagglehub. Run: pip install kagglehub'
    ) from e

  version_dir_name = gnm_data_loader._get_version_dir_name(version)
  major_tag = version_dir_name.split('_', maxsplit=1)[0]
  model_file_name = gnm_data_loader._get_model_filename(version, variant)
  npz_stem = model_file_name.removesuffix('.npz')
  variation_slug = f'{npz_stem}_{version_dir_name}'
  kaggle_handle = f'google/gnm-{major_tag}/other/{variation_slug}'

  downloaded_path = kagglehub.model_download(
      kaggle_handle,
      path=model_file_name,
      force_download=force_download,
  )
  result_path = epath.Path(downloaded_path)
  if result_path.is_dir():
    result_path = result_path / model_file_name
  return result_path


def load_model_from_remote(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    source: gnm_specs.GNMRemoteSource = gnm_specs.GNMRemoteSource.HTTP,
    *,
    cache_dir: epath.PathLike | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
  """Loads GNM model data via remote repository (HTTP, Hugging Face, or Kaggle).

  Args:
    version: GNM major version.
    variant: GNM model variant.
    source: Remote repository source. Defaults to HTTP (CDN direct download).
    cache_dir: Custom local cache directory (Path or str). Defaults to
      `~/.cache/gnm/models/`.
    force_download: If True, forces redownload even if cached locally.

  Returns:
    A dictionary containing the standardized GNM model data.

  Raises:
    ImportError: If the required SDK is not installed (for Kaggle).
    FileNotFoundError: If the model file cannot be downloaded.
    ValueError: If validation of the model data fails or source is invalid.
  """
  cache_path = (
      epath.Path(cache_dir)
      if cache_dir is not None
      else get_default_gnm_cache_dir()
  )
  if source in (
      gnm_specs.GNMRemoteSource.HTTP,
      'http',
  ):
    model_file = _resolve_remote_model_file(
        version, variant, cache_path, force_download
    )
  elif source in (
      gnm_specs.GNMRemoteSource.HUGGING_FACE,
      'huggingface',
      'hf',
  ):
    model_file = _resolve_huggingface_model_file(
        version, variant, cache_path, force_download
    )
  elif source in (
      gnm_specs.GNMRemoteSource.KAGGLE,
      'kaggle',
  ):
    model_file = _resolve_kaggle_model_file(
        version, variant, cache_path, force_download
    )
  else:
    raise ValueError(f'Unsupported remote source: {source}')

  logging.info(
      'Loading GNM model version %s, variant %s from %s: %s',
      version,
      variant,
      source,
      model_file,
  )
  return gnm_data_loader._load_model_dict_from_file(
      model_file, version, variant
  )


def load_model_from_huggingface(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    *,
    cache_dir: epath.PathLike | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
  """Loads GNM model data from Hugging Face Hub."""
  return load_model_from_remote(
      version=version,
      variant=variant,
      source=gnm_specs.GNMRemoteSource.HUGGING_FACE,
      cache_dir=cache_dir,
      force_download=force_download,
  )


def load_model_from_kaggle(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    *,
    cache_dir: epath.PathLike | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
  """Loads GNM model data from Kaggle Models."""
  return load_model_from_remote(
      version=version,
      variant=variant,
      source=gnm_specs.GNMRemoteSource.KAGGLE,
      cache_dir=cache_dir,
      force_download=force_download,
  )
