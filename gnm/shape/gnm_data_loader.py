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

"""GNM data loader."""

from collections.abc import Sequence
import functools
import importlib
import os
from typing import Any
import urllib.request

from absl import logging
from etils import epath
from gnm.shape import gnm_data_schema
from gnm.shape.data.versions import gnm_models_catalog
from gnm.shape.data.versions import gnm_specs
import numpy as np

_pkg = __package__ or 'gnm.shape'
_MODELS_VERSIONS_DIR = epath.resource_path(f'{_pkg}.data.versions')
_VARIANT_TO_MODEL_FILE_NAME_MAP = (
    gnm_models_catalog.VARIANT_TO_MODEL_FILE_NAME_MAP
)

DEFAULT_HF_REPO = 'google/gnm-{major}'
DEFAULT_KAGGLE_HANDLE_PREFIX = 'google/gnm-{major}/other'
DEFAULT_HF_CDN_BASE_URL = (
    'https://huggingface.co/google/gnm-{major}/resolve/main'
)


class GNMModelDataNotLinkedError(Exception):
  """Raised when a GNM model data is not linked into the binary."""

  pass


def _get_model_path_from_version_and_variant(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
) -> epath.Path:
  """Returns the GNM model runfiles path for given variant and version."""
  version_value = major_to_newest_full_version(version).value.replace('.', '_')
  version_dir_name = f'v{version_value}'
  model_file_name = f'{_VARIANT_TO_MODEL_FILE_NAME_MAP[variant]}.npz'
  return _MODELS_VERSIONS_DIR / version_dir_name / model_file_name


def major_to_newest_full_version(
    major: gnm_specs.GNMMajorVersion,
) -> gnm_specs.GNMVersion:
  """Returns the newest GNMVersion for a given GNMMajorVersion."""
  minors = [
      e for e in gnm_specs.GNMVersion if e.value.split('.')[0] == major.value
  ]
  return sorted(minors, key=lambda e: int(e.value.split('.')[1]))[-1]


def full_version_to_major(
    version: gnm_specs.GNMVersion,
) -> gnm_specs.GNMMajorVersion:
  """Returns the major version of a GNMVersion."""
  return gnm_specs.GNMMajorVersion(version.value.split('.')[0])


@functools.lru_cache
def load_model_from_runfile(
    version: gnm_specs.GNMMajorVersion, variant: gnm_specs.GNMVariant
) -> dict[str, Any]:
  """Loads GNM model data from a runfile for the given version/variant."""
  model_file = _get_model_path_from_version_and_variant(version, variant)

  logging.info(
      'Loading GNM model version %s, variant %s from runfiles: %s',
      version,
      variant,
      model_file,
  )
  with model_file.open('rb') as f:
    data_dict = dict(np.load(f))

  # Validate the data.
  valid, missing, extra = _validate_gnm_data(data_dict)
  if not valid:
    raise ValueError(
        f'Validation failed for version {version}, variant {variant}.'
        f' Missing: {missing}, Extra: {extra}'
    )

  return _standardize_gnm_data_types(data_dict)


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


def _get_version_dir_name(version: gnm_specs.GNMMajorVersion) -> str:
  """Returns directory name for version (e.g. 'v3_0')."""
  version_value = major_to_newest_full_version(version).value.replace('.', '_')
  return f'v{version_value}'


def _get_model_filename(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
) -> str:
  """Returns filename for model variant (e.g. 'gnm_head.npz')."""
  del version
  return f'{_VARIANT_TO_MODEL_FILE_NAME_MAP[variant]}.npz'


def _load_model_dict_from_file(
    model_file: epath.Path,
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
) -> dict[str, Any]:
  """Loads and standardizes model dict from a local file path."""
  with model_file.open('rb') as f:
    data_dict = dict(np.load(f))

  del version, variant

  # Validate the data.
  valid, missing, extra = _validate_gnm_data(data_dict)
  if not valid:
    raise ValueError(
        f'Validation failed for model from {model_file}.'
        f' Missing: {missing}, Extra: {extra}'
    )

  return _standardize_gnm_data_types(data_dict)


def _resolve_remote_model_file(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    cache_dir: epath.Path,
    force_download: bool = False,
) -> epath.Path:
  """Downloads the model file via HTTP/HTTPS from the official CDN."""
  version_dir_name = _get_version_dir_name(version)
  major_tag = version_dir_name.split('_', maxsplit=1)[0]
  model_file_name = _get_model_filename(version, variant)
  cached_file = cache_dir / version_dir_name / model_file_name
  if cached_file.exists() and not force_download:
    return cached_file

  cdn_base = (
      DEFAULT_HF_CDN_BASE_URL.format(major=major_tag)
      if '{major}' in DEFAULT_HF_CDN_BASE_URL
      else DEFAULT_HF_CDN_BASE_URL
  )
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
    repo_id: str,
    revision: str,
    cache_dir: epath.Path,
    force_download: bool = False,
) -> epath.Path:
  """Resolves model file from HF Hub via huggingface_hub SDK or CDN fallback."""
  version_dir_name = _get_version_dir_name(version)
  major_tag = version_dir_name.split('_', maxsplit=1)[0]
  model_file_name = _get_model_filename(version, variant)
  filename = f'{version_dir_name}/{model_file_name}'

  if '{major}' in repo_id:
    effective_repo_id = repo_id.format(major=major_tag)
  elif repo_id == 'google/gnm':
    effective_repo_id = f'google/gnm-{major_tag}'
  else:
    effective_repo_id = repo_id

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
    handle_prefix: str,
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

  version_dir_name = _get_version_dir_name(version)
  major_tag = version_dir_name.split('_', maxsplit=1)[0]
  model_file_name = _get_model_filename(version, variant)
  npz_stem = model_file_name.removesuffix('.npz')
  variation_slug = f'{npz_stem}_{version_dir_name}'

  if not handle_prefix or handle_prefix in ('google/gnm/other', 'google/gnm'):
    resolved_prefix = f'google/gnm-{major_tag}/other'
  elif '{major}' in handle_prefix:
    resolved_prefix = handle_prefix.format(major=major_tag)
  else:
    resolved_prefix = handle_prefix

  parts = resolved_prefix.strip('/').split('/')
  match len(parts):
    case 1:
      kaggle_handle = f'{parts[0]}/gnm-{major_tag}/other/{variation_slug}'
    case 2:
      kaggle_handle = f'{parts[0]}/{parts[1]}/other/{variation_slug}'
    case 3:
      kaggle_handle = f'{resolved_prefix}/{variation_slug}'
    case _:
      kaggle_handle = resolved_prefix
  downloaded_path = kagglehub.model_download(
      kaggle_handle,
      path=model_file_name,
      force_download=force_download,
  )
  result_path = epath.Path(downloaded_path)
  if result_path.is_dir():
    result_path = result_path / model_file_name
  return result_path


@functools.lru_cache
def load_model_from_remote(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    *,
    cache_dir: epath.PathLike | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
  """Loads GNM model data via HTTP/HTTPS from remote CDN.

  Args:
    version: GNM major version.
    variant: GNM model variant.
    cache_dir: Custom local cache directory (Path or str). Defaults to
      `~/.cache/gnm/models/`.
    force_download: If True, forces redownload even if cached locally.

  Returns:
    A dictionary containing the standardized GNM model data.

  Raises:
    FileNotFoundError: If the model file cannot be downloaded.
    ValueError: If validation of the model data fails.
  """
  cache_path = (
      epath.Path(cache_dir)
      if cache_dir is not None
      else get_default_gnm_cache_dir()
  )
  model_file = _resolve_remote_model_file(
      version, variant, cache_path, force_download
  )
  logging.info(
      'Loading GNM model version %s, variant %s from remote file: %s',
      version,
      variant,
      model_file,
  )
  return _load_model_dict_from_file(model_file, version, variant)


@functools.lru_cache
def load_model_from_huggingface(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    *,
    repo_id: str = DEFAULT_HF_REPO,
    revision: str = 'main',
    cache_dir: epath.PathLike | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
  """Loads GNM model data from Hugging Face Hub.

  Args:
    version: GNM major version.
    variant: GNM model variant.
    repo_id: Hugging Face repository ID. Defaults to 'google/gnm-{major}'.
    revision: Git revision / branch / tag on Hugging Face. Defaults to 'main'.
    cache_dir: Custom local cache directory (Path or str). Defaults to
      `~/.cache/gnm/models/`.
    force_download: If True, forces redownload even if cached locally.

  Returns:
    A dictionary containing the standardized GNM model data.

  Raises:
    FileNotFoundError: If the model file cannot be downloaded from Hugging Face.
    ValueError: If validation of the model data fails.
  """
  cache_path = (
      epath.Path(cache_dir)
      if cache_dir is not None
      else get_default_gnm_cache_dir()
  )
  model_file = _resolve_huggingface_model_file(
      version, variant, repo_id, revision, cache_path, force_download
  )
  logging.info(
      'Loading GNM model version %s, variant %s from Hugging Face: %s',
      version,
      variant,
      model_file,
  )
  return _load_model_dict_from_file(model_file, version, variant)


@functools.lru_cache
def load_model_from_kaggle(
    version: gnm_specs.GNMMajorVersion,
    variant: gnm_specs.GNMVariant,
    *,
    handle_prefix: str = DEFAULT_KAGGLE_HANDLE_PREFIX,
    cache_dir: epath.PathLike | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
  """Loads GNM model data from Kaggle Models.

  Args:
    version: GNM major version.
    variant: GNM model variant.
    handle_prefix: Kaggle handle prefix. Defaults to 'google/gnm-{major}/other'.
    cache_dir: Optional custom local cache directory (Path or str). Note:
      kagglehub manages its own internal cache; accepted for interface
      consistency.
    force_download: If True, forces redownload even if cached locally.

  Returns:
    A dictionary containing the standardized GNM model data.

  Raises:
    ImportError: If kagglehub is not installed.
    FileNotFoundError: If the model file cannot be downloaded from Kaggle.
    ValueError: If validation of the model data fails.
  """
  cache_path = (
      epath.Path(cache_dir)
      if cache_dir is not None
      else get_default_gnm_cache_dir()
  )
  model_file = _resolve_kaggle_model_file(
      version, variant, handle_prefix, cache_path, force_download
  )
  logging.info(
      'Loading GNM model version %s, variant %s from Kaggle: %s',
      version,
      variant,
      model_file,
  )
  return _load_model_dict_from_file(model_file, version, variant)


def _validate_gnm_data(
    data: dict[str, Any],
) -> tuple[bool, Sequence[str], Sequence[str]]:
  """Validates the GNM data dict.

  It returns any extra or missing fields and a boolean indicating if the data
  dict has exactly the expected fields.

  Args:
    data: The GNM data dict to validate.

  Returns:
    A tuple of (bool, Sequence[str], Sequence[str]) indicating if the data dict
    has exactly the expected fields, the missing fields and the extra fields.
  """
  expected_fields = set(gnm_data_schema.GNM_DATA_ATTRIBUTES)
  missing_fields = list(expected_fields - set(data.keys()))
  extra_fields = list(
      set(data.keys()) - set(gnm_data_schema.GNM_DATA_ATTRIBUTES)
  )
  return not missing_fields and not extra_fields, missing_fields, extra_fields


def _standardize_gnm_data_types(data: dict[str, Any]) -> dict[str, Any]:
  """Standardizes the GNM data data types in-place.

  The data loaded from the .npz model files are defined as Numpy arrays. This
  function converts the items to their expected Python types.

  Args:
    data: The GNM data dict to standardize.

  Returns:
    The GNM data dict with standardized data types.
  """
  keys_to_standardize = (
      'version',
      'variant',
      'identity_names',
      'joint_names',
      'expression_names',
      'mesh_component_names',
      'vertex_group_names',
  )
  for k in keys_to_standardize:
    if k not in data:
      raise ValueError(f'Required attribute {k} not found in GNM data.')

  try:
    data['version'] = gnm_specs.GNMVersion(str(data['version']))
  except ValueError as e:
    version_val = data['version']
    raise ValueError(f'Unknown GNM version: {version_val}') from e
  try:
    data['variant'] = gnm_specs.GNMVariant(str(data['variant']))
  except ValueError as e:
    variant_val = data['variant']
    raise ValueError(f'Unknown GNM variant: {variant_val}') from e
  for key in (
      'identity_names',
      'joint_names',
      'expression_names',
      'mesh_component_names',
      'vertex_group_names',
  ):
    data[key] = [str(v) for v in data[key]]

  return data


def _rename_legacy_basis_names(data: dict[str, Any]) -> None:
  """Renames legacy identity and expression basis names to their new names."""
  if 'identity_names' in data:
    identity_renames = [
        ('eyeballs_', 'eyes_'),
    ]
    new_identity_names = []
    for name in data['identity_names']:
      name_str = str(name)
      for old_prefix, new_prefix in identity_renames:
        if name_str.startswith(old_prefix) and not name_str.startswith(
            new_prefix
        ):
          name_str = new_prefix + name_str[len(old_prefix) :]
          break
      new_identity_names.append(name_str)
    data['identity_names'] = new_identity_names

  if 'expression_names' in data:
    expression_renames = [
        ('left_eye_', 'left_eye_region_'),
        ('right_eye_', 'right_eye_region_'),
        ('mouth_', 'lower_face_region_'),
        ('eyeballs_', 'pupils_'),
    ]
    new_expression_names = []
    for name in data['expression_names']:
      name_str = str(name)
      for old_prefix, new_prefix in expression_renames:
        if name_str.startswith(old_prefix) and not name_str.startswith(
            new_prefix
        ):
          name_str = new_prefix + name_str[len(old_prefix) :]
          break
      new_expression_names.append(name_str)
    data['expression_names'] = new_expression_names


def _populate_legacy_vertex_group_aliases(data: dict[str, Any]) -> None:
  """Populates standardized aliases for legacy vertex groups."""
  if 'vertex_group_names' not in data or 'vertex_groups' not in data:
    return

  vertex_group_names_list = [str(v) for v in data['vertex_group_names']]
  vertex_group_weights_array = np.array(data['vertex_groups'], dtype=np.float32)
  if (
      vertex_group_weights_array.ndim != 2
      or len(vertex_group_names_list) != vertex_group_weights_array.shape[0]
  ):
    return

  index_lookup = {name: i for i, name in enumerate(vertex_group_names_list)}
  extra_group_names = []
  extra_group_weights = []

  # Mapping to retrieve new vertex group name by combinating legacy ones.
  # Note that this list is not exhaustive, and mostly here to support the
  # standard API calls in third_party/py/gnm.
  vertex_group_mappings = {
      'upper_teeth_and_gums': ('upper_teeth',),
      'lower_teeth_and_gums': ('lower_teeth',),
      'eyes': ('left_eye', 'right_eye'),
      'eye_interiors': ('eyeball_interior',),
      'eye_exteriors': ('eyeball_exterior',),
      'scleras': ('sclera',),
      'irises': ('iris',),
      'pupils': ('pupil',),
      'ears': ('left_ear', 'right_ear'),
  }
  for target_group_name, source_group_names in vertex_group_mappings.items():
    if target_group_name not in index_lookup and all(
        name in index_lookup for name in source_group_names
    ):
      source_weights_list = [
          vertex_group_weights_array[index_lookup[name]]
          for name in source_group_names
      ]
      extra_group_names.append(target_group_name)
      extra_group_weights.append(np.maximum.reduce(source_weights_list))

  if extra_group_names:
    data['vertex_group_names'] = vertex_group_names_list + extra_group_names
    data['vertex_groups'] = np.concatenate(
        (vertex_group_weights_array, np.stack(extra_group_weights)), axis=0
    )
