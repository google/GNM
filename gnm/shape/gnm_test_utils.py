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

"""Test utilities for GNM, e.g. for lazily loading models and mocking TFHub."""

import collections
from collections.abc import Callable, Iterable, Iterator, Mapping
import inspect
from typing import Any, Generic, TypeVar
import unittest

from absl.testing import absltest

from gnm.shape import gnm_numpy
from gnm.shape import gnm_xnp
import numpy as np
import numpy.typing as npt

DTypeLike = npt.DTypeLike


def default_gnm_parameters(
    gnm: gnm_numpy.GNM,
    batch_shape: tuple[int, ...] | None = None,
    dtype: DTypeLike = np.float32,
) -> dict[str, npt.NDArray[np.floating]]:
  """Returns default GNM parameters.

  Args:
    gnm: The GNM model to use.
    batch_shape: The batch shape to use for the parameters.
    dtype: The dtype to use for the parameters.

  Returns:
    A dictionary of GNM parameters.
  """
  if batch_shape is None:
    batch_shape = tuple()

  parameters = {
      'identity': np.zeros(batch_shape + (gnm.identity_dim,)),
      'expression': np.zeros(batch_shape + (gnm.expression_dim,)),
      'rotations': np.zeros(batch_shape + (gnm.num_joints, 3)),
      'translation': np.zeros(batch_shape + (3,)),
  }

  return {k: v.astype(dtype) for k, v in parameters.items()}


def random_gnm_parameters(
    gnm: gnm_numpy.GNM,
    batch_shape: tuple[int, ...] | None = None,
    seed: int | np.random.Generator | None = None,
    identity_range: tuple[float, float] = (-1.0, 1.0),
    expression_range: tuple[float, float] = (-1.0, 1.0),
    rotation_range_degrees: tuple[float, float] = (-30.0, 30.0),
    translation_range_m: tuple[float, float] = (-0.01, 0.01),
    dtype: DTypeLike = np.float32,
) -> dict[str, npt.NDArray[np.floating]]:
  """Returns random GNM parameters sampled from a uniform distribution.

  Args:
    gnm: The GNM model to use.
    batch_shape: The batch shape to use for the parameters.
    seed: The random seed to use. If None, a default seed is used.
    identity_range: The range of values to sample identity from.
    expression_range: The range of values to sample expression from.
    rotation_range_degrees: The range of values to sample rotation from (in
      degrees).
    translation_range_m: The range of values to sample translation from (in
      meters).
    dtype: The dtype to use for the parameters.

  Returns:
    A dictionary of GNM parameters.
  """
  if seed is None:
    seed = np.random.default_rng(0)
  if batch_shape is None:
    batch_shape = tuple()

  rng = np.random.default_rng(seed)

  rotation_range_radians = np.deg2rad(rotation_range_degrees)

  parameters = {
      'identity': rng.uniform(
          *identity_range, size=batch_shape + (gnm.identity_dim,)
      ),
      'expression': rng.uniform(
          *expression_range, size=batch_shape + (gnm.expression_dim,)
      ),
      'rotations': rng.uniform(
          *rotation_range_radians, size=batch_shape + (gnm.num_joints, 3)
      ),
      'translation': rng.uniform(*translation_range_m, size=batch_shape + (3,)),
  }

  return {k: v.astype(dtype) for k, v in parameters.items()}


_ModelT = TypeVar('_ModelT')
_GNMT = TypeVar('_GNMT', bound=gnm_xnp.GNM)


def _variant_name(variant: Any) -> str:
  """Returns the name of a `GNMVariant` (a `StrEnum`) or variant string."""
  return str(variant)


def load_gnm(gnm_cls: type[_GNMT], version: str, variant: Any) -> _GNMT:
  """Loads a GNM model for tests.

  Internally, this loads the model from the runfiles. In the public release, it
  downloads the model from the remote repository.

  Args:
    gnm_cls: The GNM class to instantiate, e.g. `gnm_numpy.GNM`.
    version: The GNM major version, e.g. 'v3'.
    variant: The GNM variant, e.g. 'head' or `GNMVariant.HEAD`.

  Returns:
    The loaded GNM model.
  """
  major_version = gnm_numpy.GNMMajorVersion(version.removeprefix('v'))
  gnm_variant = gnm_numpy.GNMVariant(_variant_name(variant))
  return gnm_cls.from_remote(major_version, gnm_variant)


class LazyGNMDict(Generic[_ModelT]):
  """A `{version: {variant: model}}` mapping which loads models on demand.

  Loading every GNM model up front keeps several GiB of model data in memory
  for the whole test run. This mapping instead loads a model on first access
  and only keeps the `max_loaded` most recently used models in memory. Pair it
  with `ModelOrderedTestLoader` so that tests using the same model run back to
  back, otherwise models get reloaded more often.

  For instance:

    cls.gnms = gnm_test_utils.LazyGNMDict(
        load_fn=functools.partial(gnm_test_utils.load_gnm, gnm_numpy.GNM),
        variants_by_version={'v3': ['head', 'body']},
    )
    cls.addClassCleanup(cls.gnms.clear)
    ...
    gnm = self.gnms.get_or_skip(version, variant)
  """

  def __init__(
      self,
      load_fn: Callable[[str, str], _ModelT],
      variants_by_version: Mapping[str, Iterable[Any]],
      max_loaded: int = 1,
  ):
    """Initializes the mapping without loading any model.

    Args:
      load_fn: Loads the model for a (version, variant) pair, e.g. ('v3',
        'head').
      variants_by_version: The variants available for each version.
      max_loaded: The maximum number of models kept in memory at once.
    """
    if max_loaded < 1:
      raise ValueError(f'max_loaded must be at least 1, got {max_loaded}.')
    self._load_fn = load_fn
    self._variants_by_version = {
        version: tuple(_variant_name(v) for v in variants)
        for version, variants in variants_by_version.items()
    }
    self._max_loaded = max_loaded
    self._loaded: collections.OrderedDict[tuple[str, str], _ModelT] = (
        collections.OrderedDict()
    )

  def __contains__(self, version: object) -> bool:
    return version in self._variants_by_version

  def __getitem__(self, version: str) -> Mapping[str, _ModelT]:
    if version not in self._variants_by_version:
      raise KeyError(version)
    return _LazyGNMVersionView(self, version)

  def __iter__(self) -> Iterator[str]:
    return iter(self._variants_by_version)

  def variants(self, version: str) -> tuple[str, ...]:
    """Returns the variants available for `version`."""
    return self._variants_by_version[version]

  def get_model(self, version: str, variant: Any) -> _ModelT:
    """Returns the model for (version, variant), loading it if needed."""
    key = (version, _variant_name(variant))
    if key[1] not in self._variants_by_version.get(version, ()):
      raise KeyError(key)
    if key in self._loaded:
      self._loaded.move_to_end(key)
      return self._loaded[key]
    # Release the least recently used models before loading a new one, so the
    # old and new models are never in memory at the same time.
    while len(self._loaded) >= self._max_loaded:
      self._loaded.popitem(last=False)
    model = self._load_fn(*key)
    self._loaded[key] = model
    return model

  def get_or_skip(self, version: str, variant: Any) -> _ModelT:
    """Returns the model for (version, variant), or skips the current test.

    Use this in tests parameterized over variants which aren't available in
    every version.

    Args:
      version: The GNM major version, e.g. 'v3'.
      variant: The GNM variant, e.g. 'head' or `GNMVariant.HEAD`.

    Raises:
      unittest.SkipTest: If `variant` isn't available in `version`.
      KeyError: If `version` is unknown.
    """
    variant_name = _variant_name(variant)
    if variant_name not in self.variants(version):
      raise unittest.SkipTest(
          f'variant {variant_name} not supported in {version}.'
      )
    return self.get_model(version, variant_name)

  def clear(self) -> None:
    """Releases all loaded models."""
    self._loaded.clear()


class _LazyGNMVersionView(Mapping[str, _ModelT]):
  """The `{variant: model}` view of a `LazyGNMDict` for a single version."""

  def __init__(self, gnms: LazyGNMDict[_ModelT], version: str):
    self._gnms = gnms
    self._version = version

  def __contains__(self, variant: object) -> bool:
    return _variant_name(variant) in self._gnms.variants(self._version)

  def __getitem__(self, variant: Any) -> _ModelT:
    return self._gnms.get_model(self._version, variant)

  def __iter__(self) -> Iterator[str]:
    return iter(self._gnms.variants(self._version))

  def __len__(self) -> int:
    return len(self._gnms.variants(self._version))


def _get_test_parameters(test_fn: Any) -> Mapping[str, Any]:
  """Returns the arguments of an absl `parameterized` test, if available.

  absl doesn't expose the parameters of a generated test method, so this reads
  them from the closure of the generated method. Returns an empty mapping when
  that fails, e.g. for non-parameterized tests.

  Args:
    test_fn: A test method, possibly generated by absl `parameterized`.
  """
  closure = getattr(test_fn, '__closure__', None)
  code = getattr(test_fn, '__code__', None)
  if not closure or code is None:
    return {}
  try:
    cells = dict(zip(code.co_freevars, (c.cell_contents for c in closure)))
  except ValueError:  # An empty closure cell.
    return {}
  test_method = cells.get('test_method')
  params = cells.get('testcase_params')
  if isinstance(params, Mapping):
    return params
  if test_method is None or not isinstance(params, (tuple, list)):
    return {}
  try:
    bound = inspect.signature(test_method).bind_partial(None, *params)
  except (TypeError, ValueError):
    return {}
  return bound.arguments


def _model_sort_key(test_fn: Any) -> tuple[Any, ...]:
  """Returns a sort key grouping tests by the GNM version and variants used."""
  params = _get_test_parameters(test_fn)
  version = params.get('version')
  if version is None:
    # Run tests which don't use a specific model last.
    return (1,)
  variants = tuple(
      _variant_name(value)
      for name, value in sorted(params.items())
      if 'variant' in name and isinstance(value, str)
  )
  return (0, str(version), variants)


class ModelOrderedTestLoader(absltest.TestLoader):
  """Test loader running the tests of each GNM model back to back.

  By default, test methods run in alphabetical order, which cycles through all
  the GNM models for every test method. This loader instead orders the tests of
  each test case by the `version` and `*variant*` arguments of
  `parameterized` tests, so that a `LazyGNMDict` only needs to load each model
  once per test case. Tests without a `version` argument run last. Sharding is
  unaffected, as absltest assigns tests to shards by name before ordering them.

  Usage:

    if __name__ == '__main__':
      absltest.main(testLoader=gnm_test_utils.ModelOrderedTestLoader())
  """

  def getTestCaseNames(self, testCaseClass):  # pylint: disable=invalid-name
    names = super().getTestCaseNames(testCaseClass)
    return sorted(
        names,
        key=lambda name: _model_sort_key(getattr(testCaseClass, name)),
    )
