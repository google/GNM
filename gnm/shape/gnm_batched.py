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

"""Frozen dataclasses which behave like arrays over their batch dimensions.

A `BatchedDataclass` is a frozen dataclass whose array fields share the same
leading batch dimensions `*shape`, followed by per-field inner dimensions
declared with `etils.array_types` annotations:

```python
from gnm.shape import gnm_batched
from gnm.shape import gnm_typing

enpt = gnm_typing.enpt

class Ray(gnm_batched.BatchedDataclass):
  origin: enpt.FloatArray['*shape 3']
  direction: enpt.FloatArray['*shape 3']
  name: str = 'ray'  # Non-array fields are static metadata.

rays = Ray(origin=np.zeros((4, 3)), direction=np.ones((4, 3)))
```

Subclasses are automatically turned into frozen dataclasses (do not decorate
them with `@dataclasses.dataclass`). Their fields are classified from their
annotations: array, nested and static fields.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
import types
import typing
from typing import Any, ClassVar

from etils import enp
import numpy as np

Shape = tuple[int, ...]
# An array of any backend (NumPy, JAX, TensorFlow or PyTorch).
Array = Any
# A NumPy-like module (np, jnp, tnp or torch).
NpModule = Any

# Leading token of the shape annotations of the array fields.
_BATCH_TOKEN = '*shape'

# ------------------------------------------------------------------------------
# Fields specification.
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ArrayField:
  """Specification of an array field.

  Attributes:
    inner_dims: The inner (non-batch) dimensions, e.g. `('J', 3)` for
      `FloatArray['*shape J 3']`.
    dtype: The `etils` dtype of the annotation, e.g. `AnyFloat` for
      `FloatArray`.
  """

  inner_dims: tuple[int | str, ...]
  dtype: enp.dtypes.DType

  @property
  def inner_ndim(self) -> int:
    """The number of inner dimensions."""
    return len(self.inner_dims)


@dataclasses.dataclass(frozen=True)
class _ClassSpec:
  """Classification of the fields of a `BatchedDataclass`.

  Attributes:
    arrays: The array fields.
    nested: The nested `BatchedDataclass` fields and their annotated type.
    static: The static (non-array) fields and their type hint.
    children: The names of the array and nested fields (the pytree children), in
      declaration order.
    optional: The names of the children annotated `| None`, the only ones which
      may be None.
  """

  arrays: Mapping[str, _ArrayField]
  nested: Mapping[str, type[BatchedDataclass]]
  static: Mapping[str, Any]
  children: tuple[str, ...]
  optional: frozenset[str]


def _is_optional(hint: Any) -> bool:
  """Returns whether `hint` is a `T | None` or `Optional[T]` hint."""
  return typing.get_origin(hint) in (
      typing.Union,
      types.UnionType,
  ) and types.NoneType in typing.get_args(hint)


def _strip_optional(hint: Any) -> Any:
  """Returns `T` for `T | None` or `Optional[T]` hints, else `hint`."""
  if _is_optional(hint):
    arguments = [a for a in typing.get_args(hint) if a is not types.NoneType]
    if len(arguments) == 1:
      return arguments[0]
  return hint


def _parse_array_field(
    cls: type[Any], name: str, hint: enp.typing.ArrayAliasMeta
) -> _ArrayField:
  """Parses an `etils.array_types` annotation, e.g. FloatArray['*shape J 3']."""
  tokens = hint.shape.split()
  if tokens[:1] != [_BATCH_TOKEN] or any(
      '*' in token or token == '...' for token in tokens[1:]
  ):
    raise TypeError(
        f'The annotation of {cls.__name__}.{name} must be of the form'
        f" `{hint.__name__}['{_BATCH_TOKEN} <inner dims>']`, e.g."
        f" `FloatArray['{_BATCH_TOKEN} N 3']`, got {hint!r}."
    )
  inner_dims = tuple(
      int(token) if token.isdigit() else token for token in tokens[1:]
  )
  return _ArrayField(inner_dims=inner_dims, dtype=hint.dtype)


def _is_batched_hint(hint: Any) -> bool:
  """Returns whether `hint` is (or contains) an array or a batched type."""
  if typing.get_origin(hint) in (typing.Union, types.UnionType):
    return any(_is_batched_hint(argument) for argument in typing.get_args(hint))
  if isinstance(hint, enp.typing.ArrayAliasMeta):
    return True
  if not isinstance(hint, type) or typing.get_origin(hint) is not None:
    return False
  lazy = enp.lazy
  return (
      issubclass(hint, (np.ndarray, BatchedDataclass))
      or (lazy.has_jax and issubclass(hint, lazy.jax.Array))
      or (lazy.has_torch and issubclass(hint, lazy.torch.Tensor))
      or (lazy.has_tf and issubclass(hint, lazy.tf.Tensor))
  )


@functools.cache
def _class_spec(cls: type[BatchedDataclass]) -> _ClassSpec:
  """Returns the classification of the fields of `cls`, from its annotations."""
  try:
    hints = typing.get_type_hints(cls)
  except NameError as e:
    raise TypeError(
        f'Cannot resolve the annotations of {cls.__name__}: {e}. Annotations'
        ' must reference module-level names.'
    ) from e
  arrays = {}
  nested = {}
  static = {}
  for field in dataclasses.fields(cls):  # pyrefly: ignore[bad-argument-type]
    hint = _strip_optional(hints[field.name])
    if isinstance(hint, enp.typing.ArrayAliasMeta):
      arrays[field.name] = _parse_array_field(cls, field.name, hint)
    elif isinstance(hint, type) and issubclass(hint, BatchedDataclass):
      nested[field.name] = hint
    elif _is_batched_hint(hint):
      # Would silently be a static field: never indexed, nor converted.
      raise TypeError(
          f'Unsupported annotation of {cls.__name__}.{field.name}:'
          f' {hints[field.name]!r}. Array fields must be annotated with a'
          ' single `etils.array_types` type, e.g.'
          f" `FloatArray['{_BATCH_TOKEN} N 3']`, and nested fields with a"
          ' single batched type (both optionally `| None`).'
      )
    else:
      static[field.name] = hint
  children = tuple(
      field.name
      for field in dataclasses.fields(cls)  # pyrefly: ignore[bad-argument-type]
      if field.name in arrays or field.name in nested
  )
  return _ClassSpec(
      arrays=arrays,
      nested=nested,
      static=static,
      children=children,
      optional=frozenset(
          name for name in children if _is_optional(hints[name])
      ),
  )


# ------------------------------------------------------------------------------
# BatchedDataclass.
# ------------------------------------------------------------------------------


@typing.dataclass_transform(frozen_default=True, eq_default=False)
class BatchedDataclass:
  """Frozen dataclass whose array fields share leading batch dimensions.

  See the module docstring for an overview.

  Instances are compared by identity (`==` is not overridden), as array
  equality is ambiguous; use `assert_allclose` in tests.

  The arrays are neither copied nor made read-only: indexing, `reshape` and
  `replace` may share memory with the original arrays. Treat them as read-only
  and use `replace` to update them.
  """

  # Whether fields with no or size-1 batch dimensions are broadcast at
  # construction. Set with `class A(BatchedDataclass, broadcast=False)`.
  _broadcast: ClassVar[bool] = True

  def __init_subclass__(cls, *, broadcast: bool | None = None, **kwargs):
    super().__init_subclass__(**kwargs)
    if broadcast is not None:
      cls._broadcast = broadcast
    dataclasses.dataclass(frozen=True, eq=False, repr=False)(cls)
    reserved = [
        field.name
        for field in dataclasses.fields(cls)  # pyrefly: ignore[bad-argument-type]
        if hasattr(BatchedDataclass, field.name)
    ]
    if reserved:
      raise TypeError(
          f'The fields {reserved} of {cls.__name__} are reserved names of'
          ' `BatchedDataclass` attributes (e.g. `shape` or `reshape`).'
      )

  def __post_init__(self) -> None:
    _class_spec(type(self))  # Classifies the fields (or raises), once.
