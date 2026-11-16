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
rays.shape  # (4,)
```

Subclasses are automatically turned into frozen dataclasses (do not decorate
them with `@dataclasses.dataclass`), and support:

* Validation at construction: inner dimensions are checked against the
  annotations, including named dimensions shared between fields (e.g. `'*shape
  N 3'` and `'*shape N'` must agree on `N`), and the dtypes are normalized
  (e.g. integers are cast to float for `FloatArray` fields).
* Broadcasting at construction: fields without batch dimensions, or with batch
  dimensions of size 1 and the same rank as the other fields, are broadcast to
  the common batch shape. Disable with `class A(BatchedDataclass,
  broadcast=False)`.
* Arrays of any backend (NumPy, JAX, TensorFlow, PyTorch). NumPy arrays and
  array-likes are converted to the backend of the other fields.
* Composition: fields annotated with another `BatchedDataclass` share the batch
  dimensions of their parent, and all the operations apply recursively.
  Nested values without arrays (e.g. with only optional fields set to None)
  have no batch shape, and adapt to the batch shape of their parent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import math
import types
import typing
from typing import Any, ClassVar, Self

from etils import enp
import ml_dtypes
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
# Array helpers.
# ------------------------------------------------------------------------------


def _get_xnp(cls: type[Any], name: str, value: Any) -> NpModule:
  try:
    return enp.lazy.get_xnp(value, strict=False)
  except TypeError as e:
    raise TypeError(
        f'{cls.__name__}.{name} must be an array, got {type(value).__name__}.'
    ) from e


def _common_xnp(cls: type[Any], xnps: Mapping[str, NpModule]) -> NpModule:
  """Returns the backend of the arrays: NumPy arrays adopt the other backend."""
  backends = {
      name: xnp for name, xnp in xnps.items() if not enp.lazy.is_np_xnp(xnp)
  }
  if len(set(backends.values())) > 1:
    details = ', '.join(
        f'{name}: {xnp.__name__}' for name, xnp in backends.items()
    )
    raise ValueError(
        f'Cannot mix array backends in {cls.__name__} ({details}). Convert them'
        ' first, e.g. with `as_jax()`.'
    )
  return next(iter(backends.values()), np)


def _is_extended_dtype(dtype: np.dtype) -> bool:
  """Returns whether `dtype` is an `ml_dtypes` type, e.g. bfloat16 or int4."""
  return dtype.kind == 'V' and dtype.names is None and dtype.type is not np.void


def _is_float_dtype(dtype: np.dtype | None) -> bool:
  # Also extended floating-point types (e.g. `ml_dtypes.bfloat16`), which have
  # the `V` kind, but not other void (e.g. structured) dtypes.
  return dtype is not None and (
      np.issubdtype(dtype, np.floating)
      or (_is_extended_dtype(dtype) and 'float' in dtype.name)
  )


def _extended_dtype(name: str) -> np.dtype | None:
  """Returns the `ml_dtypes` dtype `name` (e.g. 'bfloat16'), if any."""
  extended = getattr(ml_dtypes, name, None)
  return np.dtype(extended) if isinstance(extended, type) else None


def _dtype(value: Array) -> np.dtype | None:
  """Returns the NumPy dtype of an array of any backend."""
  if enp.lazy.is_torch(value):  # `etils` does not support e.g. torch.bfloat16.
    extended = _extended_dtype(str(value.dtype).removeprefix('torch.'))
    if extended is not None:
      return extended
  return enp.lazy.dtype_from_array(value)


def _backend_dtype(dtype: np.dtype, xnp: NpModule) -> Any:
  """Returns the `xnp` dtype of the NumPy `dtype`."""
  if enp.lazy.is_torch_xnp(xnp) and _is_extended_dtype(dtype):
    return getattr(xnp, dtype.name)  # E.g. torch.bfloat16.
  return enp.lazy.as_dtype(dtype, xnp=xnp)


def _cast(value: Array, dtype: np.dtype) -> Array:
  """Casts an array to `dtype`, in its backend, keeping PyTorch gradients."""
  if _dtype(value) == dtype:
    return value
  xnp = enp.lazy.get_xnp(value)
  backend_dtype = _backend_dtype(dtype, xnp)
  if enp.lazy.is_torch_xnp(xnp):
    # Not `torch.asarray`, whose result does not require grad in released
    # PyTorch versions.
    return value.to(backend_dtype)
  if enp.lazy.is_tf_xnp(xnp):  # `tf.Tensor` has no `astype`.
    return xnp.asarray(value, dtype=backend_dtype)
  return value.astype(backend_dtype)


def _to_numpy(value: Array) -> np.ndarray:
  """Converts a NumPy, JAX, TensorFlow or PyTorch array to NumPy."""
  if enp.lazy.is_torch(value):
    value = value.detach().cpu()
    dtype = _dtype(value)
    if dtype is not None and _is_extended_dtype(dtype):
      # PyTorch cannot convert e.g. bfloat16 to NumPy: convert the bits.
      bits = value.view(getattr(enp.lazy.torch, f'int{8 * dtype.itemsize}'))
      return bits.numpy().view(dtype)
  return np.asarray(value)


def _array_spec(value: Array) -> enp.ArraySpec | None:
  """Returns the shape and dtype of an array, displayed as e.g. `f32[4 3]`."""
  if enp.lazy.is_torch(value):  # Not supported by `ArraySpec.from_array`.
    return enp.ArraySpec(tuple(value.shape), _dtype(value))
  return enp.ArraySpec.from_array(value)


def _convert(value: Array, xnp: NpModule) -> Array:
  """Converts an array to the `xnp` backend."""
  if enp.lazy.get_xnp(value, strict=False) is xnp:
    return value
  value = _to_numpy(value)
  if enp.lazy.is_np_xnp(xnp):
    return value
  if enp.lazy.is_torch_xnp(xnp):
    # PyTorch does not support negative strides (e.g. of `x[::-1]`), nor
    # extended dtypes (e.g. bfloat16, converted from their bits).
    dtype = value.dtype
    if _is_extended_dtype(dtype):
      value = value.view(f'i{dtype.itemsize}')
    if any(stride < 0 for stride in value.strides) or not value.flags.writeable:
      value = value.copy()
    tensor = xnp.from_numpy(value)
    if dtype != value.dtype:
      tensor = tensor.view(getattr(xnp, dtype.name))
    return tensor
  return xnp.asarray(value)


def _torch_device(what: str, arrays: Mapping[str, Array]) -> Any:
  """Returns the device of the PyTorch tensors among `arrays`, if any.

  The arrays converted to PyTorch (e.g. from NumPy, on the CPU) are moved to
  it.

  Args:
    what: The merged or constructed value, for error messages.
    arrays: The arrays, of any backend, by name.

  Returns:
    The device of the PyTorch tensors, or None if there is none.

  Raises:
    ValueError: If the PyTorch tensors are on different devices.
  """
  devices = {
      name: array.device
      for name, array in arrays.items()
      if enp.lazy.is_torch(array)
  }
  if len(set(devices.values())) > 1:
    details = ', '.join(f'{name}: {device}' for name, device in devices.items())
    raise ValueError(
        f'Cannot mix PyTorch devices in {what} ({details}). Move the tensors'
        ' first, e.g. with `to(device)`.'
    )
  return next(iter(devices.values()), None)


def _to_device(value: Array, device: Any) -> Array:
  """Moves a PyTorch tensor to `device`, unless it is None."""
  if device is None or value.device == device:
    return value
  return value.to(device)


def _needs_cast(field: _ArrayField, value: Array, xnp: NpModule) -> bool:
  """Returns whether `value` must be converted to match `field` and `xnp`."""
  if not enp.lazy.is_array(value) or enp.lazy.get_xnp(value) is not xnp:
    return True
  dtype = field.dtype
  if isinstance(dtype, enp.dtypes.AnyDType):
    return False
  value_dtype = _dtype(value)
  if isinstance(dtype, enp.dtypes.AnyFloat):
    return not _is_float_dtype(value_dtype)
  if isinstance(dtype, enp.dtypes.AnyInt):
    return not np.issubdtype(value_dtype, np.integer)
  assert isinstance(dtype, enp.dtypes.NpDType), dtype
  return value_dtype != dtype.np_dtype


def _inner_shape(value: Array, inner_ndim: int) -> Shape:
  shape = tuple(value.shape)
  return shape[len(shape) - inner_ndim :] if inner_ndim else ()


def _batch_shape(value: Array, inner_ndim: int) -> Shape:
  shape = tuple(value.shape)
  return shape[: len(shape) - inner_ndim]


def _as_shape(shape: int | Sequence[int]) -> Shape:
  if isinstance(shape, (int, np.integer)):
    return (int(shape),)
  return tuple(int(dim) for dim in shape)


def _broadcast_batch_shapes(
    cls: type[Any], shapes: Mapping[str, Shape], *, broadcast: bool
) -> Shape:
  """Returns the common batch shape of the fields.

  Args:
    cls: The class, for error messages.
    shapes: The batch shape of each field.
    broadcast: Whether fields without batch dimensions, or with batch dimensions
      of size 1 and the full batch rank, are broadcast.

  Returns:
    The common batch shape.

  Raises:
    ValueError: If the batch shapes are incompatible.
  """
  unique = set(shapes.values())
  if broadcast:
    batched = [shape for shape in unique if shape]
    ndim = max(map(len, batched), default=0)
    # Lower-rank batch shapes are ambiguous.
    compatible = all(len(shape) == ndim for shape in batched)
    # The sizes other than 1 along each axis, which must agree.
    sizes = [
        {shape[axis] for shape in batched} - {1}
        for axis in range(ndim if compatible else 0)
    ]
    compatible = compatible and all(
        len(axis_sizes) <= 1 for axis_sizes in sizes
    )
    result = tuple(min(axis_sizes, default=1) for axis_sizes in sizes)
  else:
    result = next(iter(unique), ())
    compatible = len(unique) <= 1
  if not compatible:
    details = ', '.join(f'{name}: {shape}' for name, shape in shapes.items())
    if broadcast:
      rule = (
          'Fields must have the same batch shape, or no batch dimensions, or'
          ' batch dimensions of size 1 with the full batch rank.'
      )
    else:
      rule = (
          f'{cls.__name__} does not broadcast, so all the fields must have the'
          ' same batch shape.'
      )
    raise ValueError(
        f'Conflicting batch shapes in {cls.__name__} ({details}). {rule}'
    )
  return result


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
    cls = type(self)
    spec = _class_spec(cls)
    for name in spec.children:
      if name not in spec.optional and getattr(self, name) is None:
        raise TypeError(
            f'{cls.__name__}.{name} is required, got None. Annotate it with'
            ' `| None` to make it optional.'
        )
    arrays = {
        name: value
        for name in spec.arrays
        if (value := getattr(self, name)) is not None
    }
    nested = {
        name: value
        for name in spec.nested
        if (value := getattr(self, name)) is not None
    }
    for name, value in nested.items():
      if not isinstance(value, BatchedDataclass):
        raise TypeError(
            f'{cls.__name__}.{name} must be a {spec.nested[name].__name__},'
            f' got {type(value).__name__}.'
        )
    xnp = _common_xnp(
        cls,
        {name: _get_xnp(cls, name, value) for name, value in arrays.items()}
        | {name: value.xnp for name, value in nested.items()},
    )
    device = None
    if enp.lazy.is_torch_xnp(xnp):
      all_arrays = dict(arrays)
      for name, value in nested.items():
        for key, array in _flat_arrays(value).items():
          all_arrays[f'{name}.{key}'] = array
      device = _torch_device(cls.__name__, all_arrays)

    batch_shapes = {}
    named_dims = {}
    for name, value in arrays.items():
      field = spec.arrays[name]
      if _needs_cast(field, value, xnp):
        value = _convert_array_like(value, xnp)
        if enp.lazy.is_array(value) and isinstance(
            field.dtype, enp.dtypes.NpDType
        ):
          # Not `field.dtype.asarray`, which detaches PyTorch tensors from the
          # autograd graph (see `_cast`). The other casts (between integers
          # and floating-point numbers) cannot keep gradients anyway.
          value = _cast(value, field.dtype.np_dtype)
        else:
          value = field.dtype.asarray(value, xnp=xnp)
        if isinstance(
            field.dtype, (enp.dtypes.AnyFloat, enp.dtypes.AnyInt)
        ) and _needs_cast(field, value, xnp):
          # E.g. generic void (raw bytes) arrays, which `etils` does not cast.
          raise TypeError(
              f'Invalid dtype {_dtype(value)} for'
              f' {cls.__name__}.{name}: expected {field.dtype}.'
          )
        value = _to_device(value, device)
      self._check_inner_shape(name, tuple(value.shape), field, named_dims)
      batch_shapes[name] = _batch_shape(value, field.inner_ndim)
      arrays[name] = value
    for name, value in nested.items():
      converted = value.as_xnp(xnp)
      if device is not None and converted is not value:  # Converted on the CPU.
        converted = converted.map_arrays(lambda x: _to_device(x, device))
      nested[name] = value = converted
      if _has_arrays(value):  # Else, no batch shape.
        batch_shapes[name] = value.shape

    shape = _broadcast_batch_shapes(cls, batch_shapes, broadcast=cls._broadcast)
    for name, value in arrays.items():
      if batch_shapes[name] != shape:
        inner_ndim = spec.arrays[name].inner_ndim
        value = xnp.broadcast_to(value, shape + _inner_shape(value, inner_ndim))
      object.__setattr__(self, name, value)
    for name, value in nested.items():
      if name in batch_shapes and batch_shapes[name] != shape:
        value = value.broadcast_to(shape)
      object.__setattr__(self, name, value)

  def _check_inner_shape(
      self,
      name: str,
      shape: tuple[int | None, ...],
      field: _ArrayField,
      named_dims: dict[str, int],
  ) -> None:
    """Checks the inner dimensions of an array against its annotation."""

    def error(details: str = '') -> ValueError:
      expected = ' '.join(map(str, (_BATCH_TOKEN, *field.inner_dims)))
      return ValueError(
          f'Invalid shape {shape} for {type(self).__name__}.{name}: expected'
          f' ({expected}){details}.'
      )

    if len(shape) < field.inner_ndim:
      raise error()
    for dim, size in zip(
        field.inner_dims, shape[len(shape) - field.inner_ndim :]
    ):
      if size is None:  # Unknown (TensorFlow) dimension.
        continue
      if isinstance(dim, int):
        if size != dim:
          raise error()
      elif size != (named_size := named_dims.setdefault(dim, size)):
        raise error(f', with {dim}={named_size} as in the previous fields')

  # ====== Array-like properties ======

  @property
  def shape(self) -> Shape:
    """The batch shape, shared by all the array and nested fields.

    It is `()` without any non-None array, see `_first_child`.
    """
    spec = _class_spec(type(self))
    name, value = self._first_child()
    if name in spec.arrays:
      return _batch_shape(value, spec.arrays[name].inner_ndim)
    return value.shape if name else ()

  @property
  def ndim(self) -> int:
    """The number of batch dimensions."""
    return len(self.shape)

  @property
  def size(self) -> int:
    """The number of elements in the batch."""
    return math.prod(self.shape)

  @property
  def xnp(self) -> NpModule:
    """The NumPy-like module of the arrays (np, jnp, tnp or torch)."""
    name, value = self._first_child()
    if name in _class_spec(type(self)).nested:
      return value.xnp
    return enp.lazy.get_xnp(value) if name else np

  def _first_child(self) -> tuple[str | None, Any]:
    """Returns the name and value of the first child with a batch shape.

    That is the first non-None array field, or nested field with arrays. Values
    without any (e.g. with only optional fields set to None) have no batch
    shape: their `shape` is `()`. When nested, they adapt to the batch shape of
    their parent: they are ignored when computing it (also without
    broadcasting), and are left unchanged by indexing, reshaping, broadcasting
    and the other batch operations.

    Returns:
      The name and value of the first child with a batch shape, or
      `(None, None)` if there is none.
    """
    spec = _class_spec(type(self))
    for name in spec.children:
      value = getattr(self, name)
      if value is not None and (name in spec.arrays or _has_arrays(value)):
        return name, value
    return None, None

  def __len__(self) -> int:
    shape = self.shape
    if not shape:
      raise TypeError(f'len() of unbatched {type(self).__name__}.')
    return shape[0]

  def __bool__(self) -> bool:
    # Like all Python objects, instances are truthy (rather than relying on
    # `__len__`, which fails without batch dimensions), except for empty batches
    # which are ambiguous.
    if self.shape and not self.shape[0]:
      raise ValueError(
          f'The truth value of an empty {type(self).__name__} is ambiguous.'
          ' Use `len(x)` or `x is not None`.'
      )
    return True

  def __repr__(self) -> str:
    parts = [f'shape={self.shape}']
    for field in dataclasses.fields(self):  # pyrefly: ignore[bad-argument-type]
      if field.repr:
        value = getattr(self, field.name)
        if enp.lazy.is_array(value):
          value = _array_spec(value)
        parts.append(f'{field.name}={value!r}')
    return f'{type(self).__name__}({", ".join(parts)})'

  def _replace_unchecked(self, changes: Mapping[str, Any]) -> Self:
    """Returns a copy with the given fields replaced, without validation."""
    new = object.__new__(type(self))
    for field in dataclasses.fields(self):  # pyrefly: ignore[bad-argument-type]
      value = (
          changes[field.name]
          if field.name in changes
          else getattr(self, field.name)
      )
      object.__setattr__(new, field.name, value)
    return new

  def _map(
      self,
      array_fn: Callable[[Array, _ArrayField], Array],
      nested_fn: Callable[[BatchedDataclass], BatchedDataclass],
  ) -> Self:
    """Applies `array_fn(array, field)` and `nested_fn(nested)` to fields.

    The result is not validated, so the functions must keep the batch shapes of
    all the fields consistent.

    Args:
      array_fn: The function applied to the (non-None) arrays, also given the
        specification of their field.
      nested_fn: The function applied to the nested fields with arrays. The
        nested fields without arrays have no batch shape and nothing to
        transform, so they are kept unchanged (see `_first_child`).

    Returns:
      A copy with the transformed fields.
    """
    spec = _class_spec(type(self))
    changes = {}
    for name, field in spec.arrays.items():
      value = getattr(self, name)
      if value is not None:
        changes[name] = array_fn(value, field)
    for name in spec.nested:
      value = getattr(self, name)
      if value is not None and _has_arrays(value):
        changes[name] = nested_fn(value)
    return self._replace_unchecked(changes)

  def broadcast_to(self, shape: int | Sequence[int]) -> Self:
    """Broadcasts the batch dimensions to `shape`, like `np.broadcast_to`."""
    shape = _as_shape(shape)
    return self._map(
        lambda x, f: enp.lazy.get_xnp(x).broadcast_to(
            x, shape + _inner_shape(x, f.inner_ndim)
        ),
        lambda v: v.broadcast_to(shape),
    )

  def replace(self, **changes: Any) -> Self:
    """Returns a copy with the given fields replaced (validated, broadcast)."""
    return dataclasses.replace(self, **changes)  # pyrefly: ignore[bad-specialization]

  # ====== Backends ======

  def as_xnp(self, xnp: NpModule) -> Self:
    """Converts the arrays to the given NumPy-like module."""
    if self.xnp is xnp:
      return self
    return self._map(lambda x, _: _convert(x, xnp), lambda v: v.as_xnp(xnp))

  def as_np(self) -> Self:
    """Converts the arrays to `np.ndarray`."""
    return self.as_xnp(np)

  def as_jax(self) -> Self:
    """Converts the arrays to `jax.Array`."""
    return self.as_xnp(enp.lazy.jnp)

  def as_tf(self) -> Self:
    """Converts the arrays to `tf.Tensor`."""
    return self.as_xnp(enp.lazy.tnp)

  def as_torch(self) -> Self:
    """Converts the arrays to `torch.Tensor`."""
    return self.as_xnp(enp.lazy.torch)


def _has_arrays(value: BatchedDataclass) -> bool:
  """Returns whether `value` has a batch shape, see `_first_child`."""
  return value._first_child()[0] is not None  # pylint: disable=protected-access


def _convert_array_like(value: Any, xnp: NpModule) -> Any:
  """Converts arrays of another backend (e.g. NumPy) to `xnp`, if needed."""
  if enp.lazy.is_array(value) and enp.lazy.get_xnp(value) is not xnp:
    return _convert(value, xnp)
  return value


def _flat_arrays(value: BatchedDataclass) -> dict[str, Array]:
  """Returns all the (non-None) arrays, with flat '.'-joined names."""
  spec = _class_spec(type(value))
  arrays = {}
  for name in spec.children:
    field_value = getattr(value, name)
    if field_value is None:
      continue
    if name in spec.nested:
      for key, array in _flat_arrays(field_value).items():
        arrays[f'{name}.{key}'] = array
    else:
      arrays[name] = field_value
  return arrays
