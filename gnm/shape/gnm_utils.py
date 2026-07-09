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

"""Utility functions for manipulating GNM identity and expression basis."""

from collections.abc import Mapping, Sequence
import enum
from typing import Any

from gnm.shape import gnm_numpy
import numpy as np
import numpy.typing as npt
from scipy import linalg


class BasisType(enum.StrEnum):
  IDENTITY = 'identity'
  EXPRESSION = 'expression'


BASIS_DIM_ATTRIBUTE_MAP = {
    BasisType.IDENTITY: 'identity_dim',
    BasisType.EXPRESSION: 'expression_dim',
}

_BASIS_NAMES_ATTRIBUTE_MAP = {
    BasisType.IDENTITY: 'identity_names',
    BasisType.EXPRESSION: 'expression_names',
}

_BASIS_BASIS_ATTRIBUTE_MAP = {
    BasisType.IDENTITY: 'vertex_identity_basis',
    BasisType.EXPRESSION: 'expression_basis',
}


# Used to compute the expression components stddevs.
_EXPRESSION_REGION_VERTEX_GROUP_MAP = {
    'left_eye': ['expression_basis_left_eye'],
    'right_eye': ['expression_basis_right_eye'],
    'mouth': ['expression_basis_mouth_nose_ears', 'lower_teeth', 'tongue'],
    'tongue': ['tongue'],
    'eyeballs': ['eyeball_interior'],
}

# Used to compute the identity components stddevs.
_IDENTITY_REGION_VERTEX_GROUP_MAP = {}

_BASIS_REGION_VERTEX_GROUP_MAP = {
    BasisType.IDENTITY: _IDENTITY_REGION_VERTEX_GROUP_MAP,
    BasisType.EXPRESSION: _EXPRESSION_REGION_VERTEX_GROUP_MAP,
}

_TONGUE_MEAN_NAME = 'tongue_mean'


# gnm_utils supports only the following variants.
_SUPPORTED_HEAD_VARIANTS = frozenset([
    gnm_numpy.GNMVariant.HEAD,
])

_SUPPORTED_GNM_VARIANTS = frozenset(_SUPPORTED_HEAD_VARIANTS)


def _get_major_version(version_str: str) -> int:
  """Extracts the major version from a version string."""
  try:
    v = version_str.lower().lstrip('v')
    return int(v.split('.')[0])
  except ValueError:
    return 1


def validate_gnm(gnm: gnm_numpy.GNM) -> None:
  """Validates that the GNM model is supported by gnm_utils."""
  variant = getattr(gnm, 'variant', None)
  if variant not in _SUPPORTED_GNM_VARIANTS:
    raise ValueError(f"GNM variant '{variant}' is not supported by gnm_utils.")


def _validate_conversion(
    from_gnm: gnm_numpy.GNM,
    to_gnm: gnm_numpy.GNM,
    basis_type: BasisType,
) -> None:
  """Validates that the GNM models can be converted between each other."""
  del basis_type

  validate_gnm(from_gnm)
  validate_gnm(to_gnm)


def get_gnm_attribute(gnm: gnm_numpy.GNM, attribute: str) -> Any:
  """Returns the attribute from the GNM model."""
  if (data := getattr(gnm, attribute, None)) is None:
    raise ValueError(
        f'No attribute {attribute} in GNM model version {gnm.version} and'
        f' variant {gnm.variant}.'
    )
  return data


def _get_corresponding_indices(
    basis_type: BasisType,
    from_gnm: gnm_numpy.GNM,
    to_gnm: gnm_numpy.GNM,
) -> tuple[np.ndarray, np.ndarray]:
  """Get indices mapping common names subset from `from_basis` to `to_basis`."""
  attribute_name = _BASIS_NAMES_ATTRIBUTE_MAP[basis_type]
  from_names = np.array(get_gnm_attribute(from_gnm, attribute_name))
  to_names = np.array(get_gnm_attribute(to_gnm, attribute_name))

  # Check uniqueness of names.
  from_names_unique = np.unique(from_names).size == from_names.size
  to_names_unique = np.unique(to_names).size == to_names.size
  if not (from_names_unique and to_names_unique):
    raise ValueError('Names must be unique.')

  from_indices, to_indices = np.where(from_names[:, None] == to_names[None])
  return from_indices.astype(np.int32), to_indices.astype(np.int32)


def components_names(
    basis_type: BasisType, gnm: gnm_numpy.GNM
) -> Sequence[str]:
  """Returns the names of the basis components."""
  attribute_name = _BASIS_NAMES_ATTRIBUTE_MAP[basis_type]
  return get_gnm_attribute(gnm, attribute_name)


def region_names_from_components_names(names: Sequence[str]) -> Sequence[str]:
  """Returns the names of the basis regions keeping the components order."""
  regions_names = ['_'.join(n.split('_')[:-1]) for n in names]
  _, indices = np.unique(regions_names, return_index=True)
  return [regions_names[i] for i in np.sort(indices)]


def region_to_component_indices_map(
    basis_type: BasisType, gnm: gnm_numpy.GNM
) -> Mapping[str, np.ndarray]:
  """Return a map from basis region name to component indices."""
  names = components_names(basis_type, gnm)
  region_names = region_names_from_components_names(names)

  region_indices_map = {}
  for r in region_names:
    indices = [index for index, name in enumerate(names) if name.startswith(r)]
    region_indices_map[r] = np.array(indices, dtype=np.int32)
  return region_indices_map


def convert_coefficients(
    coefficients: npt.NDArray[np.floating],
    *,
    basis_type: BasisType,
    from_gnm: gnm_numpy.GNM,
    to_gnm: gnm_numpy.GNM,
) -> npt.NDArray[np.floating]:
  """Convert coefficients from one GNM model to another.

  Args:
    coefficients: Coefficients corresponding to the basis in `from_gnm`, shape
      ([A1, ..., An], C_from).
    basis_type: Type of the basis, either identity or expression.
    from_gnm: GNM model instance to convert from.
    to_gnm: GNM model instance to convert to.

  Returns:
    Coefficients corresponding to the basis in `to_gnm`, shape
    ([A1, ..., An], C_to).
  """
  if from_gnm is None or to_gnm is None:
    raise ValueError('Both from_gnm and to_gnm must be provided.')

  _validate_conversion(from_gnm, to_gnm, basis_type)

  batch_shape = coefficients.shape[:-1]

  attribute_name = BASIS_DIM_ATTRIBUTE_MAP[basis_type]
  from_dim = get_gnm_attribute(from_gnm, attribute_name)
  to_dim = get_gnm_attribute(to_gnm, attribute_name)

  if coefficients.shape[-1] != from_dim:
    raise ValueError(
        f'Dimension mismatch: {coefficients.shape[-1]} vs {from_dim}.'
    )

  to_indices, from_indices = _get_corresponding_indices(
      basis_type, from_gnm=to_gnm, to_gnm=from_gnm
  )

  # Convert.
  converted = np.zeros(batch_shape + (to_dim,), dtype=coefficients.dtype)
  converted[..., to_indices] = coefficients[..., from_indices]
  return converted


def _coefficients_to_regions(
    coefficients: np.ndarray, basis_type: BasisType, gnm: gnm_numpy.GNM
) -> dict[str, np.ndarray]:
  """Splits the coefficients ([A1, ..., An], C) to face regions."""
  validate_gnm(gnm)
  attribute_name = BASIS_DIM_ATTRIBUTE_MAP[basis_type]
  coeffs_dim = get_gnm_attribute(gnm, attribute_name)
  if coefficients.shape[-1] != coeffs_dim:
    raise ValueError(
        f'Unexpected coefficients dimension of {coefficients.shape[-1]}.'
        f' Expected {coeffs_dim} for the basis type {basis_type} and GNM model'
        f' version {gnm.version} and variant {gnm.variant}.'
    )

  region_indices = region_to_component_indices_map(basis_type, gnm)
  regions = {
      r: coefficients[..., indices] for r, indices in region_indices.items()
  }
  for region_name, region_coefficients in regions.items():
    assert region_coefficients.base is None or (
        region_coefficients.base is not coefficients
        and region_coefficients.base is not coefficients.base
    ), (
        f'Region {region_name} is a view of the original array. This is most'
        ' likely a bug and may cause unexpected behavior.'
    )
  return regions


def _regions_to_coefficients(
    regions: Mapping[str, np.ndarray],
    basis_type: BasisType,
    gnm: gnm_numpy.GNM,
) -> np.ndarray:
  """Combines face regions to a single basis array ([A1, ..., An], C])."""
  validate_gnm(gnm)
  attribute_name = BASIS_DIM_ATTRIBUTE_MAP[basis_type]
  coeffs_dim = get_gnm_attribute(gnm, attribute_name)

  if not regions:
    return np.zeros((coeffs_dim,), dtype=np.float32)

  region_indices = region_to_component_indices_map(basis_type, gnm)

  tmp_region_name = list(regions.keys())[0]
  batch_shape = regions[tmp_region_name].shape[:-1]
  dtype = regions[tmp_region_name].dtype

  coeffs = np.zeros(batch_shape + (coeffs_dim,), dtype=dtype)
  for region_name, values in regions.items():
    if region_name not in region_indices:
      gnm_type_name = getattr(gnm, 'model_type', 'unknown')
      raise ValueError(f'No region {region_name} in model {gnm_type_name}.')
    else:
      coeffs[..., region_indices[region_name]] = values

  return coeffs


def _region_components(
    basis_type: BasisType, gnm: gnm_numpy.GNM
) -> dict[str, np.ndarray]:
  """Return per-region basis components, each region shape (Ri, V, 3)."""
  validate_gnm(gnm)
  attribute_name = _BASIS_BASIS_ATTRIBUTE_MAP[basis_type]
  basis = get_gnm_attribute(gnm, attribute_name)
  region_indices = region_to_component_indices_map(basis_type, gnm)
  return {r: basis[indices] for r, indices in region_indices.items()}


def compute_scaling_for_basis(
    vertex_basis: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
  """Computes the scaling matrix to make basis equivalent to vertex losses.

  Args:
    vertex_basis: The basis to compute the scaling for, shape (D, V, 3).

  Returns:
    A scaling matrix to apply to differences between basis components,
    shape (D, D).
  """
  basis_dim = vertex_basis.shape[0]
  vertex_basis = vertex_basis.reshape(basis_dim, -1).astype(np.float64)

  basis_dot_transpose = vertex_basis @ vertex_basis.T
  return linalg.sqrtm(basis_dot_transpose).astype(np.float32)


def scaling_for_expression_loss(
    gnm: gnm_numpy.GNM,
) -> npt.NDArray[np.floating]:
  """Scaling matrix to make expression losses equivalent to vertex losses.

  The loss between expression components is: || e1 - e2 ||^2.
  The loss between vertices is: || v1 - v2 ||^2.
  || v1 - v2 ||^2 = || (template + B_v @ e1 - (template + B_v @ c2) ||^2 =
  || B_v @ (c1 - c2) ||^2 =
  (c1 - c2)^T @ B_v^T @ B_v @ (c1 - c2) =
  (c1 - c2)^T @ M @ (c1 - c2), M = B_v^T @ B_v (1)

  This if we want to make the expression loss equivalent to the vertex loss,
  we need to scale the expression loss by: M = matrix_sqrt(B_v^T @ B_v).


  Args:
    gnm: The GNM model instance to compute the scaling for.

  Returns:
    A scaling matrix to apply to differences between expression components,
    shape (E, E).
  """
  validate_gnm(gnm)
  return compute_scaling_for_basis(
      get_gnm_attribute(gnm, _BASIS_BASIS_ATTRIBUTE_MAP[BasisType.EXPRESSION])
  )


def scaling_for_identity_loss(
    gnm: gnm_numpy.GNM,
) -> npt.NDArray[np.floating]:
  """Scaling matrix to make identity losses equivalent to vertex losses."""
  validate_gnm(gnm)
  return compute_scaling_for_basis(
      get_gnm_attribute(gnm, _BASIS_BASIS_ATTRIBUTE_MAP[BasisType.IDENTITY])
  )


def identity_to_regions(
    identity: np.ndarray, gnm: gnm_numpy.GNM
) -> dict[str, np.ndarray]:
  """Splits the identity ([A1, ..., An], I]) to face regions."""
  return _coefficients_to_regions(
      identity, basis_type=BasisType.IDENTITY, gnm=gnm
  )


def expression_to_regions(
    expression: np.ndarray, gnm: gnm_numpy.GNM
) -> dict[str, np.ndarray]:
  """Splits the expression ([A1, ..., An], E]) to face regions."""
  return _coefficients_to_regions(
      expression, basis_type=BasisType.EXPRESSION, gnm=gnm
  )


def expression_regions_indices(
    gnm: gnm_numpy.GNM,
) -> dict[str, np.ndarray]:
  """Returns mapping from region name to expression component indices."""
  validate_gnm(gnm)
  return dict(region_to_component_indices_map(BasisType.EXPRESSION, gnm))


def identity_regions_indices(
    gnm: gnm_numpy.GNM,
) -> dict[str, np.ndarray]:
  """Returns mapping from region name to identity component indices."""
  validate_gnm(gnm)
  return dict(region_to_component_indices_map(BasisType.IDENTITY, gnm))


def identity_dim(gnm: gnm_numpy.GNM) -> int:
  """Returns the identity dimension of the GNM model."""
  validate_gnm(gnm)
  return get_gnm_attribute(gnm, 'identity_dim')


def expression_dim(gnm: gnm_numpy.GNM) -> int:
  """Returns the expression dimension of the GNM model."""
  validate_gnm(gnm)
  return get_gnm_attribute(gnm, 'expression_dim')


def expression_regions_dims(gnm: gnm_numpy.GNM) -> dict[str, int]:
  """Returns mapping from region name to expression component dimension."""
  validate_gnm(gnm)
  region_indices = region_to_component_indices_map(BasisType.EXPRESSION, gnm)
  return {k: v.size for k, v in region_indices.items()}


def identity_regions_dims(gnm: gnm_numpy.GNM) -> dict[str, int]:
  """Returns mapping from region name to identity component dimension."""
  validate_gnm(gnm)
  region_indices = region_to_component_indices_map(BasisType.IDENTITY, gnm)
  return {k: v.size for k, v in region_indices.items()}


def joint_rotations_to_regions(
    joint_rotations: np.ndarray, gnm: gnm_numpy.GNM
) -> dict[str, np.ndarray]:
  """Splits the joint rotations ([A1, ..., An], J, 3]) to invidual regions."""
  validate_gnm(gnm)
  joint_names: list[str] = get_gnm_attribute(gnm, 'joint_names')
  if len(joint_names) != joint_rotations.shape[-2]:
    raise ValueError(
        f'Number of joints in the GNM model ({len(joint_names)}) does not'
        ' match the number of joints in the input rotations'
        f' ({joint_rotations.shape[-2]}).'
    )
  return {
      joint_name: np.copy(joint_rotations[..., joint_index, :])
      for joint_index, joint_name in enumerate(joint_names)
  }


def regions_to_identity(
    regions: Mapping[str, np.ndarray], gnm: gnm_numpy.GNM
) -> np.ndarray:
  """Combines face regions to single identity array ([A1, ..., An], I])."""
  return _regions_to_coefficients(regions, BasisType.IDENTITY, gnm)


def regions_to_expression(
    regions: Mapping[str, np.ndarray], gnm: gnm_numpy.GNM
) -> np.ndarray:
  """Combines face regions to single expression array ([A1, ..., An], E])."""
  return _regions_to_coefficients(regions, BasisType.EXPRESSION, gnm)


def regions_to_joint_rotations(
    regions: Mapping[str, np.ndarray], gnm: gnm_numpy.GNM
) -> np.ndarray:
  """Combines joint rotations to single rotations array.

  Returns array of shape ([A1, ..., An], J, 3]).
  """
  validate_gnm(gnm)
  joint_names: list[str] = get_gnm_attribute(gnm, 'joint_names')
  if not regions:
    raise ValueError('No regions provided.')

  tmp_region_name = list(regions.keys())[0]
  batch_shape = regions[tmp_region_name].shape[:-1]
  dtype = regions[tmp_region_name].dtype

  missing_joint_value = np.zeros((*batch_shape, 3), dtype=dtype)
  return np.stack(
      [
          regions.get(
              joint_name,
              missing_joint_value,
          )
          for joint_name in joint_names
      ],
      axis=-2,
  )


def region_identity_components(
    gnm: gnm_numpy.GNM,
) -> dict[str, np.ndarray]:
  """Return per-region identity components, each region shape (Ri, V, 3)."""
  return _region_components(BasisType.IDENTITY, gnm)


def region_expression_components(
    gnm: gnm_numpy.GNM,
) -> dict[str, np.ndarray]:
  """Return per-region expression components, each region shape (Ri, V, 3)."""
  return _region_components(BasisType.EXPRESSION, gnm)


def _sigmas(basis_type: BasisType, gnm: gnm_numpy.GNM) -> np.ndarray:
  """Return the standard deviations of basis components, shape (C, )."""
  validate_gnm(gnm)

  if basis_type == BasisType.IDENTITY:
    raise NotImplementedError('Identity sigmas are not implemented yet.')

  region_group = _BASIS_REGION_VERTEX_GROUP_MAP[basis_type]
  regions = {}
  for name, components in _region_components(basis_type, gnm).items():
    vertex_indices = gnm.vertex_group_indices(*region_group[name])
    regions[name] = np.linalg.norm(components[:, vertex_indices], axis=(1, 2))
  region_sigmas = _regions_to_coefficients(regions, basis_type, gnm)

  # Treat tongue mean in expressions individually, it is not a real component.
  if basis_type == BasisType.EXPRESSION:
    expression_names = gnm.expression_names
    if _TONGUE_MEAN_NAME in expression_names:
      # Convert to list to use .index()
      expression_names_list = list(expression_names)
      region_sigmas[expression_names_list.index(_TONGUE_MEAN_NAME)] = 1.0

  if np.any(np.isclose(region_sigmas, 0.0, atol=1e-8)):
    raise RuntimeError('Expression components have zero standard deviation.')

  return region_sigmas


def expression_sigmas(gnm: gnm_numpy.GNM) -> np.ndarray:
  """Return the standard deviations of expression components, shape (E, )."""
  return _sigmas(BasisType.EXPRESSION, gnm)


def _normalize_coefficients(
    coefficients: np.ndarray, basis_type: BasisType, gnm: gnm_numpy.GNM
) -> np.ndarray:
  """Normalize coefficients ([A1, ..., An], C) by corresponding sigmas."""
  s = _sigmas(basis_type, gnm)
  return coefficients / s


def _denormalize_coefficients(
    coefficients: np.ndarray, basis_type: BasisType, gnm: gnm_numpy.GNM
) -> np.ndarray:
  """Denormalize coefficients ([A1, ..., An], C) by corresponding sigmas."""
  s = _sigmas(basis_type, gnm)
  return coefficients * s


def normalize_expression(
    expression: np.ndarray, gnm: gnm_numpy.GNM
) -> np.ndarray:
  """Normalize expression coefficients ([A1, ..., An], E)."""
  return _normalize_coefficients(expression, BasisType.EXPRESSION, gnm)


def denormalize_expression(
    expression: np.ndarray, gnm: gnm_numpy.GNM
) -> np.ndarray:
  """Denormalize expression coefficients ([A1, ..., An], E)."""
  return _denormalize_coefficients(expression, BasisType.EXPRESSION, gnm)


def identity_region_names(gnm: gnm_numpy.GNM) -> Sequence[str]:
  """Gets identity basis region names ordered as they appear in the basis."""
  validate_gnm(gnm)
  names = components_names(BasisType.IDENTITY, gnm)
  return region_names_from_components_names(names)


def expression_region_names(gnm: gnm_numpy.GNM) -> Sequence[str]:
  """Gets expression basis region names ordered as they appear in the basis."""
  validate_gnm(gnm)
  names = components_names(BasisType.EXPRESSION, gnm)
  return region_names_from_components_names(names)
