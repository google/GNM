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

"""Tests for GNM utility functions."""

# pylint: disable=protected-access

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
from gnm.shape import gnm_utils
from gnm.shape.data.versions import gnm_catalog
import numpy as np

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP

_SUPPORTED_VARIANTS = frozenset([
    gnm_numpy.GNMVariant.HEAD,
])

_HEAD_VARIANTS = gnm_utils._SUPPORTED_HEAD_VARIANTS
_BODY_VARIANTS = frozenset()
_UPPER_BODY_VARIANTS = frozenset()


class BaseGNMUtilsTest(parameterized.TestCase):

  gnms: dict[str, dict[str, gnm_numpy.GNM]]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Cache GNM instances to speed up tests
    cls.gnms = {}
    for version in _MAINTAINED_MAJOR_GNM_VERSIONS:
      cls.gnms[version] = {}
      for variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]:
        cls.gnms[version][variant] = gnm_numpy.GNM.from_local(
            gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
            gnm_numpy.GNMVariant(variant),
        )


class GNMUtilsConversionTest(BaseGNMUtilsTest):

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      from_variant=tuple(_SUPPORTED_VARIANTS),
      to_variant=tuple(_SUPPORTED_VARIANTS),
      basis_type=(
          gnm_utils.BasisType.IDENTITY,
          gnm_utils.BasisType.EXPRESSION,
      ),
  )
  def test_convert_coefficients(
      self, version, from_variant, to_variant, basis_type
  ):
    if from_variant not in self.gnms[version]:
      self.skipTest(
          f'from_variant {from_variant} not supported in GNM version {version}.'
      )
    if to_variant not in self.gnms[version]:
      self.skipTest(
          f'to_variant {to_variant} not supported in GNM version {version}.'
      )

    # For expression, all combinations are allowed, but not for identity.
    if basis_type == gnm_utils.BasisType.IDENTITY:
      from_head = from_variant in _HEAD_VARIANTS
      to_head = to_variant in _HEAD_VARIANTS
      from_body = from_variant in _BODY_VARIANTS
      to_body = to_variant in _BODY_VARIANTS
      from_upper = from_variant in _UPPER_BODY_VARIANTS
      to_upper = to_variant in _UPPER_BODY_VARIANTS

      if (from_head and to_body) or (to_head and from_body):
        self.skipTest('Cannot convert identity between head and body.')
      if (from_upper and to_body) or (to_upper and from_body):
        self.skipTest('Cannot convert identity between upper body and body.')

    from_gnm = self.gnms[version][from_variant]
    to_gnm = self.gnms[version][to_variant]

    from_dim = gnm_utils.get_gnm_attribute(
        from_gnm, gnm_utils.BASIS_DIM_ATTRIBUTE_MAP[basis_type]
    )
    from_coeffs = np.random.rand(from_dim).astype(np.float32)

    converted = gnm_utils.convert_coefficients(
        from_coeffs,
        basis_type=basis_type,
        from_gnm=from_gnm,
        to_gnm=to_gnm,
    )

    to_dim = gnm_utils.get_gnm_attribute(
        to_gnm, gnm_utils.BASIS_DIM_ATTRIBUTE_MAP[basis_type]
    )
    self.assertEqual(converted.shape, (to_dim,))

    reconstructed = gnm_utils.convert_coefficients(
        converted,
        basis_type=basis_type,
        from_gnm=to_gnm,
        to_gnm=from_gnm,
    )
    self.assertEqual(reconstructed.shape, (from_dim,))

    # We only expect the elements that exist in both GNM variants to be
    # preserved. If `to_gnm` does not contain some components of `from_gnm`
    # (despite having a larger overall dimension), those components are dropped
    # and their reconstructed value is 0.
    from_indices, _ = gnm_utils._get_corresponding_indices(
        basis_type, from_gnm, to_gnm
    )

    # The values mapped to the target basis should be fully reconstructed
    np.testing.assert_allclose(
        from_coeffs[from_indices], reconstructed[from_indices], atol=1e-5
    )

    # The values not mapped to the target basis are expected to be zeroed out
    unmapped_mask = np.ones(from_dim, dtype=bool)
    if from_indices.size > 0:
      unmapped_mask[from_indices] = False
    np.testing.assert_allclose(reconstructed[unmapped_mask], 0.0, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      from_variant=tuple(_SUPPORTED_VARIANTS),
      to_variant=tuple(_SUPPORTED_VARIANTS),
      basis_type=(
          gnm_utils.BasisType.IDENTITY,
          gnm_utils.BasisType.EXPRESSION,
      ),
  )
  def test_raises_on_dimension_mismatch(
      self, version, from_variant, to_variant, basis_type
  ):
    if from_variant not in self.gnms[version]:
      self.skipTest(f'from_variant {from_variant} not supported in {version}.')
    if to_variant not in self.gnms[version]:
      self.skipTest(f'to_variant {to_variant} not supported in {version}.')

    # For expression, all combinations are allowed, but not for identity.
    if basis_type == gnm_utils.BasisType.IDENTITY:
      from_head = from_variant in _HEAD_VARIANTS
      to_head = to_variant in _HEAD_VARIANTS
      from_body = from_variant in _BODY_VARIANTS
      to_body = to_variant in _BODY_VARIANTS
      from_upper = from_variant in _UPPER_BODY_VARIANTS
      to_upper = to_variant in _UPPER_BODY_VARIANTS

      if (from_head and to_body) or (to_head and from_body):
        self.skipTest('Cannot convert identity between head and body.')
      if (from_upper and to_body) or (to_upper and from_body):
        self.skipTest('Cannot convert identity between upper body and body.')

    from_gnm = self.gnms[version][from_variant]
    to_gnm = self.gnms[version][to_variant]

    from_dim = gnm_utils.get_gnm_attribute(
        from_gnm, gnm_utils.BASIS_DIM_ATTRIBUTE_MAP[basis_type]
    )

    from_coeffs = np.random.rand(from_dim - 1).astype(np.float32)

    with self.assertRaisesRegex(ValueError, 'Dimension mismatch'):
      gnm_utils.convert_coefficients(
          from_coeffs,
          basis_type=basis_type,
          from_gnm=from_gnm,
          to_gnm=to_gnm,
      )


class GNMUtilsMethodsTest(BaseGNMUtilsTest):

  def _random_identity(self, gnm, batch: tuple[int, ...] = ()) -> np.ndarray:
    shape = batch + (gnm_utils.identity_dim(gnm),)
    return np.random.uniform(-3.0, 3.0, shape).astype(np.float32)

  def _random_expression(self, gnm, batch: tuple[int, ...] = ()) -> np.ndarray:
    shape = batch + (gnm_utils.expression_dim(gnm),)
    return np.random.uniform(-3.0, 3.0, shape).astype(np.float32)

  def _random_rotations(self, gnm, batch: tuple[int, ...] = ()) -> np.ndarray:
    shape = batch + (len(gnm.joint_names), 3)
    return np.random.uniform(-3.0, 3.0, shape).astype(np.float32)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_identity_to_regions_and_back(self, version, variant, batch_shape):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    orig_identity = self._random_identity(gnm, batch_shape)
    regions = gnm_utils.identity_to_regions(orig_identity, gnm)
    identity_again = gnm_utils.regions_to_identity(regions, gnm)
    np.testing.assert_allclose(orig_identity, identity_again, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_expression_to_regions_and_back(self, version, variant, batch_shape):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    orig_expression = self._random_expression(gnm, batch_shape)
    regions = gnm_utils.expression_to_regions(orig_expression, gnm)
    expression_again = gnm_utils.regions_to_expression(regions, gnm)
    np.testing.assert_allclose(orig_expression, expression_again, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(
          gnm_numpy.GNMVariant.HEAD,
      ),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_normalize_denormalize_expression(
      self, version, variant, batch_shape
  ):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    coeffs = self._random_expression(gnm, batch_shape)
    norm = gnm_utils.normalize_expression(coeffs, gnm)
    denorm = gnm_utils.denormalize_expression(norm, gnm)
    np.testing.assert_allclose(coeffs, denorm, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_joint_rotations_cycle(self, version, variant, batch_shape):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')

    gnm = self.gnms[version][variant]

    rots = self._random_rotations(gnm, batch_shape)
    regions = gnm_utils.joint_rotations_to_regions(rots, gnm)
    self.assertLen(regions, len(gnm.joint_names))
    reconstructed = gnm_utils.regions_to_joint_rotations(regions, gnm)
    np.testing.assert_allclose(rots, reconstructed, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_identity_composed_from_individual_regions(
      self, version, variant, batch_shape
  ):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    orig_identity = self._random_identity(gnm, batch_shape)
    regions = gnm_utils.identity_to_regions(orig_identity, gnm)

    region_identities = []
    for region_name, region_coeffs in regions.items():
      region = {region_name: region_coeffs}
      identity = gnm_utils.regions_to_identity(region, gnm)
      region_identities.append(identity)

    reconstructed_identity = np.sum(region_identities, axis=0)
    np.testing.assert_allclose(orig_identity, reconstructed_identity, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_expression_composed_from_individual_regions(
      self, version, variant, batch_shape
  ):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    orig_expression = self._random_expression(gnm, batch_shape)
    regions = gnm_utils.expression_to_regions(orig_expression, gnm)

    region_expressions = []
    for region_name, region_coeffs in regions.items():
      region = {region_name: region_coeffs}
      expression = gnm_utils.regions_to_expression(region, gnm)
      region_expressions.append(expression)

    reconstructed_expression = np.sum(region_expressions, axis=0)
    np.testing.assert_allclose(
        orig_expression, reconstructed_expression, atol=1e-5
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_raises_on_wrong_coeffs_shape(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    coeffs = self._random_identity(gnm, (2, 3))
    dim = coeffs.shape[-1]
    with self.assertRaisesRegex(
        ValueError, 'Unexpected coefficients dimension of'
    ):
      gnm_utils.identity_to_regions(coeffs[..., : dim - 1], gnm)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_raises_on_wrong_region_name(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    bad_region_name = 'non-existent'
    bad_regions = {f'{bad_region_name}': np.zeros((10,))}
    with self.assertRaisesRegex(ValueError, f'No region {bad_region_name} in'):
      gnm_utils.regions_to_identity(bad_regions, gnm)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_region_identity_components_combine(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    regions = gnm_utils.region_identity_components(gnm)

    regions = {k: v.transpose(1, 2, 0) for k, v in regions.items()}
    combined = gnm_utils.regions_to_identity(regions, gnm)
    combined = combined.transpose(2, 0, 1)

    np.testing.assert_allclose(combined, gnm.vertex_identity_basis, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_region_expression_components_combine(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    regions = gnm_utils.region_expression_components(gnm)

    regions = {k: v.transpose(1, 2, 0) for k, v in regions.items()}
    combined = gnm_utils.regions_to_expression(regions, gnm)
    combined = combined.transpose(2, 0, 1)

    np.testing.assert_allclose(combined, gnm.expression_basis, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_expression_region_indices_matches_expression_regions(
      self, version, variant
  ):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    expression_vector = self._random_expression(gnm, ())
    expression_regions = gnm_utils.expression_to_regions(expression_vector, gnm)
    expression_region_indices = gnm_utils.expression_regions_indices(gnm)
    for region_name in expression_regions:
      with self.subTest(region_name):
        np.testing.assert_equal(
            expression_regions[region_name],
            expression_vector[expression_region_indices[region_name]],
        )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_identity_region_indices_matches_identity_regions(
      self, version, variant
  ):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    identity_vector = self._random_identity(gnm, ())
    identity_regions = gnm_utils.identity_to_regions(identity_vector, gnm)
    identity_region_indices = gnm_utils.identity_regions_indices(gnm)
    for region_name in identity_regions:
      with self.subTest(region_name):
        np.testing.assert_equal(
            identity_regions[region_name],
            identity_vector[identity_region_indices[region_name]],
        )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_identity_dim_is_correct(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    expected_dim = gnm.identity_dim
    self.assertEqual(gnm_utils.identity_dim(gnm), expected_dim)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_expression_dim_is_correct(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    expected_dim = gnm.expression_dim
    self.assertEqual(gnm_utils.expression_dim(gnm), expected_dim)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_expression_region_dims_are_correct(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    expected_dim = gnm.expression_dim
    regions = gnm_utils.expression_to_regions(np.zeros(expected_dim), gnm)

    region_dims = gnm_utils.expression_regions_dims(gnm)

    self.assertEqual(list(regions.keys()), list(region_dims.keys()))
    self.assertEqual(sum(region_dims.values()), expected_dim)
    for k in regions:
      self.assertEqual(regions[k].size, region_dims[k])

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_identity_region_dims_are_correct(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    expected_dim = gnm.identity_dim
    regions = gnm_utils.identity_to_regions(np.zeros(expected_dim), gnm)

    region_dims = gnm_utils.identity_regions_dims(gnm)

    self.assertEqual(list(regions.keys()), list(region_dims.keys()))
    self.assertEqual(sum(region_dims.values()), expected_dim)
    for k in regions:
      self.assertEqual(regions[k].size, region_dims[k])

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(
          gnm_numpy.GNMVariant.HEAD,
      ),
  )
  def test_expression_sigmas(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    region_name = 'left_eye_region'
    vertex_group_name = 'expression_basis_left_eye'

    # Get the region sigmas.
    sigmas = gnm_utils.expression_sigmas(gnm)
    region_sigmas = gnm_utils.expression_to_regions(sigmas, gnm)[region_name]

    # Manually compute the region sigmas.
    regions = gnm_utils.region_expression_components(gnm)
    components = regions[region_name]
    vertex_indices = gnm.vertex_group_indices(vertex_group_name)
    groups = components[:, vertex_indices]
    expected_region_sigmas = np.sqrt(np.sum(np.square(groups), axis=(1, 2)))

    # Compare.
    np.testing.assert_allclose(expected_region_sigmas, region_sigmas, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_identity_region_names(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    identity = self._random_identity(gnm, ())

    region_names = gnm_utils.identity_region_names(gnm)
    regions = gnm_utils.identity_to_regions(identity, gnm)

    recovered = np.concatenate([regions[name] for name in region_names], axis=0)
    np.testing.assert_allclose(identity, recovered)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_expression_region_names(self, version, variant):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    expression = self._random_expression(gnm, ())

    region_names = gnm_utils.expression_region_names(gnm)
    regions = gnm_utils.expression_to_regions(expression, gnm)

    recovered = np.concatenate([regions[name] for name in region_names], axis=0)
    np.testing.assert_allclose(expression, recovered)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
      missing_joint_index=list(range(4)),
  )
  def test_joint_rotations_to_regions_with_missing_regions(
      self, version, variant, batch_shape, missing_joint_index
  ):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    # if not hasattr(gnm, 'joint_names') or not gnm.joint_names:
    #   self.skipTest(f'variant {variant} does not have joint rotations.')
    if missing_joint_index >= len(gnm.joint_names):
      self.skipTest('missing_joint_index out of bounds.')

    orig_joint_rotations = self._random_rotations(gnm, batch_shape)
    regions = dict(
        gnm_utils.joint_rotations_to_regions(orig_joint_rotations, gnm)
    )
    missing_joint = gnm.joint_names[missing_joint_index]
    regions.pop(missing_joint)  # pyrefly: ignore[bad-argument-type]

    orig_joint_rotations[..., missing_joint_index, :] = 0.0
    joint_rotations_again = gnm_utils.regions_to_joint_rotations(regions, gnm)
    np.testing.assert_array_equal(orig_joint_rotations, joint_rotations_again)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(
          gnm_numpy.GNMVariant.HEAD,
      ),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_scaling_for_expression_loss(self, version, variant, batch_shape):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    expression1 = self._random_expression(gnm, batch_shape)
    expression2 = self._random_expression(gnm, batch_shape)

    vertices1 = gnm(expression=expression1)
    vertices2 = gnm(expression=expression2)

    scaling = gnm_utils.scaling_for_expression_loss(gnm)

    expression_loss = np.square(
        np.einsum('mk,...k->...m', scaling, expression1 - expression2)
    ).sum(axis=-1)
    vertex_loss = np.square(vertices1 - vertices2).sum(axis=(-1, -2))

    np.testing.assert_allclose(expression_loss, vertex_loss, atol=1e-3)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(
          gnm_numpy.GNMVariant.HEAD,
      ),
      batch_shape=(tuple(), (5,), (5, 4), (5, 4, 3)),
  )
  def test_scaling_for_identity_loss(self, version, variant, batch_shape):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    identity1 = self._random_identity(gnm, batch_shape)
    identity2 = self._random_identity(gnm, batch_shape)

    scaling = gnm_utils.scaling_for_identity_loss(gnm)

    identity_loss = np.square(
        np.einsum('mk,...k->...m', scaling, identity1 - identity2)
    ).sum(axis=-1)

    vertices1 = gnm(identity=identity1)
    vertices2 = gnm(identity=identity2)
    vertex_loss = np.square(vertices1 - vertices2).sum(axis=(-1, -2))

    np.testing.assert_allclose(identity_loss, vertex_loss, atol=1e-2)

  def test_compute_scaling_for_basis(self):
    np.random.seed(0)
    # create a random vertex basis (D, V, 3) -> D=4, V=10, 3
    vertex_basis = np.random.randn(4, 10, 3)

    scaling_matrix = gnm_utils.compute_scaling_for_basis(vertex_basis)

    # Calculate losses for random parameter differences.
    delta_params = np.random.randn(4)

    # 1. Vertex space loss.
    delta_vertices = np.einsum('d,dvi->vi', delta_params, vertex_basis)
    vertex_loss = np.sum(delta_vertices**2)

    # 2. Parameter space loss.
    scaled_delta_params = scaling_matrix @ delta_params
    param_loss = np.sum(scaled_delta_params**2)

    np.testing.assert_allclose(param_loss, vertex_loss, atol=1e-5)

    # Also verify the matrix square root directly.
    basis_dot_transpose = (
        vertex_basis.reshape(4, -1) @ vertex_basis.reshape(4, -1).T
    )
    np.testing.assert_allclose(
        scaling_matrix @ scaling_matrix, basis_dot_transpose, atol=1e-5
    )

    self.assertEqual(scaling_matrix.dtype, np.float32)


if __name__ == '__main__':
  absltest.main()
