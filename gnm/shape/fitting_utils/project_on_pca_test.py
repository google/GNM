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

"""Tests for project_on_pca."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
from gnm.shape import gnm_utils
from gnm.shape.data.versions import gnm_catalog
from gnm.shape.fitting_utils import project_on_pca
import numpy as np

_M_TO_MM = 1000.0


class ProjectOnPcaTest(parameterized.TestCase):

  gnms: dict[str, gnm_numpy.GNM]
  rng: np.random.Generator

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.rng = np.random.default_rng(0)

    cls.gnms = {}
    for version in gnm_catalog.MAINTAINED_MAJOR_VERSIONS:
      if (
          gnm_numpy.GNMVariant.HEAD
          in gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version]
      ):
        cls.gnms[version] = gnm_numpy.GNM.from_local(
            gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
            gnm_numpy.GNMVariant.HEAD,
        )

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      index=[0, 2, 4],
      value=[1.0],
  )
  def test_can_recover_identity(self, version: str, index: int, value: float):
    """Tests that the estimated identity is close to the ground truth."""
    gnm_np = self.gnms[version]
    identity = np.zeros(gnm_np.identity_dim)
    expression = np.zeros(gnm_np.expression_dim)
    identity[index] = value

    vertices = gnm_np.vertex_positions_bind_pose(identity, expression)

    result = project_on_pca.project_on_linear_vertex_basis(
        vertices,
        gnm_np.template_vertex_positions,
        gnm_np.vertex_identity_basis,
    )
    estimated_identity = result.coefficients[0]

    np.testing.assert_allclose(estimated_identity, identity, atol=1.0e-06)

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      index=[0, 2, 4],
      value=[1.0],
  )
  def test_can_recover_expression(self, version: str, index: int, value: float):
    """Tests that the estimated expression is close to the ground truth."""
    gnm_np = self.gnms[version]
    identity = np.zeros(gnm_np.identity_dim)
    expression = np.zeros(gnm_np.expression_dim)
    expression[index] = value

    vertices = gnm_np.vertex_positions_bind_pose(identity, expression)

    result = project_on_pca.project_on_linear_vertex_basis(
        vertices,
        gnm_np.template_vertex_positions,
        gnm_np.expression_basis,
    )
    estimated_expression = result.coefficients[0]

    np.testing.assert_allclose(estimated_expression, expression, atol=1.0e-06)

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      num_components_and_threshold=[(10, 2.0), (100, 1.0)],
  )
  def test_can_fit_identity_using_subset_of_basis(
      self, version, num_components_and_threshold
  ):
    """Tests that we can estimate identity using a subset of the basis."""
    if (
        gnm_numpy.GNMVariant.HEAD
        not in gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version]
    ):
      self.skipTest(f'Head variant not available in version {version}')

    num_components, threshold = num_components_and_threshold

    major_version = gnm_numpy.GNMMajorVersion(version.removeprefix('v'))
    gnm = gnm_numpy.GNM.from_local(major_version, gnm_numpy.GNMVariant.HEAD)

    identity = self.rng.uniform(low=-1.5, high=1.5, size=gnm.identity_dim)
    expression = np.zeros(gnm.expression_dim)

    vertices = gnm.vertex_positions_bind_pose(identity, expression)

    result = project_on_pca.project_on_linear_vertex_basis(
        vertices,
        gnm.template_vertex_positions,
        gnm.vertex_identity_basis,
        num_components=num_components,
    )

    # Check that the mesh error is within the threshold. We do not compare the
    # coefficients because the least-squares fit for the mesh with fewer
    # components is not expected to be the same as using all components.
    mean_error = np.linalg.norm(
        result.reconstruction - vertices, axis=-1
    ).mean()  # pyrefly: ignore[unsupported-operation]
    mesh_error = mean_error * _M_TO_MM
    self.assertLess(mesh_error, threshold)

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_size_num_components_threshold=[
          (1, 10, 2.0),
          (2, 10, 2.0),
          (4, 10, 2.0),
          (1, 100, 1.0),
          (2, 100, 1.0),
          (4, 100, 1.0),
      ],
  )
  def test_can_recover_identity_with_object(
      self, version, batch_size_num_components_threshold
  ):
    """Tests that the projection object can estimate identity ."""
    if (
        gnm_numpy.GNMVariant.HEAD
        not in gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version]
    ):
      self.skipTest(f'Head variant not available in version {version}')

    batch_size, num_components, threshold = batch_size_num_components_threshold

    major_version = gnm_numpy.GNMMajorVersion(version.removeprefix('v'))
    gnm = gnm_numpy.GNM.from_local(major_version, gnm_numpy.GNMVariant.HEAD)

    identity = self.rng.uniform(
        low=-1.5, high=1.5, size=(batch_size, gnm.identity_dim)
    )
    expression = np.zeros([batch_size, gnm.expression_dim])

    vertices = []
    for i in range(batch_size):
      vertices.append(
          gnm.vertex_positions_bind_pose(identity[i], expression[i])
      )
    vertices = np.stack(vertices, axis=0)

    pca_basis_projection = project_on_pca.PCABasisProjection(
        gnm.template_vertex_positions,
        gnm.vertex_identity_basis,
        num_components=num_components,
    )

    result = pca_basis_projection(vertices)

    # Check that the mesh error is within the threshold. We do not compare the
    # coefficients because the least-squares fit for the mesh with fewer
    # components is not expected to be the same as using all components.
    mesh_error = (
        np.linalg.norm(result.reconstruction - vertices, axis=-1).mean()
        * _M_TO_MM
    )
    self.assertLess(mesh_error, threshold)

  @parameterized.product(
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
      batch_size_num_components_threshold=[
          (1, 10, 2.0),
          (2, 10, 2.0),
          (4, 10, 2.0),
          (1, 100, 1.0),
          (2, 100, 1.0),
          (4, 100, 1.0),
      ],
  )
  def test_can_recover_expression_with_object(
      self, version, batch_size_num_components_threshold
  ):
    """Tests that the projection object can estimate expression."""
    if (
        gnm_numpy.GNMVariant.HEAD
        not in gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version]
    ):
      self.skipTest(f'Head variant not available in version {version}')

    batch_size, num_components, threshold = batch_size_num_components_threshold
    major_version = gnm_numpy.GNMMajorVersion(version.removeprefix('v'))
    gnm = gnm_numpy.GNM.from_local(major_version, gnm_numpy.GNMVariant.HEAD)

    expected_regions = [
        'left_eye_region',
        'right_eye_region',
        'lower_face_region',
    ]
    region_components = gnm_utils.region_expression_components(gnm)
    self.assertContainsSubset(
        expected_regions,
        region_components.keys(),
        msg='Variant {variant} does not have all expected expression regions.',
    )

    # Take the first num_components / 3 components from each region.
    n_take = math.ceil(num_components / 3)
    left_eye_take = region_components['left_eye_region'][:n_take]
    right_eye_take = region_components['right_eye_region'][:n_take]
    mouth_take = region_components['lower_face_region'][:n_take]
    expression_basis_subset = np.concatenate(
        [left_eye_take, right_eye_take, mouth_take], axis=0
    )[:num_components]

    # Generate expression, and zero out unwanted expression components.
    expression = self.rng.uniform(
        low=-1.5, high=1.5, size=(batch_size, gnm.expression_dim)
    )
    regions = gnm_utils.expression_to_regions(expression, gnm)
    regions = {k: v for k, v in regions.items() if k in expected_regions}
    expression = gnm_utils.regions_to_expression(regions, gnm)

    # Generate zero identity.
    identity = np.zeros([batch_size, gnm.identity_dim])

    vertices = []
    for i in range(batch_size):
      vertices.append(
          gnm.vertex_positions_bind_pose(identity[i], expression[i])
      )
    vertices = np.stack(vertices, axis=0)

    pca_basis_projection = project_on_pca.PCABasisProjection(
        gnm.template_vertex_positions,
        expression_basis_subset,
        num_components=num_components,
    )

    result = pca_basis_projection(vertices)

    mesh_error = (
        np.linalg.norm(result.reconstruction - vertices, axis=-1).mean()
        * _M_TO_MM
    )
    self.assertLess(mesh_error, threshold)


if __name__ == '__main__':
  absltest.main()
