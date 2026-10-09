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

"""Tests for GNM PyTorch implementation."""

from collections.abc import Sequence
import copy
import functools
import itertools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_numpy
from gnm.shape import gnm_pytorch
from gnm.shape import gnm_test_utils
from gnm.shape.data.versions import gnm_test_catalog
import numpy as np
import torch

_SUPPORTED_VARIANTS = frozenset([
    gnm_pytorch.GNMVariant.HEAD,
])

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_test_catalog.MAJOR_VERSION_TO_VARIANTS_MAP


_VARIANTS_BY_VERSION = {
    version: [
        variant
        for variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]
        if variant in _SUPPORTED_VARIANTS
    ]
    for version in _MAINTAINED_MAJOR_GNM_VERSIONS
}


def _load_gnm_pytorch(version: str, variant: str) -> gnm_pytorch.GNM:
  gnm = gnm_test_utils.load_gnm(gnm_pytorch.GNM, version, variant)
  if torch.cuda.is_available():
    gnm.cuda()
  return gnm


class GNMPytorchTest(parameterized.TestCase):
  gnms_np: gnm_test_utils.LazyGNMDict[gnm_numpy.GNM]
  gnms_pytorch: gnm_test_utils.LazyGNMDict[gnm_pytorch.GNM]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(0)
    torch.random.manual_seed(0)

    # Load the models on demand, keeping one of each in memory at a time.
    cls.gnms_np = gnm_test_utils.LazyGNMDict(
        functools.partial(gnm_test_utils.load_gnm, gnm_numpy.GNM),
        _VARIANTS_BY_VERSION,
    )
    cls.addClassCleanup(cls.gnms_np.clear)
    cls.gnms_pytorch = gnm_test_utils.LazyGNMDict(
        _load_gnm_pytorch, _VARIANTS_BY_VERSION
    )
    cls.addClassCleanup(cls.gnms_pytorch.clear)

  def _get_gnms(
      self, version: str, variant: Any
  ) -> tuple[gnm_pytorch.GNM, gnm_numpy.GNM]:
    """Returns the PyTorch and NumPy GNMs, or skips the test if unavailable."""
    return (
        self.gnms_pytorch.get_or_skip(version, variant),
        self.gnms_np.get_or_skip(version, variant),
    )

  def _get_default_kwargs(
      self, gnm_np: gnm_numpy.GNM, n_batch: int = 1, device: str = 'cpu'
  ) -> dict[str, torch.Tensor]:
    """Returns default parameter values."""

    return {
        'identity': torch.zeros(
            size=(n_batch, gnm_np.identity_dim), device=device
        ),
        'expression': torch.zeros(
            size=(n_batch, gnm_np.expression_dim), device=device
        ),
        'rotations': torch.zeros(
            size=(n_batch, gnm_np.num_joints, 3), device=device
        ),
        'translation': torch.zeros(size=(n_batch, 3), device=device),
    }

  def _get_random_kwargs(
      self,
      gnm_np: gnm_numpy.GNM,
      batch: Sequence[int] = (1,),
      device: str = 'cpu',
  ) -> dict[str, torch.Tensor]:
    """Returns random parameter values."""

    return {
        'identity': torch.rand(
            size=(*batch, gnm_np.identity_dim), device=device
        ),
        'expression': torch.rand(
            size=(*batch, gnm_np.expression_dim), device=device
        ),
        'rotations': torch.rand(
            size=(*batch, gnm_np.num_joints, 3), device=device
        ),
        'translation': torch.rand(size=(*batch, 3), device=device),
    }

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_properties_match(self, version: str, variant: Any):
    """Tests that important properties match between implementations."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    self.assertEqual(gnm_torch.version, gnm_np.version)
    self.assertEqual(gnm_torch.identity_dim, gnm_np.identity_dim)
    self.assertEqual(gnm_torch.expression_dim, gnm_np.expression_dim)
    self.assertEqual(gnm_torch.num_joints, gnm_np.num_joints)
    self.assertEqual(gnm_torch.num_vertices, gnm_np.num_vertices)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_parity_with_gnm_numpy(self, version: str, variant: Any):
    """Tests that PyTorch GNM poses vertices the same as NumPy GNM."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    # Build a batch of random parameters.
    n_batch = 10
    parameters_torch = self._get_random_kwargs(
        gnm_np, [n_batch], device=self.device
    )
    actual = gnm_torch(**parameters_torch)

    parameters_np = {
        k: v.detach().cpu().numpy() for k, v in parameters_torch.items()
    }
    desired = gnm_np(**parameters_np)

    np.testing.assert_almost_equal(
        actual.detach().cpu().numpy(), desired, decimal=4
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(gnm_pytorch.GNMVariant.HEAD,),
      batch_size=[(), (2,), (2, 3)],
  )
  def test_vertices_and_landmarks(
      self, version: str, variant: Any, batch_size: tuple[int, ...]
  ):
    """Tests extracting vertices and landmarks in PyTorch."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    parameters_torch = self._get_random_kwargs(
        gnm_np, batch_size, device=self.device
    )
    verts, landmarks = gnm_torch.vertices_and_landmarks(
        gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68, **parameters_torch
    )
    self.assertEqual(verts.shape, (*batch_size, gnm_torch.num_vertices, 3))
    self.assertEqual(landmarks.shape, (*batch_size, 68, 3))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape(self, version: str, variant: Any):
    """Badly shaped parameter should throw an error."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    bad_dimension = (
        gnm_torch.expression_dim + gnm_torch.identity_dim + gnm_np.num_joints
    )
    bad_input = torch.zeros(bad_dimension)
    n_batch = 5
    kwargs = self._get_default_kwargs(gnm_np, n_batch, device=self.device)
    for key in kwargs:
      with self.assertRaises(RuntimeError):
        gnm_torch(**(kwargs | {key: bad_input}))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape_joint_transforms(self, version: str, variant: Any):
    """Badly shaped parameter should throw an error."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    bad_dimension = (
        gnm_torch.expression_dim + gnm_torch.identity_dim + gnm_np.num_joints
    )
    n_batch = 5
    bad_inputs = [
        {'identity': torch.zeros(bad_dimension)},
        {'rotations': torch.zeros([n_batch, gnm_np.num_joints + 1, 3])},
        {'rotations': torch.zeros([n_batch, gnm_np.num_joints, 4])},
        {'translation': torch.zeros([n_batch, 4])},
    ]
    joint_transform_kwargs = self._get_default_kwargs(
        gnm_np, device=self.device
    )
    joint_transform_kwargs.pop('expression')

    for bad_input_dict in bad_inputs:
      with self.assertRaises(RuntimeError):
        gnm_torch.get_posed_joint_transforms(
            **(joint_transform_kwargs | bad_input_dict)
        )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_joint_transforms_numpy_parity(self, version: str, variant: Any):
    """Test that the joint transformations function matches NumPy."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    # Build a batch of random parameters.
    n_batch = 10
    gnm_torch_parameters = self._get_random_kwargs(
        gnm_np, [n_batch], device=self.device
    )
    actual = gnm_torch.get_posed_joint_transforms(
        rotations=gnm_torch_parameters['rotations'],
        identity=gnm_torch_parameters['identity'],
        translation=gnm_torch_parameters['translation'],
    )

    for i in range(n_batch):
      desired = gnm_np.get_posed_joint_transforms(
          rotations=gnm_torch_parameters['rotations'][i].detach().cpu().numpy(),
          translation=gnm_torch_parameters['translation'][i]
          .detach()
          .cpu()
          .numpy(),
          identity=gnm_torch_parameters['identity'][i].detach().cpu().numpy(),
      )
      np.testing.assert_almost_equal(
          actual[i].detach().cpu().numpy(), desired, decimal=4
      )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_prune_vertices(self, version: str, variant: Any):
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    # Prune a copy to avoid modifying the cached model for other tests.
    gnm_pruned = copy.deepcopy(gnm_torch)

    keep_vertices = gnm_np.quads[0]
    gnm_pruned.prune_vertices(keep_vertices)

    # Build a batch of random parameters.
    n_batch = 10
    gnm_torch_parameters = self._get_random_kwargs(
        gnm_np, [n_batch], device=self.device
    )

    vertices_torch = gnm_torch(**gnm_torch_parameters)
    vertices_gathered = vertices_torch[:, keep_vertices]
    vertices_pruned = gnm_pruned(**gnm_torch_parameters)

    self.assertLen(keep_vertices, gnm_pruned.num_vertices)
    self.assertEqual(gnm_pruned.quads.shape, (1, 4))
    self.assertEqual(gnm_pruned.triangles.shape, (2, 3))
    torch.testing.assert_close(vertices_pruned, vertices_gathered)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_no_batch(self, version: str, variant: Any):
    """Tests we can use Torch GNM without a leading batch dimension."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)
    parameters = {
        k: v[0]
        for k, v in self._get_default_kwargs(gnm_np, device=self.device).items()
    }
    vertices = gnm_torch(**parameters)
    self.assertEqual(vertices.shape, (gnm_torch.num_vertices, 3))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_omit_all_parameters(self, version: str, variant: Any):
    """If we omit all parameters, Torch GNM should return the template."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    expected = gnm_np.template_vertex_positions
    actual = gnm_torch().detach().cpu().numpy()
    np.testing.assert_allclose(expected, actual, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dims=[[], [1], [1, 2], [2, 1, 2]],
      parameter_count=[1, 2, 3, 4],
  )
  def test_variable_batch_and_omitted_parameters(
      self,
      version: str,
      variant: Any,
      batch_dims: list[int],
      parameter_count: int,
  ):
    """Exercise GNM with various batch dimensions and omitted parameters."""
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    parameters = self._get_random_kwargs(gnm_np, batch_dims, device=self.device)
    # Omit some parameters.
    for keys in itertools.combinations(parameters, parameter_count):
      sub_parameters = {k: parameters[k] for k in keys}
      gnm_torch(**sub_parameters)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      n_batch=[1, 4, 8],
  )
  def test_vertex_positions_world(
      self, version: str, variant: Any, n_batch: int
  ):
    gnm_torch, gnm_np = self._get_gnms(version, variant)

    parameters = self._get_random_kwargs(gnm_np, [n_batch], device=self.device)

    vertex_positions_bind_pose = gnm_torch.vertex_positions_bind_pose(
        parameters['identity'], parameters['expression']
    )
    pose_correctives = gnm_torch.compute_pose_correctives(
        parameters['rotations']
    )
    vertex_positions_bind_pose += pose_correctives

    joint_positions_bind_pose = gnm_torch.joint_positions_bind_pose(
        parameters['identity']
    )

    actual_vertices = gnm_torch.vertex_positions_world(
        vertex_positions_bind_pose,
        joint_positions_bind_pose,
        parameters['rotations'],
        parameters['translation'],
    )

    expected_vertices = gnm_torch(**parameters)

    np.testing.assert_allclose(
        actual_vertices.detach().cpu().numpy(),
        expected_vertices.detach().cpu().numpy(),
        atol=1e-5,
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_to_numpy_data_dict(self, version: str, variant: Any):
    """Tests to_numpy_data_dict method."""
    gnm = self.gnms_pytorch.get_or_skip(version, variant)

    data_dict = gnm.to_numpy_data_dict()

    for attr in gnm_data_schema.GNM_DATA_ATTRIBUTES:
      self.assertIn(attr, data_dict)

    for key, val in data_dict.items():
      if isinstance(val, np.ndarray):
        pass
      elif isinstance(val, (str, list, tuple)):
        pass
      elif val is None:
        pass
      else:
        self.fail(f'Unexpected type {type(val)} for key {key}')

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_from_gnm(self, version: str, variant: Any):
    """Tests from_gnm factory method."""
    gnm, gnm_np = self._get_gnms(version, variant)

    new_gnm = gnm_pytorch.GNM.from_gnm(gnm)

    self.assertEqual(new_gnm.version, gnm.version)
    self.assertEqual(new_gnm.variant, gnm.variant)

    parameters = self._get_random_kwargs(gnm_np, batch=[10], device=self.device)

    vertices_orig = gnm(**parameters)
    vertices_new = new_gnm(**parameters)
    torch.testing.assert_close(vertices_orig, vertices_new)


class GNMPytorchFactoryMethodsTest(parameterized.TestCase):
  """Tests for instantiating GNM using factory methods."""

  @parameterized.product(
      variant=tuple(_SUPPORTED_VARIANTS),
      version=gnm_test_catalog.MAINTAINED_MAJOR_VERSIONS,
  )
  def test_from_remote_successful(self, variant, version):
    variants = gnm_test_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version]
    if variant.value in variants:
      major_version = gnm_pytorch.GNMMajorVersion(version[1:])

      model = gnm_pytorch.GNM.from_remote(major_version, variant)
      self.assertIsInstance(model, gnm_pytorch.GNM)
    else:
      self.skipTest(f'Variant {variant} not available in version {version}')


if __name__ == '__main__':
  absltest.main(testLoader=gnm_test_utils.ModelOrderedTestLoader())
