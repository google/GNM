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

"""Tests for GNM JAX implementation."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_jax
from gnm.shape import gnm_numpy
from gnm.shape.data.versions import gnm_catalog
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

_SUPPORTED_VARIANTS = frozenset([
    gnm_jax.GNMVariant.HEAD,
])

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP


DTYPES = (
    jnp.float32,
    jnp.float16,
)

DTYPE_TO_DECIMAL = {
    jnp.float32: 4,
    jnp.float16: 2,
}


class GNMJaxTest(parameterized.TestCase):
  rng: np.random.Generator

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.rng = np.random.default_rng(0)

    cls.gnms_np = {}
    cls.gnms_jax = {}
    for version in _MAINTAINED_MAJOR_GNM_VERSIONS:
      cls.gnms_np[version] = {}
      cls.gnms_jax[version] = {}
      for variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]:
        if variant in [v.value for v in _SUPPORTED_VARIANTS]:
          cls.gnms_np[version][variant] = gnm_numpy.GNM.from_local(
              gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
              gnm_numpy.GNMVariant(variant),
          )
          cls.gnms_jax[version][variant] = gnm_jax.GNM.from_local(
              gnm_jax.GNMMajorVersion(version.removeprefix('v')),
              gnm_jax.GNMVariant(variant),
          )

  def _get_default_kwargs(
      self, gnm_np: gnm_numpy.GNM, n_batch: int = 1
  ) -> dict[str, jnp.ndarray]:
    """Returns default parameter values."""

    return {
        'identity': jnp.zeros(shape=(n_batch, gnm_np.identity_dim)),
        'expression': jnp.zeros(shape=(n_batch, gnm_np.expression_dim)),
        'rotations': jnp.zeros(shape=(n_batch, gnm_np.num_joints, 3)),
        'translation': jnp.zeros(shape=(n_batch, 3)),
    }

  def _get_random_kwargs(
      self, gnm_np: gnm_numpy.GNM, n_batch: int | tuple[int, ...] = 1
  ) -> dict[str, jnp.ndarray]:
    """Returns default parameter values."""
    if isinstance(n_batch, int):
      n_batch = (n_batch,)

    expression_shape = (*n_batch, gnm_np.expression_dim)
    return {  # pyrefly: ignore[bad-return]
        'identity': self.rng.uniform(size=(*n_batch, gnm_np.identity_dim)),
        'expression': self.rng.uniform(size=expression_shape),
        'rotations': self.rng.uniform(size=(*n_batch, gnm_np.num_joints, 3)),
        'translation': self.rng.uniform(size=(*n_batch, 3)),
    }

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_properties_match(self, version: str, variant: Any):
    """Tests that important properties match between implementations."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    self.assertEqual(gnm.version, gnm_np.version)
    self.assertEqual(gnm.identity_dim, gnm_np.identity_dim)
    self.assertEqual(gnm.expression_dim, gnm_np.expression_dim)
    self.assertEqual(gnm.num_joints, gnm_np.num_joints)
    self.assertEqual(gnm.num_vertices, gnm_np.num_vertices)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dim=[
          1,
          16,
          [2, 3],
      ],
      dtype=DTYPES,
  )
  def test_parity_with_gnm_numpy(
      self,
      version: str,
      variant: Any,
      batch_dim: int | tuple[int, ...],
      dtype: jnp.dtype,
  ):
    """Tests that JAX GNM poses vertices the same as NumPy GNM."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    # Build a batch of random parameters.
    gnm_jax_parameters = self._get_random_kwargs(gnm_np, batch_dim)
    actual = gnm(**jax.tree.map(jnp.asarray, gnm_jax_parameters))

    desired = gnm_np(**jax.tree.map(np.asarray, gnm_jax_parameters))
    np.testing.assert_almost_equal(
        actual, desired, decimal=DTYPE_TO_DECIMAL[dtype]
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(gnm_jax.GNMVariant.HEAD,),
      batch_size=[(), (2,), (2, 3)],
  )
  def test_vertices_and_landmarks(
      self,
      version: str,
      variant: Any,
      batch_size: tuple[int, ...],
  ):
    """Tests extracting vertices and landmarks in JAX."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    gnm_jax_parameters = self._get_random_kwargs(gnm_np, batch_size)
    verts, landmarks = gnm.vertices_and_landmarks(
        gnm_jax.GNMLandmarksType.HEAD_SPARSE_68,
        **jax.tree.map(jnp.asarray, gnm_jax_parameters),
    )
    self.assertEqual(verts.shape, (*batch_size, gnm.num_vertices, 3))
    self.assertEqual(landmarks.shape, (*batch_size, 68, 3))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      dtype=DTYPES,
  )
  def test_jax_jit_behavior(self, version: str, variant: Any, dtype: jnp.dtype):
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]
    # Build a batch of random parameters.
    n_batch = 10

    gnm_jax_parameters = self._get_random_kwargs(gnm_np, n_batch)
    gnm_jax_parameters = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=dtype), gnm_jax_parameters
    )

    desired = gnm(**gnm_jax_parameters)

    @jax.jit
    def jax_jit_wrapper(identity, expression, rotations, translation):
      return gnm(identity, expression, rotations, translation)

    jax_jit_result = jax_jit_wrapper(**gnm_jax_parameters)
    np.testing.assert_almost_equal(
        np.array(jax_jit_result),
        np.array(desired),
        decimal=DTYPE_TO_DECIMAL[dtype],
    )
    np.testing.assert_almost_equal(
        np.array(jax_jit_result),
        np.array(desired),
        decimal=DTYPE_TO_DECIMAL[dtype],
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape(self, version: str, variant: Any):
    """Badly shaped parameter should throw an error."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    bad_dimension = gnm.expression_dim + gnm.identity_dim + gnm.num_joints
    n_batch = 5
    bad_input = jnp.zeros((n_batch, bad_dimension))
    kwargs = self._get_default_kwargs(gnm_np, n_batch)
    for key in kwargs:
      with self.assertRaises((jt.TypeCheckError, ValueError, TypeError)):
        gnm(**(kwargs | {key: bad_input}))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape_joint_transforms(self, version: str, variant: Any):
    """Badly shaped parameter should throw an error."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    bad_dimension = gnm.identity_dim + gnm.num_joints
    bad_inputs = [
        {'identity': np.zeros(bad_dimension)},
        {'rotations': np.zeros([gnm.num_joints + 1, 3])},
        {'rotations': np.zeros([gnm.num_joints, 4])},
        {'translation': np.zeros([4])},
    ]
    joint_transform_kwargs = self._get_default_kwargs(gnm_np)
    joint_transform_kwargs.pop('expression')
    for bad_input_dict in bad_inputs:
      with self.assertRaises((jt.TypeCheckError, ValueError, TypeError)):
        gnm.get_posed_joint_transforms(
            **(joint_transform_kwargs | bad_input_dict)
        )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      dtype=DTYPES,
  )
  def test_joint_transforms_numpy_parity(
      self, version: str, variant: Any, dtype: jnp.dtype
  ):
    """Test that the joint transformations function matches Numpy."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    # Build a batch of random parameters.
    n_batch = 10
    gnm_jax_parameters = self._get_random_kwargs(gnm_np, n_batch)
    gnm_jax_parameters = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=dtype), gnm_jax_parameters
    )
    actual = gnm.get_posed_joint_transforms(
        rotations=gnm_jax_parameters['rotations'],
        identity=gnm_jax_parameters['identity'],
        translation=gnm_jax_parameters['translation'],
    )

    for i in range(n_batch):
      desired = gnm_np.get_posed_joint_transforms(
          rotations=np.array(gnm_jax_parameters['rotations'][i]),
          translation=np.array(gnm_jax_parameters['translation'][i]),
          identity=np.array(gnm_jax_parameters['identity'][i]),
      )
      np.testing.assert_almost_equal(
          actual[i], desired, decimal=DTYPE_TO_DECIMAL[dtype]
      )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dim=[1, 16],
      use_random_kwargs=[True, False],
      dtype=[jnp.float32, jnp.float64, jnp.bfloat16],
  )
  def test_gradients(
      self,
      version: str,
      variant: Any,
      batch_dim: int,
      use_random_kwargs: bool = False,
      dtype: jnp.dtype = jnp.float32,
  ):
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    # Build a batch of random parameters.
    grad_func = jax.grad(
        lambda *args: jnp.square(gnm(*args)).mean(),
        argnums=np.array([0, 1, 2, 3]),
    )

    if use_random_kwargs:
      gnm_jax_parameters = self._get_random_kwargs(gnm_np, batch_dim)
    else:
      gnm_jax_parameters = self._get_default_kwargs(gnm_np, batch_dim)
    gnm_jax_parameters = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=dtype), gnm_jax_parameters
    )
    grads = grad_func(
        gnm_jax_parameters['identity'],
        gnm_jax_parameters['expression'],
        gnm_jax_parameters['rotations'],
        gnm_jax_parameters['translation'],
    )
    self.assertLen(grads, 4)
    self.assertEqual(grads[0].shape, (batch_dim, gnm_np.identity_dim))
    self.assertEqual(grads[1].shape, (batch_dim, gnm_np.expression_dim))
    self.assertEqual(grads[2].shape, (batch_dim, gnm_np.num_joints, 3))
    self.assertEqual(grads[3].shape, (batch_dim, 3))

    np.testing.assert_array_equal(
        np.isnan(np.array(grads[0])),
        False,
        err_msg='Found NaN in identity gradient.',
    )
    np.testing.assert_array_equal(
        np.isnan(np.array(grads[1])),
        False,
        err_msg='Found NaN in expression gradient.',
    )
    np.testing.assert_array_equal(
        np.isnan(np.array(grads[2])),
        False,
        err_msg='Found NaN in rotations gradient.',
    )
    np.testing.assert_array_equal(
        np.isnan(np.array(grads[3])),
        False,
        err_msg='Found NaN in translation gradient.',
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_transformed_local_offsets(self, version: str, variant: Any):
    """Tests that vertex offset is applied to the posed positions."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant_str]
    gnm = self.gnms_jax[version][variant_str]

    gnm_jax_parameters = self._get_default_kwargs(n_batch=10, gnm_np=gnm_np)
    local_vertices = gnm.vertex_positions_bind_pose(
        gnm_jax_parameters['identity'], gnm_jax_parameters['expression']
    )
    vertex_offsets = jnp.ones_like(local_vertices)

    joints = gnm.joint_positions_bind_pose(gnm_jax_parameters['identity'])
    world_vertices_with_offset = gnm.apply_linear_blend_skinning(
        local_vertices + vertex_offsets,
        joints,
        gnm_jax_parameters['rotations'],
        gnm_jax_parameters['translation'],
    )
    world_vertices_without_offset = gnm.apply_linear_blend_skinning(
        local_vertices,
        joints,
        gnm_jax_parameters['rotations'],
        gnm_jax_parameters['translation'],
    )
    self.assertSequenceEqual(
        world_vertices_with_offset.shape, world_vertices_without_offset.shape
    )
    # Check that distances between vertices are equal to length of offsets.
    mesh_space_offset_norm = jnp.linalg.norm(vertex_offsets, axis=-1)
    world_space_offset_norm = jnp.linalg.norm(
        world_vertices_with_offset - world_vertices_without_offset, axis=-1
    )
    np.testing.assert_array_almost_equal(
        world_space_offset_norm,
        mesh_space_offset_norm,
        decimal=4,
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_to_numpy_data_dict(self, version: str, variant: Any):
    """Tests to_numpy_data_dict method."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]

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
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_jax[version][variant_str]

    new_gnm = gnm_jax.GNM.from_gnm(gnm)

    self.assertEqual(new_gnm.version, gnm.version)
    self.assertEqual(new_gnm.variant, gnm.variant)

    parameters = self._get_random_kwargs(
        self.gnms_np[version][variant_str], n_batch=(10,)
    )
    parameters_jax = jax.tree.map(jnp.asarray, parameters)

    vertices_orig = gnm(**parameters_jax)
    vertices_new = new_gnm(**parameters_jax)
    np.testing.assert_almost_equal(
        np.array(vertices_orig), np.array(vertices_new), decimal=4
    )


class GNMJaxFactoryMethodsTest(parameterized.TestCase):
  """Tests for instantiating GNM using factory methods."""

  @parameterized.product(
      variant=tuple(_SUPPORTED_VARIANTS),
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
  )
  def test_from_local_successful(self, variant, version):
    if variant.value in gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version]:
      major_version = gnm_jax.GNMMajorVersion(version[1:])

      model = gnm_jax.GNM.from_local(major_version, variant)
      self.assertIsInstance(model, gnm_jax.GNM)
    else:
      self.skipTest(f'Variant {variant} not available in version {version}')


if __name__ == '__main__':
  absltest.main()
