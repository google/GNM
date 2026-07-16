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

"""Tests for GNM TensorFlow implementation."""

import itertools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_numpy
from gnm.shape import gnm_tensorflow
from gnm.shape import gnm_test_utils
from gnm.shape.data.versions import gnm_catalog
import numpy as np
import tensorflow as tf

_SUPPORTED_VARIANTS = frozenset([
    gnm_tensorflow.GNMVariant.HEAD,
])

_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP

BATCH_DIMS = [tuple(), tuple([1]), tuple([1, 2]), tuple([2, 1, 2])]


class GNMTensorflowTest(parameterized.TestCase):
  rng: np.random.Generator

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.rng = np.random.default_rng(0)
    tf.random.set_seed(0)

    cls.gnms_np = {}
    cls.gnms_tf = {}
    for version in _MAINTAINED_MAJOR_GNM_VERSIONS:
      cls.gnms_np[version] = {}
      cls.gnms_tf[version] = {}
      for variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]:
        if variant in [v.value for v in _SUPPORTED_VARIANTS]:
          cls.gnms_np[version][variant] = gnm_numpy.GNM.from_local(
              gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
              gnm_numpy.GNMVariant(variant),
          )
          cls.gnms_tf[version][variant] = gnm_tensorflow.GNM.from_local(
              gnm_tensorflow.GNMMajorVersion(version.removeprefix('v')),
              gnm_tensorflow.GNMVariant(variant),
          )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_properties_match(self, version: str, variant: Any):
    """Tests that important properties match between implementations."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm_tf = self.gnms_tf[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    self.assertEqual(gnm_tf.version, gnm_np.version)
    self.assertEqual(gnm_tf.identity_dim, gnm_np.identity_dim)
    self.assertEqual(gnm_tf.expression_dim, gnm_np.expression_dim)
    self.assertEqual(gnm_tf.num_joints, gnm_np.num_joints)
    self.assertEqual(gnm_tf.num_vertices, gnm_np.num_vertices)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dims=BATCH_DIMS,
  )
  def test_parity_with_gnm_numpy(
      self, version: str, variant: str, batch_dims: tuple[int, ...]
  ):
    """Tests that TensorFlow GNM poses vertices the same as NumPy GNM."""
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    # Build a batch of random parameters.
    parameters_np = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=batch_dims, seed=self.rng
    )
    parameters_tf = {
        k: tf.convert_to_tensor(v, dtype=tf.float32)
        for k, v in parameters_np.items()
    }
    actual = gnm_tf(**parameters_tf)
    desired = gnm_np(**parameters_np)

    np.testing.assert_almost_equal(actual, desired, decimal=4)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(gnm_tensorflow.GNMVariant.HEAD,),
      batch_size=[(), (2,), (2, 3)],
  )
  def test_vertices_and_landmarks(
      self, version: str, variant: Any, batch_size: tuple[int, ...]
  ):
    """Tests extracting vertices and landmarks in TensorFlow."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm_tf = self.gnms_tf[version][variant_str]
    gnm_np = self.gnms_np[version][variant_str]

    parameters_np = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=batch_size, seed=self.rng
    )
    parameters_tf = {
        k: tf.convert_to_tensor(v, dtype=tf.float32)
        for k, v in parameters_np.items()
    }
    verts, landmarks = gnm_tf.vertices_and_landmarks(
        gnm_tensorflow.GNMLandmarksType.HEAD_SPARSE_68, **parameters_tf
    )
    self.assertEqual(verts.shape, (*batch_size, gnm_tf.num_vertices, 3))
    self.assertEqual(landmarks.shape, (*batch_size, 68, 3))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dims=BATCH_DIMS,
  )
  def test_tf_function_behavior(
      self, version: str, variant: str, batch_dims: tuple[int, ...]
  ):
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    # Build a batch of random parameters.
    gnm_tf_parameters = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=batch_dims, seed=self.rng
    )

    desired = gnm_tf(**gnm_tf_parameters)

    @tf.function
    def tf_function_wrapper(identity, expression, rotations, translation):
      return gnm_tf(identity, expression, rotations, translation)

    @tf.function(jit_compile=True)
    def tf_function_wrapper_xla(identity, expression, rotations, translation):
      return gnm_tf(identity, expression, rotations, translation)

    tf_func_result = tf_function_wrapper(**gnm_tf_parameters)
    tf_func_xla_result = tf_function_wrapper_xla(**gnm_tf_parameters)

    np.testing.assert_almost_equal(
        tf_func_result.numpy(), desired.numpy(), decimal=4
    )
    np.testing.assert_almost_equal(
        tf_func_xla_result.numpy(), desired.numpy(), decimal=4
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape(self, version: str, variant: str):
    """Badly shaped parameter should throw an error."""
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    bad_dimension = (
        gnm_tf.expression_dim + gnm_tf.identity_dim + gnm_np.num_joints
    )
    n_batch = 5
    bad_input = tf.zeros((n_batch, bad_dimension), dtype=tf.float32)
    kwargs = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=(n_batch,), seed=self.rng
    )
    for key in kwargs:
      with self.assertRaises(ValueError):
        gnm_tf(**(kwargs | {key: bad_input}))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape_joint_transforms(self, version: str, variant: str):
    """Badly shaped parameter should throw an error."""
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    bad_dimension = (
        gnm_tf.expression_dim + gnm_tf.identity_dim + gnm_np.num_joints
    )
    n_batch = 5
    bad_inputs = [
        {'identity': tf.zeros((bad_dimension), dtype=tf.float32)},
        {'identity': tf.zeros((n_batch, bad_dimension), dtype=tf.float32)},
        {'rotations': tf.zeros([n_batch, gnm_np.num_joints + 1, 3])},
        {'rotations': tf.zeros([n_batch, gnm_np.num_joints, 4])},
        {'translation': tf.zeros([n_batch, 4])},
    ]
    joint_transform_kwargs = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=(n_batch,), seed=self.rng
    )
    joint_transform_kwargs.pop('expression')

    for bad_input_dict in bad_inputs:
      with self.assertRaises(ValueError):
        gnm_tf.get_posed_joint_transforms(
            **(joint_transform_kwargs | bad_input_dict)
        )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dims=BATCH_DIMS,
  )
  def test_joint_transforms_numpy_parity(
      self, version: str, variant: str, batch_dims: tuple[int, ...]
  ):
    """Test that the joint transformations function matches NumPy."""
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    # Build a batch of random parameters.
    parameters_np = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=batch_dims, seed=self.rng
    )
    gnm_tf_parameters = {
        k: tf.convert_to_tensor(v, dtype=tf.float32)
        for k, v in parameters_np.items()
    }
    actual = gnm_tf.get_posed_joint_transforms(
        rotations=gnm_tf_parameters['rotations'],
        identity=gnm_tf_parameters['identity'],
        translation=gnm_tf_parameters['translation'],
    )

    actual = np.asarray(actual, dtype=np.float32)

    n_batch = int(np.prod(batch_dims))
    actual = actual.reshape(n_batch, gnm_np.num_joints, 4, 4)

    rotations = parameters_np['rotations'].reshape(
        n_batch, gnm_np.num_joints, 3
    )
    translation = parameters_np['translation'].reshape(n_batch, 3)
    identity = parameters_np['identity'].reshape(n_batch, gnm_np.identity_dim)

    for i in range(n_batch):
      desired = gnm_np.get_posed_joint_transforms(
          rotations=rotations[i],
          translation=translation[i],
          identity=identity[i],
      )
      np.testing.assert_almost_equal(actual[i], desired, decimal=4)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_prune_vertices(self, version: str, variant: str):
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    gnm_pruned = gnm_tensorflow.GNM.from_local(
        gnm_tensorflow.GNMMajorVersion(version.removeprefix('v')),
        gnm_tensorflow.GNMVariant(variant),
    )

    keep_vertices = gnm_np.quads[0]
    gnm_pruned.prune_vertices(keep_vertices)

    # Build a batch of random parameters.
    n_batch = 10
    gnm_tf_parameters = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=(n_batch,), seed=self.rng
    )

    vertices_tf = gnm_tf(**gnm_tf_parameters)
    vertices_gathered = tf.gather(vertices_tf, keep_vertices, axis=1)
    vertices_pruned = gnm_pruned(
        **gnm_tf_parameters  # pyrefly: ignore[bad-argument-type]
    )

    with self.subTest('Test number of vertices'):
      self.assertLen(keep_vertices, gnm_pruned.num_vertices)
    with self.subTest('Test quads shape'):
      self.assertEqual(gnm_pruned.quads.shape, (1, 4))
    with self.subTest('Test quad indices correct'):
      quad_indices = set(gnm_pruned.quads.numpy().flatten().tolist())
      self.assertSetEqual(quad_indices, {0, 1, 2, 3})
    with self.subTest('Test triangles shape'):
      self.assertEqual(gnm_pruned.triangles.shape, (2, 3))
    with self.subTest('Test triangle indices correct'):
      triangle_indices = set(gnm_pruned.triangles.numpy().flatten().tolist())
      self.assertSetEqual(triangle_indices, {0, 1, 2, 3})
    with self.subTest('Test vertex positions'):
      tf.debugging.assert_near(vertices_pruned, vertices_gathered)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_no_batch(self, version: str, variant: str):
    """Tests we can use TF GNM without a leading batch dimension."""
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]
    parameters = {
        k: v[0]
        for k, v in gnm_test_utils.random_gnm_parameters(
            gnm_np, batch_shape=(1,), seed=self.rng
        ).items()
    }
    vertices = gnm_tf(**parameters)
    self.assertEqual(vertices.shape, (gnm_tf.num_vertices, 3))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_omit_all_parameters(self, version: str, variant: str):
    """If we omit all parameters, TF GNM should return the template."""
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    expected = gnm_np.template_vertex_positions
    actual = gnm_tf().numpy()
    np.testing.assert_allclose(expected, actual, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dims=BATCH_DIMS,
      parameter_count=[1, 2, 3, 4],
  )
  def test_variable_batch_and_omitted_parameters(
      self,
      version: str,
      variant: str,
      batch_dims: tuple[int, ...],
      parameter_count: int,
  ):
    """Exercise GNM with various batch dimensions and omitted parameters."""
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    parameters = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=batch_dims, seed=self.rng
    )
    # Omit some parameters.
    for keys in itertools.combinations(parameters, parameter_count):
      sub_parameters = {k: parameters[k] for k in keys}
      gnm_tf(**sub_parameters)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dims=BATCH_DIMS,
  )
  def test_vertex_positions_world(
      self, version: str, variant: str, batch_dims: tuple[int, ...]
  ):
    if variant not in self.gnms_np[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms_np[version][variant]
    gnm_tf = self.gnms_tf[version][variant]

    parameters = gnm_test_utils.random_gnm_parameters(
        gnm_np, batch_shape=batch_dims, seed=self.rng
    )
    parameters = {
        k: tf.convert_to_tensor(v, dtype=tf.float32)
        for k, v in parameters.items()
    }

    vertex_positions_bind_pose = gnm_tf.vertex_positions_bind_pose(
        parameters['identity'], parameters['expression']
    )
    pose_correctives = gnm_tf.compute_pose_correctives(parameters['rotations'])
    vertex_positions_bind_pose += pose_correctives

    joint_positions_bind_pose = gnm_tf.joint_positions_bind_pose(
        parameters['identity']
    )

    actual_vertices = gnm_tf.vertex_positions_world(
        vertex_positions_bind_pose,
        joint_positions_bind_pose,
        parameters['rotations'],
        parameters['translation'],
    )

    expected_vertices = gnm_tf(**parameters)

    np.testing.assert_allclose(actual_vertices, expected_vertices, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_to_numpy_data_dict(self, version: str, variant: Any):
    """Tests to_numpy_data_dict method."""
    variant_str = variant.value if hasattr(variant, 'value') else variant
    if variant_str not in self.gnms_np[version]:
      self.skipTest(f'variant {variant_str} not supported in {version}.')
    gnm = self.gnms_tf[version][variant_str]

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
    gnm = self.gnms_tf[version][variant_str]

    new_gnm = gnm_tensorflow.GNM.from_gnm(gnm)

    self.assertEqual(new_gnm.version, gnm.version)
    self.assertEqual(new_gnm.variant, gnm.variant)

    parameters = gnm_test_utils.random_gnm_parameters(
        self.gnms_np[version][variant_str], seed=self.rng
    )
    parameters_tf = {
        k: tf.convert_to_tensor(v, dtype=tf.float32)
        for k, v in parameters.items()
    }
    vertices_orig = gnm(**parameters_tf)
    vertices_new = new_gnm(**parameters_tf)
    np.testing.assert_allclose(
        vertices_orig.numpy(), vertices_new.numpy(), atol=1e-5
    )


class GNMTensorflowFactoryMethodsTest(parameterized.TestCase):
  """Tests for instantiating GNM using factory methods."""

  @parameterized.product(
      variant=list(_SUPPORTED_VARIANTS),
      version=gnm_catalog.MAINTAINED_MAJOR_VERSIONS,
  )
  def test_from_local_successful(self, variant, version):
    if variant.value in gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP[version]:
      major_version = gnm_tensorflow.GNMMajorVersion(version[1:])

      model = gnm_tensorflow.GNM.from_local(major_version, variant)
      self.assertIsInstance(model, gnm_tensorflow.GNM)
    else:
      self.skipTest(f'Variant {variant} not available in version {version}')


if __name__ == '__main__':
  absltest.main()
