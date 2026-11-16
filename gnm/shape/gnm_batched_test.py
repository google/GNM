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

"""Unit tests of `gnm_batched` (randomized: `gnm_batched_fuzz_test`)."""

from __future__ import annotations

import dataclasses
import enum

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_batched
from gnm.shape import gnm_typing
import jax
import jax.numpy as jnp
import numpy as np
import torch

enpt = gnm_typing.enpt


class Label(enum.StrEnum):
  A = 'a'
  B = 'b'


class Ray(gnm_batched.BatchedDataclass):
  origin: enpt.FloatArray['*shape 3']
  direction: enpt.FloatArray['*shape 3']
  name: str = 'ray'


class Points(gnm_batched.BatchedDataclass):
  positions: enpt.FloatArray['*shape N 3']
  weights: enpt.FloatArray['*shape N']
  ids: enpt.IntArray['*shape']


class Scene(gnm_batched.BatchedDataclass):
  rays: Ray
  time: enpt.FloatArray['*shape']
  confidence: enpt.FloatArray['*shape'] | None = None
  label: Label = Label.A


class StrictRay(gnm_batched.BatchedDataclass, broadcast=False):
  origin: enpt.FloatArray['*shape 3']
  direction: enpt.FloatArray['*shape 3']


class Extras(gnm_batched.BatchedDataclass):
  """Only optional arrays: no batch shape when they are all None."""

  confidence: enpt.FloatArray['*shape'] | None = None
  source: str = 'model'


class Frame(gnm_batched.BatchedDataclass):
  extras: Extras  # The first child.
  time: enpt.FloatArray['*shape']


def _rays(shape=(4,), seed=0, name='ray') -> Ray:
  generator = np.random.default_rng(seed)
  return Ray(
      origin=generator.normal(size=(*shape, 3)).astype(np.float32),
      direction=generator.normal(size=(*shape, 3)).astype(np.float32),
      name=name,
  )


class ConstructionTest(parameterized.TestCase):

  def test_properties(self):
    rays = _rays(shape=(2, 3))

    self.assertEqual(rays.shape, (2, 3))
    self.assertEqual(rays.ndim, 2)
    self.assertEqual(rays.size, 6)
    self.assertLen(rays, 2)
    self.assertIs(rays.xnp, np)
    self.assertEqual(rays.name, 'ray')

  def test_is_frozen_dataclass(self):
    rays = _rays()

    self.assertTrue(dataclasses.is_dataclass(rays))
    with self.assertRaises(dataclasses.FrozenInstanceError):
      setattr(rays, 'name', 'other')

  def test_dtypes_are_normalized(self):
    points = Points(
        positions=np.zeros((2, 5, 3), np.int32),
        weights=[[1, 2, 3, 4, 5]] * 2,
        ids=np.arange(2, dtype=np.int64),
    )

    self.assertEqual(points.positions.dtype, np.float32)
    self.assertEqual(points.weights.dtype, np.float32)
    self.assertEqual(points.ids.dtype, np.int64)
    float64 = Points(positions=np.zeros((5, 3)), weights=np.zeros(5), ids=0)
    self.assertEqual(float64.positions.dtype, np.float64)

  @parameterized.named_parameters(
      ('literal_dim', (2, 5, 4), (2, 5), r'\(\*shape N 3\)\.'),
      ('named_dim', (2, 5, 3), (2, 4), 'N=5'),
      ('too_few_dims', (2, 5, 3), (), r'\(\*shape N\)\.'),
  )
  def test_invalid_inner_shapes_raise(self, positions, weights, regex):
    with self.assertRaisesRegex(ValueError, regex):
      Points(
          positions=np.zeros(positions),
          weights=np.zeros(weights),
          ids=np.zeros(2, np.int32),
      )

  def test_broadcasting(self):
    rays = Ray(origin=np.zeros(3), direction=np.ones((2, 1, 3)))
    self.assertEqual(rays.shape, (2, 1))

    rays = Ray(origin=np.zeros((1, 4, 3)), direction=np.ones((2, 1, 3)))
    self.assertEqual(rays.shape, (2, 4))
    self.assertEqual(rays.origin.shape, (2, 4, 3))

  @parameterized.named_parameters(
      ('conflicting', (3,), (4,)),
      ('lower_rank', (4,), (2, 4)),
  )
  def test_conflicting_batch_shapes_raise(self, shape1, shape2):
    with self.assertRaisesRegex(ValueError, 'Conflicting batch shapes'):
      Ray(origin=np.zeros((*shape1, 3)), direction=np.zeros((*shape2, 3)))

  def test_broadcast_can_be_disabled(self):
    with self.assertRaisesRegex(ValueError, 'does not broadcast'):
      StrictRay(origin=np.zeros(3), direction=np.zeros((2, 3)))
    self.assertEqual(
        StrictRay(origin=np.zeros((2, 3)), direction=np.zeros((2, 3))).shape,
        (2,),
    )

  def test_nested_fields_are_broadcast(self):
    scene = Scene(rays=_rays(shape=()), time=np.arange(4.0))

    self.assertEqual(scene.shape, (4,))
    self.assertEqual(scene.rays.shape, (4,))
    self.assertEqual(scene.rays.origin.shape, (4, 3))

  def test_invalid_nested_type_raises(self):
    with self.assertRaisesRegex(TypeError, 'Scene.rays must be a Ray'):
      Scene(rays=np.zeros((4, 3)), time=np.zeros(4))  # pyrefly: ignore[bad-argument-type]

  def test_invalid_annotation_raises(self):

    class Invalid(gnm_batched.BatchedDataclass):
      values: enpt.FloatArray['N 3']

    with self.assertRaisesRegex(TypeError, 'must be of the form'):
      Invalid(values=np.zeros((2, 3)))

  def test_array_annotations_without_etils_raise(self):
    # Such fields would silently be static, i.e. neither indexed nor converted.

    class NdarrayField(gnm_batched.BatchedDataclass):
      points: np.ndarray

    class UnionField(gnm_batched.BatchedDataclass):
      values: enpt.FloatArray['*shape'] | enpt.IntArray['*shape']

    class JaxField(gnm_batched.BatchedDataclass):
      values: jax.Array | None = None

    with self.assertRaisesRegex(TypeError, 'Unsupported annotation'):
      NdarrayField(points=np.zeros(3))
    with self.assertRaisesRegex(TypeError, 'Unsupported annotation'):
      UnionField(values=np.zeros(3))
    with self.assertRaisesRegex(TypeError, 'Unsupported annotation'):
      JaxField()

  def test_unions_of_batched_types_raise(self):

    class UnionField(gnm_batched.BatchedDataclass):
      x: enpt.FloatArray['*shape']
      value: Ray | Points

    with self.assertRaisesRegex(TypeError, 'Unsupported annotation'):
      UnionField(x=np.zeros(4), value=_rays())

  def test_none_for_required_fields_raises(self):
    with self.assertRaisesRegex(TypeError, 'Ray.origin is required, got None'):
      Ray(origin=None, direction=np.zeros(3))  # pyrefly: ignore[bad-argument-type]
    with self.assertRaisesRegex(TypeError, 'Scene.rays is required, got None'):
      Scene(rays=None, time=np.zeros(4))  # pyrefly: ignore[bad-argument-type]
    with self.assertRaisesRegex(TypeError, 'Ray.direction is required'):
      _rays().replace(direction=None)
    # Optional fields may be None, and nested values may have no arrays.
    self.assertIsNone(Scene(rays=_rays(), time=np.zeros(4)).confidence)
    self.assertEqual(Frame(extras=Extras(), time=np.zeros(2)).shape, (2,))

  def test_repr(self):
    expected = (
        "Ray(shape=(4,), origin=f32[4 3], direction=f32[4 3], name='ray')"
    )
    self.assertEqual(repr(_rays()), expected)
    self.assertEqual(repr(_rays().as_torch()), expected)

  def test_truthiness(self):
    self.assertTrue(_rays(shape=()))
    with self.assertRaisesRegex(ValueError, 'ambiguous'):
      bool(_rays(shape=(0,)))


class TransformationTest(parameterized.TestCase):

  def test_replace_validates(self):
    rays = _rays()

    self.assertEqual(rays.replace(origin=np.zeros(3)).origin.shape, (4, 3))
    with self.assertRaisesRegex(ValueError, 'Conflicting batch shapes'):
      rays.replace(origin=np.zeros((3, 3)))


class BackendTest(parameterized.TestCase):

  def test_numpy_arrays_adopt_the_other_backend(self):
    rays = Ray(origin=np.zeros(3), direction=jnp.ones((2, 3)))

    self.assertIsInstance(rays.origin, jax.Array)
    self.assertEqual(rays.shape, (2,))

  def test_mixed_backends_raise(self):
    with self.assertRaisesRegex(ValueError, 'Cannot mix array backends'):
      Ray(origin=torch.zeros(3), direction=jnp.ones((2, 3)))


if __name__ == '__main__':
  absltest.main()
