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

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_batched
from gnm.shape import gnm_typing
import jax
import numpy as np

enpt = gnm_typing.enpt


class Ray(gnm_batched.BatchedDataclass):
  origin: enpt.FloatArray['*shape 3']
  direction: enpt.FloatArray['*shape 3']
  name: str = 'ray'


class Points(gnm_batched.BatchedDataclass):
  positions: enpt.FloatArray['*shape N 3']
  weights: enpt.FloatArray['*shape N']
  ids: enpt.IntArray['*shape']


def _rays(shape=(4,), seed=0, name='ray') -> Ray:
  generator = np.random.default_rng(seed)
  return Ray(
      origin=generator.normal(size=(*shape, 3)).astype(np.float32),
      direction=generator.normal(size=(*shape, 3)).astype(np.float32),
      name=name,
  )


class ConstructionTest(parameterized.TestCase):

  def test_is_frozen_dataclass(self):
    rays = _rays()

    self.assertTrue(dataclasses.is_dataclass(rays))
    with self.assertRaises(dataclasses.FrozenInstanceError):
      setattr(rays, 'name', 'other')

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


if __name__ == '__main__':
  absltest.main()
