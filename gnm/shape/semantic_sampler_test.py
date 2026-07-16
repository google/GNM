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

"""Tests for G-Nome identity and expression sampler."""

# pylint: disable=protected-access

from absl.testing import absltest
from gnm.shape import semantic_sampler
import numpy as np


class SamplerTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.expression_sampler = semantic_sampler.ExpressionSampler()
    cls.identity_sampler = semantic_sampler.IdentitySampler()
    cls.expression_dim = cls.expression_sampler._decoder.output_shape[-1]
    cls.identity_dim = cls.identity_sampler._decoder.output_shape[-1]

  def test_expression_sampler_sample_expression(self):
    """Tests that ExpressionSampler can sample expressions."""
    num_samples = 3
    class_label_index = semantic_sampler.Expression.SURPRISE
    generated_expressions = self.expression_sampler.sample_expression(
        class_label_index, num_samples
    )
    self.assertEqual(
        generated_expressions.shape, (num_samples, self.expression_dim)
    )

  def test_expression_sampler_blend_expressions(self):
    """Tests that ExpressionSampler can blend expressions."""
    class_weights = {
        semantic_sampler.Expression.SURPRISE: 0.5,
        semantic_sampler.Expression.HAPPY: 0.5,
    }  # surprise and happy
    blended_expression = self.expression_sampler.blend_expressions(
        class_weights
    )
    self.assertEqual(blended_expression.shape, (self.expression_dim,))

  def test_expression_sampler_randomize_expressions(self):
    """Tests that ExpressionSampler can randomize expressions."""
    num_samples = 5
    random_expressions = self.expression_sampler.randomize_expressions(
        num_samples
    )
    self.assertEqual(
        random_expressions.shape, (num_samples, self.expression_dim)
    )

  def test_identity_sampler_sample_identity(self):
    """Tests that IdentitySampler can sample identities."""
    num_samples = 2
    gender_class = semantic_sampler.Gender.FEMALE
    ethnicity_class = semantic_sampler.Ethnicity.MIDDLE_EASTERN
    generated_identities = self.identity_sampler.sample_identity(
        gender_class, ethnicity_class, num_samples
    )
    self.assertEqual(
        generated_identities.shape, (num_samples, self.identity_dim)
    )

  def test_identity_sampler_blend_identities(self):
    """Tests that IdentitySampler can blend identities."""
    num_samples = 1
    gender_weights = {
        semantic_sampler.Gender.FEMALE: 0.5,
        semantic_sampler.Gender.MALE: 0.5,
    }  # Female and Male
    ethnicity_weights = {
        semantic_sampler.Ethnicity.MIDDLE_EASTERN: 0.25,
        semantic_sampler.Ethnicity.ASIAN: 0.25,
        semantic_sampler.Ethnicity.WHITE: 0.25,
        semantic_sampler.Ethnicity.BLACK: 0.25,
    }  # All ethnicities
    blended_identity = self.identity_sampler.blend_identities(
        gender_weights, ethnicity_weights, num_samples
    )
    self.assertEqual(blended_identity.shape, (num_samples, self.identity_dim))

  def test_identity_sampler_randomize_identities(self):
    """Tests that IdentitySampler can randomize identities."""
    num_samples = 5
    random_identities = self.identity_sampler.randomize_identities(num_samples)
    self.assertEqual(random_identities.shape, (num_samples, self.identity_dim))

  def test_deterministic_sampling(self):
    """Tests that sampling with an explicit RNG is deterministic."""
    # Test ExpressionSampler
    class_label = semantic_sampler.Expression.HAPPY
    seed = 42

    rng1 = np.random.default_rng(seed)
    expr1 = self.expression_sampler.sample_expression(class_label, rng=rng1)

    rng2 = np.random.default_rng(seed)
    expr2 = self.expression_sampler.sample_expression(class_label, rng=rng2)

    np.testing.assert_array_equal(expr1, expr2)

    # Test IdentitySampler
    gender = semantic_sampler.Gender.MALE
    ethnicity = semantic_sampler.Ethnicity.ASIAN

    rng3 = np.random.default_rng(seed)
    ident1 = self.identity_sampler.sample_identity(gender, ethnicity, rng=rng3)

    rng4 = np.random.default_rng(seed)
    ident2 = self.identity_sampler.sample_identity(gender, ethnicity, rng=rng4)

    np.testing.assert_array_equal(ident1, ident2)

  def test_non_deterministic_sampling(self):
    """Tests that sampling without an explicit RNG is non-deterministic."""
    # Test ExpressionSampler
    class_label = semantic_sampler.Expression.HAPPY
    expr1 = self.expression_sampler.sample_expression(class_label, rng=None)
    expr2 = self.expression_sampler.sample_expression(class_label, rng=None)
    self.assertFalse(np.array_equal(expr1, expr2))

    # Test IdentitySampler
    gender = semantic_sampler.Gender.MALE
    ethnicity = semantic_sampler.Ethnicity.ASIAN
    ident1 = self.identity_sampler.sample_identity(gender, ethnicity, rng=None)
    ident2 = self.identity_sampler.sample_identity(gender, ethnicity, rng=None)
    self.assertFalse(np.array_equal(ident1, ident2))


if __name__ == '__main__':
  absltest.main()
