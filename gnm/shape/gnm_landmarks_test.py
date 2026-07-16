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

"""Tests for validating and loading GNM landmarks configurations."""

from absl.testing import absltest
from gnm.shape import gnm_landmarks
from gnm.shape.data.versions import gnm_specs
import numpy as np


class GNMLandmarksTest(absltest.TestCase):

  def test_load_landmarks_successful(self):
    config = gnm_landmarks.load_landmarks(
        gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68
    )
    self.assertIsInstance(config, gnm_landmarks.LandmarksConfiguration)
    self.assertEqual(config.indices.shape, (68, 3))
    self.assertEqual(config.weights.shape, (68, 3))
    self.assertEqual(config.indices.dtype, np.int32)
    self.assertEqual(config.weights.dtype, np.float32)
    np.testing.assert_allclose(np.sum(config.weights, axis=1), 1.0, atol=1e-3)

  def test_check_body_part_compatibility(self):
    # Compatible with HEAD
    gnm_landmarks.check_body_part_compatibility(
        gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68,
        gnm_specs.GNMBodyPart.HEAD,
    )


if __name__ == '__main__':
  absltest.main()
