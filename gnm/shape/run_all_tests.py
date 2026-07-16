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

"""Script to run all GNM tests."""

import os
import sys
import unittest
from absl import app
from absl.testing import absltest


def main(_):
  # Discover and run all tests inside this package directory and subdirectories.
  start_dir = os.path.dirname(os.path.abspath(__file__))
  loader = absltest.TestLoader()
  suite = loader.discover(start_dir=start_dir, pattern='*_test.py')
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)
  sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
  app.run(main)
