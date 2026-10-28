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

"""Tests for TensorFlow OpenCV and OpenGL camera conversion functions."""

from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import camera_conversions_tf
import numpy as np
from scipy.spatial import transform
import tensorflow as tf


_opencv_intrinsics_to_opengl_view_matrix = (
    camera_conversions.opencv_intrinsics_to_opengl_view_matrix
)
_opencv_intrinsics_to_opengl_view_matrix_tf = (
    camera_conversions_tf.opencv_intrinsics_to_opengl_view_matrix_tf
)

_opencv_intrinsics_matrix_to_opengl_view_matrix = (
    camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix
)
_opencv_intrinsics_matrix_to_opengl_view_matrix_tf = (
    camera_conversions_tf.opencv_intrinsics_matrix_to_opengl_view_matrix_tf
)

_opencv_extrinsics_to_opengl = camera_conversions.opencv_extrinsics_to_opengl
_opencv_extrinsics_to_opengl_tf = (
    camera_conversions_tf.opencv_extrinsics_to_opengl_tf
)

_opengl_extrinsics_to_opencv = camera_conversions.opengl_extrinsics_to_opencv
_opengl_extrinsics_to_opencv_tf = (
    camera_conversions_tf.opengl_extrinsics_to_opencv_tf
)

_Rotation: TypeAlias = transform.Rotation

_BATCH_DIMS = (
    (1,),
    (5,),
    (5, 4),
    (5, 4, 3),
    (5, 4, 3, 2),
    (5, 4, 3, 2, 1),
)


class CameraConversionsTfTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(43)
    self.near = 0.01
    self.far = 20

  def _random_extrinsics(self, batch_dims: tuple[int, ...]) -> np.ndarray:
    rotations = _Rotation.random(int(np.prod(batch_dims)), self.rng).as_matrix()
    rotations = np.reshape(rotations, batch_dims + (3, 3))
    translation = self.rng.uniform(-2, 2, size=batch_dims + (3,))

    extrinsics = np.expand_dims(np.eye(4), tuple(range(len(batch_dims))))
    extrinsics = np.tile(extrinsics, batch_dims + (1, 1))
    extrinsics[..., :3, :3] = rotations[..., :3, :3]
    extrinsics[..., :3, 3] = translation
    return extrinsics.astype(np.float32)

  @parameterized.parameters([(dims,) for dims in _BATCH_DIMS])
  def test_opencv_to_opengl_intrinsics_tf_parity(
      self, batch_dims: tuple[int, ...]
  ):
    """Tests that the TensorFlow implementation matches the NumPy one."""
    height, width = 256, 256

    focal_length = np.full(batch_dims + (2,), height, dtype=np.float32)
    principal_point = np.full(
        batch_dims + (2,), [width * 0.5, height * 0.5], dtype=np.float32
    )

    opengl_view_matrix_np = _opencv_intrinsics_to_opengl_view_matrix(
        focal_length,
        principal_point,
        width=width,
        height=height,
        near=self.near,
        far=self.far,
    )
    opengl_view_matrix_tf = _opencv_intrinsics_to_opengl_view_matrix_tf(
        tf.convert_to_tensor(focal_length, dtype=tf.float32),
        tf.convert_to_tensor(principal_point, dtype=tf.float32),
        width=width,
        height=height,
        near=self.near,
        far=self.far,
    )

    np.testing.assert_allclose(
        opengl_view_matrix_np, opengl_view_matrix_tf.numpy()
    )

  @parameterized.parameters([(dims,) for dims in _BATCH_DIMS])
  def test_opencv_intrinsics_matrix_to_opengl_view_matrix_tf_parity(
      self, batch_dims: tuple[int, ...]
  ):
    """Tests that the matrix intrinsics TF helper matches the NumPy one."""
    height, width = 256, 256
    camera_to_image = np.broadcast_to(
        np.array(
            [[256.0, 12.5, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        batch_dims + (3, 3),
    )

    opengl_view_matrix_np = _opencv_intrinsics_matrix_to_opengl_view_matrix(
        camera_to_image,
        width=width,
        height=height,
        near=self.near,
        far=self.far,
    )
    opengl_view_matrix_tf = _opencv_intrinsics_matrix_to_opengl_view_matrix_tf(
        tf.convert_to_tensor(camera_to_image, dtype=tf.float32),
        width=width,
        height=height,
        near=self.near,
        far=self.far,
    )

    np.testing.assert_allclose(
        opengl_view_matrix_np, opengl_view_matrix_tf.numpy()
    )

  @parameterized.parameters([(dims,) for dims in _BATCH_DIMS])
  def test_opencv_to_opengl_extrinsics_tf_parity(
      self, batch_dims: tuple[int, ...]
  ):
    """Tests that the TensorFlow implementation matches the NumPy one."""
    extrinsics = self._random_extrinsics(batch_dims)

    opengl_extrinsics_np = _opencv_extrinsics_to_opengl(extrinsics)
    opengl_extrinsics_tf = _opencv_extrinsics_to_opengl_tf(
        tf.convert_to_tensor(extrinsics, dtype=tf.float32)
    )

    np.testing.assert_allclose(
        opengl_extrinsics_np, opengl_extrinsics_tf.numpy()
    )

  @parameterized.parameters([(dims,) for dims in _BATCH_DIMS])
  def test_opengl_to_opencv_extrinsics_tf_parity(
      self, batch_dims: tuple[int, ...]
  ):
    """Tests that OpenGL-to-OpenCV extrinsics in TF match NumPy."""
    extrinsics = self._random_extrinsics(batch_dims)

    opencv_extrinsics_np = _opengl_extrinsics_to_opencv(extrinsics)
    opencv_extrinsics_tf = _opengl_extrinsics_to_opencv_tf(
        tf.convert_to_tensor(extrinsics, dtype=tf.float32)
    )

    np.testing.assert_allclose(
        opencv_extrinsics_np, opencv_extrinsics_tf.numpy()
    )


if __name__ == "__main__":
  absltest.main()
