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

"""Contains utilities for converting cameras to different camera conventions.

OpenCV uses a right-handed coordinate system with the X-axis pointing right,
the Y-axis pointing down and the Z-axis pointing towards the screen. OpenGL uses
a right-handed coordinate system in *world-space* with the X-axis pointing
right, the Y-axis pointing up and the Z-axis pointing towards the viewer.
"""

import numpy as np
import tensorflow as tf

# Rotate OpenCV's camera coordinate system 180 degrees around the X-axis to
# align it with the OpenGL coordinate frame.
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(dtype=np.float32)
OPENGL_TO_OPENCV = OPENCV_TO_OPENGL.copy()


def opencv_extrinsics_to_opengl(extrinsics: np.ndarray) -> np.ndarray:
  """Converts OpenCV extrinsics to OpenGL.

  Args:
    extrinsics: The camera extrinsics or world-to-camera transformation matrix,
      (A1, A2, ..., An, 4, 4).

  Returns:
    Camera extrinsics in OpenGL coordinate system, with the same shape as input.
  """
  return np.einsum('mk,...kn->...mn', OPENCV_TO_OPENGL, extrinsics)


def opengl_extrinsics_to_opencv(extrinsics: np.ndarray) -> np.ndarray:
  """Converts OpenGL extrinsics to OpenCV.

  Args:
    extrinsics: The camera extrinsics or world-to-camera transformation matrix,
      (A1, A2, ..., An, 4, 4).

  Returns:
    Camera extrinsics in OpenCV coordinate system, with the same shape as input.
  """
  return np.einsum('mk,...kn->...mn', OPENGL_TO_OPENCV, extrinsics)


def opengl_intrinsics_to_opencv_matrix(
    opengl_intrinsics: np.ndarray,
    height: np.ndarray | int,
    width: np.ndarray | int,
) -> np.ndarray:
  """Converts OpenGL intrinsics to OpenCV matrix.

  Args:
    opengl_intrinsics: The camera intrinsics or camera-to-image transformation
      matrix, (A1, A2, ..., An, 4, 4).
    height: The height of the image.
    width: The width of the image.

  Returns:
    Camera intrinsics in OpenCV coordinate system, with the same shape as input.
  """

  fx = opengl_intrinsics[..., 0, 0] * width / 2.0
  fy = opengl_intrinsics[..., 1, 1] * height / 2.0
  cx = (1 - opengl_intrinsics[..., 0, 2]) * width / 2.0
  cy = (opengl_intrinsics[..., 1, 2] + 1) * height / 2.0

  skew = -opengl_intrinsics[..., 0, 1] / 2.0 * width

  # Add a singleton dimension to the last two dimensions.
  fx = fx[..., np.newaxis]
  fy = fy[..., np.newaxis]
  cx = cx[..., np.newaxis]
  cy = cy[..., np.newaxis]
  skew = skew[..., np.newaxis]

  zero = np.zeros_like(fx)
  ones = np.ones_like(fx)

  row1 = [fx, skew, cx, zero]
  row2 = [zero, fy, cy, zero]
  row3 = [zero, zero, ones, zero]
  row4 = [zero, zero, zero, ones]

  view_matrix = np.concatenate(row1 + row2 + row3 + row4, axis=-1)
  output_shape = np.concatenate([opengl_intrinsics.shape[:-2], [4, 4]], axis=0)
  output_shape = output_shape.astype(np.int32)
  view_matrix = np.reshape(view_matrix, output_shape)
  return view_matrix


def opencv_intrinsics_to_opengl_view_matrix(
    focal_length: np.ndarray,
    principal_point: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
    skew_coefficient: np.ndarray | float = 0.0,
) -> np.ndarray:
  """Converts OpenCV camera intrinsics to OpenGL view matrix.

  See here for more information:
  https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

  Args:
    focal_length: Camera focal length, ([A1, ...., An], 2), in pixels.
    principal_point: Camera center, ([A1, ...., An], 2), in pixels.
    width: Width of the image, in pixels.
    height: Height of the image, in pixels.
    near: Depth of the near clipping plane in meters (m).
    far: Depth of the far clipping plane in meters (m).
    skew_coefficient: Optional skew coefficient, (A1, ...., An, 1). If not
      given, it will be set to 0.

  Returns:
    A projection matrix, with shape (A1, ..., An, 4, 4), that follows the
      OpenGL convention.
  """
  fx, fy = focal_length[..., :1], focal_length[..., 1:]
  cx, cy = principal_point[..., :1], principal_point[..., 1:]

  zero = np.zeros_like(fx)
  ones = np.ones_like(fx)

  near = ones * near  # pyrefly: ignore[bad-assignment]
  far = ones * far  # pyrefly: ignore[bad-assignment]
  near_minus_far = near - far

  # The next two rows apply the camera intrinsics to the XY coordinates and map
  # the result to [-1, 1].
  row1 = [
      2.0 / width * ones * fx,
      -2.0 / width * skew_coefficient * ones,
      -(2 * cx / width - ones),
      zero,
  ]
  row2 = [zero, 2.0 / height * ones * fy, (2 * cy / height - ones), zero]

  # Maps z coordinate values from (-near, -far) to (-1, 1).
  row3 = [
      zero,
      zero,
      (far + near) / near_minus_far,
      2.0 * far * near / near_minus_far,
  ]
  row4 = [zero, zero, -ones, zero]

  view_matrix = np.concatenate(row1 + row2 + row3 + row4, axis=-1)

  output_shape = np.concatenate([focal_length.shape[:-1], [4, 4]], axis=0)
  output_shape = output_shape.astype(np.int32)
  view_matrix = np.reshape(view_matrix, output_shape)
  return view_matrix


def opencv_intrinsics_matrix_to_opengl_view_matrix(
    camera_to_image: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
) -> np.ndarray:
  """Converts OpenCV camera intrinsics to OpenGL view matrix.

  Args:
    camera_to_image: Camera intrinsics matrix, (A1, ..., An, 3, 3).
    width: Width of the image, in pixels.
    height: Height of the image, in pixels.
    near: Depth of the near clipping plane in meters (m).
    far: Depth of the far clipping plane in meters (m).

  Returns:
    A projection matrix, with shape (A1, ..., An, 4, 4), that follows the
    OpenGL convention.
  """
  focal_length = np.stack(
      [camera_to_image[..., 0, 0], camera_to_image[..., 1, 1]], axis=-1
  )
  principal_point = camera_to_image[..., :2, 2]
  skew_coefficient = camera_to_image[..., 0, 1]
  return opencv_intrinsics_to_opengl_view_matrix(
      focal_length,
      principal_point,
      width,
      height,
      near,
      far,
      skew_coefficient=skew_coefficient[..., np.newaxis],
  )


# TensorFlow implementations
def opencv_extrinsics_to_opengl_tf(extrinsics: tf.Tensor) -> tf.Tensor:
  """Converts OpenCV extrinsics to OpenGL using TensorFlow.

  Args:
    extrinsics: The camera extrinsics or world-to-camera transformation matrix,
      (A1, A2, ..., An, 4, 4).

  Returns:
    Camera extrinsics in OpenGL coordinate system, with the same shape as input.
  """
  return tf.einsum(
      'mk,...kn->...mn',
      tf.convert_to_tensor(OPENCV_TO_OPENGL, dtype=tf.float32),
      extrinsics,
  )


def opencv_intrinsics_matrix_to_opengl_view_matrix_tf(
    camera_to_image: tf.Tensor,
    width: int,
    height: int,
    near: float,
    far: float,
) -> tf.Tensor:
  """Converts OpenCV camera intrinsics to OpenGL view matrix.

  Args:
    camera_to_image: Camera intrinsics matrix, (A1, ..., An, 3, 3).
    width: Width of the image, in pixels.
    height: Height of the image, in pixels.
    near: Depth of the near clipping plane in meters (m).
    far: Depth of the far clipping plane in meters (m).

  Returns:
    A projection matrix, with shape (A1, ..., An, 4, 4), that follows the
      OpenGL convention.
  """
  fx = camera_to_image[..., 0, 0]
  fy = camera_to_image[..., 1, 1]
  focal_length = tf.stack([fx, fy], axis=-1)
  principal_point = camera_to_image[..., :2, 2]
  skew_coefficient = camera_to_image[..., 0, 1:2]
  return opencv_intrinsics_to_opengl_view_matrix_tf(
      focal_length,
      principal_point,
      width,
      height,
      near,
      far,
      skew_coefficient=skew_coefficient,
  )


def opencv_intrinsics_to_opengl_view_matrix_tf(
    focal_length: tf.Tensor,
    principal_point: tf.Tensor,
    width: int,
    height: int,
    near: float,
    far: float,
    skew_coefficient: tf.Tensor | None = None,
) -> tf.Tensor:
  """Converts OpenCV camera intrinsics to OpenGL view matrix using TensorFlow.

  See here for more information:
  https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

  Args:
    focal_length: Camera focal length, (A1, ...., An, 2).
    principal_point: Camera center, (A1, ...., An, 2).
    width: Width of the image.
    height: Height of the image.
    near: Depth of the near clipping plane.
    far: Depth of the far clipping plane.
    skew_coefficient: Optional skew coefficient, (A1, ...., An, 1). If not
      given, it will be set to 0.

  Returns:
    A projection matrix, with shape (A1, ..., An, 4, 4), that follows the
      OpenGL convention.
  """
  fx, fy = focal_length[..., :1], focal_length[..., 1:]
  cx, cy = principal_point[..., :1], principal_point[..., 1:]

  zero = tf.zeros_like(fx)
  ones = tf.ones_like(fx)

  if skew_coefficient is None:
    skew_coefficient = zero

  near = ones * near
  far = ones * far
  near_minus_far = near - far

  # The next two rows apply the camera intrinsics to the XY coordinates and map
  # the result to [-1, 1].
  row1 = [
      2.0 / width * ones * fx,
      -2.0 / width * skew_coefficient,  # pyrefly: ignore[unsupported-operation]
      -(2 * cx / width - ones),
      zero,
  ]
  row2 = [zero, 2.0 / height * ones * fy, (2 * cy / height - ones), zero]

  # Maps z coordinate values from (-near, -far) to (-1, 1).
  row3 = [
      zero,
      zero,
      (far + near) / near_minus_far,
      2.0 * far * near / near_minus_far,
  ]
  row4 = [zero, zero, -ones, zero]

  view_matrix = tf.concat(row1 + row2 + row3 + row4, axis=-1)

  output_shape = tf.concat([focal_length.shape[:-1], [4, 4]], axis=0)
  view_matrix = tf.reshape(view_matrix, output_shape)
  return view_matrix
