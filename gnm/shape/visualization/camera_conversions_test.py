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

"""Tests for camera_conversions."""

import os
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from etils import epath
from gnm.shape.visualization import camera_conversions
import mediapy as media
import numpy as np
from scipy.spatial import transform
import tensorflow as tf

try:
  from tensorflow_graphics.rendering import rasterization_backend
  from tensorflow_graphics.rendering import triangle_rasterizer

  _HAS_TF_GRAPHICS = True
except ImportError:
  _HAS_TF_GRAPHICS = False


_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(dtype=np.float32)

_opencv_intrinsics_to_opengl_view_matrix = (
    camera_conversions.opencv_intrinsics_to_opengl_view_matrix
)

_opencv_intrinsics_to_opengl_view_matrix_tf = (
    camera_conversions.opencv_intrinsics_to_opengl_view_matrix_tf
)


_opencv_extrinsics_to_opengl = camera_conversions.opencv_extrinsics_to_opengl

_opencv_extrinsics_to_opengl_tf = (
    camera_conversions.opencv_extrinsics_to_opengl_tf
)


_Rotation: TypeAlias = transform.Rotation

_OUTPUTS_DIR = epath.Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])

_RECTANGLE_COLOR = (00, 00, 255)


def _unproject_points(
    points2d: np.ndarray,
    depths: np.ndarray,
    focal_length: np.ndarray,
    principal_point: np.ndarray,
    skew: float = 0.0,
) -> np.ndarray:
  """Unprojects points using the provided intrinsics and depth."""

  intrinsics = np.eye(3)
  intrinsics[0, 0] = focal_length[0]
  intrinsics[1, 1] = focal_length[1]
  intrinsics[:2, 2] = principal_point
  intrinsics[0, 1] = skew

  points2d_h = np.concatenate(
      [points2d, np.ones_like(points2d[:, :1])], axis=-1
  )

  points_3d = np.einsum("mk,pk->pm", np.linalg.inv(intrinsics), points2d_h)
  points_3d *= depths

  return points_3d


class CameraConversionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.rng = np.random.default_rng(43)

    height, width = 1024, 1024
    self.image_size = (height, width)

    self.principal_point = np.array(
        [width * 0.5 - 0.0, height * 0.5 - 0.0], dtype=np.float32
    )
    self.focal_length = np.array([height, height], dtype=np.float32)

    self.offset = 20
    self.near = 0.01
    self.far = 20

    self.rectangle_depth = 0.5 * (self.near + self.far)
    self.rectangle_2d = np.array(
        [
            [self.offset, self.offset],
            [width - self.offset, self.offset],
            [width - self.offset, height - self.offset],
            [self.offset, height - self.offset],
        ],
        dtype=np.float32,
    )

    self.rectangle_3d = _unproject_points(
        self.rectangle_2d,
        np.full_like(self.rectangle_2d[:, :1], self.rectangle_depth),
        self.focal_length,
        self.principal_point,
    ).astype(np.float32)

    self.rectangle_triangles = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)

    self.vertex_colors = np.full_like(
        self.rectangle_3d, _RECTANGLE_COLOR, dtype=np.float32
    )
    self.vertex_colors /= 255

    top_left = self.rectangle_2d[0]
    bottom_right = self.rectangle_2d[2] - 1

    self.background_image = np.full(
        [height, width, 3], (255, 255, 255), dtype=np.uint8
    )
    self.cv2_rectangle_image = cv2.rectangle(
        self.background_image.copy(),
        tuple(top_left.astype(np.int32).tolist()),
        tuple(bottom_right.astype(np.int32).tolist()),
        _RECTANGLE_COLOR,
        cv2.FILLED,
    )
    media.write_image(
        _OUTPUTS_DIR / "opencv_rectangles.png", self.cv2_rectangle_image
    )

  @parameterized.parameters([
      {"skew": 0.0},
      {"skew": 2000.0},
  ])
  def test_opencv_to_opengl_intrinsics_rectangle(self, skew: float):
    """Tests that rendering with the converted view matrix works."""
    height, width = self.image_size

    view_matrix = _opencv_intrinsics_to_opengl_view_matrix(
        self.focal_length,
        self.principal_point,
        width,
        height,
        self.near,
        self.far,
        skew,
    )
    rectangle_3d = _unproject_points(
        self.rectangle_2d,
        np.full_like(self.rectangle_2d[:, :1], self.rectangle_depth),
        self.focal_length,
        self.principal_point,
        skew,
    ).astype(np.float32)

    view_projection_matrix = view_matrix @ _OPENCV_TO_OPENGL

    rendered_image = self._rasterize_rectangle(
        rectangle_3d, view_projection_matrix
    )

    outputs_dir = _OUTPUTS_DIR / f"skew={skew:.2f}_intrinsics"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    media.write_image(
        outputs_dir / "opencv_rectangles.png", self.cv2_rectangle_image
    )

    media.write_image(outputs_dir / "opengl_rectangles.png", rendered_image)

    diff = rendered_image - self.cv2_rectangle_image
    abs_diff_image = np.abs(diff)
    media.write_image(outputs_dir / "diff.png", abs_diff_image)

    mean_sq_diff = np.mean(np.square(diff).mean() / 255)

    self.assertLess(mean_sq_diff, 1.0e-03)

  @parameterized.parameters([
      ((1,),),
      ((5,),),
      ((5, 4),),
      ((5, 4, 3),),
      ((5, 4, 3, 2),),
      ((5, 4, 3, 2, 1),),
  ])
  def test_opencv_to_opengl_shape(self, batch_dims: tuple[int, ...]):
    height, width = 256, 256

    focal_length = np.full(batch_dims + (2,), height, dtype=np.float32)
    principal_point = np.full(
        batch_dims + (2,), [width * 0.5, height * 0.5], dtype=np.float32
    )
    # 1 cm and 20 meters.
    near, far = 0.01, 20

    opengl_view_matrix = _opencv_intrinsics_to_opengl_view_matrix(
        focal_length,
        principal_point,
        width=width,
        height=height,
        near=near,
        far=far,
    )

    with self.subTest("Shape is correct"):
      self.assertSequenceEqual(opengl_view_matrix.shape, batch_dims + (4, 4))

  def test_opencv_to_opengl_extrinsics(self):
    """Tests that the extrinsics are correctly transformed."""

    # Build a camera-to-world transformation.
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = _Rotation.from_rotvec([1, 2, 3]).as_matrix()
    camera_to_world[:3, 3] = [100, 200, 300]

    # Transform the rectangle points from the camera to the world coordinate
    # frame.
    rectangle_3d_world = np.einsum(
        "mn,bn->bm", camera_to_world[:3, :3], self.rectangle_3d
    )
    rectangle_3d_world += camera_to_world[:3, 3]

    # Compute the world-to-camera transformation it and make it compatible with
    # OpenGL.
    world_to_camera = np.linalg.inv(camera_to_world)
    world_to_camera_opengl = camera_conversions.opencv_extrinsics_to_opengl(
        world_to_camera
    )

    height, width = self.image_size
    view_matrix = _opencv_intrinsics_to_opengl_view_matrix(
        self.focal_length,
        self.principal_point,
        width,
        height,
        self.near,
        self.far,
    )
    view_projection_matrix = view_matrix @ world_to_camera_opengl

    rendered_image = self._rasterize_rectangle(
        rectangle_3d_world, view_projection_matrix
    )

    outputs_dir = _OUTPUTS_DIR / "extrinsics"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    media.write_image(outputs_dir / "opengl_rectangles.png", rendered_image)

    diff = rendered_image - self.cv2_rectangle_image
    abs_diff_image = np.abs(diff)
    media.write_image(outputs_dir / "diff.png", abs_diff_image)

    mean_sq_diff = np.mean(np.square(diff).mean() / 255)

    self.assertLess(mean_sq_diff, 1.0e-03)

  def test_opengl_to_opencv_intrinsics_consistency(self):
    """Tests that the OpenGL to OpenCV conversion is consistent."""
    opencv_camera_matrix = np.eye(4, dtype=np.float32)
    opencv_camera_matrix[:2, :2] = np.diag(self.focal_length)
    opencv_camera_matrix[:2, 2] = self.principal_point

    height, width = self.image_size
    as_opengl = (
        camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
            opencv_camera_matrix, height=height, width=width, near=0.01, far=20
        )
    )

    as_opencv = camera_conversions.opengl_intrinsics_to_opencv_matrix(
        as_opengl, height=height, width=width
    )

    np.testing.assert_allclose(as_opencv, opencv_camera_matrix)

  # TF Parity Tests

  @parameterized.parameters([
      ((1,),),
      ((5,),),
      ((5, 4),),
      ((5, 4, 3),),
      ((5, 4, 3, 2),),
      ((5, 4, 3, 2, 1),),
  ])
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

  @parameterized.parameters([
      ((1,),),
      ((5,),),
      ((5, 4),),
      ((5, 4, 3),),
      ((5, 4, 3, 2),),
      ((5, 4, 3, 2, 1),),
  ])
  def test_opencv_to_opengl_extrinsics_tf_parity(
      self, batch_dims: tuple[int, ...]
  ):
    """Tests that the TensorFlow implementation matches the NumPy one."""
    rotations = _Rotation.random(int(np.prod(batch_dims)), self.rng).as_matrix()
    rotations = np.reshape(rotations, batch_dims + (3, 3))

    translation = self.rng.uniform(-2, 2, size=batch_dims + (3,))

    extrinsics = np.expand_dims(np.eye(4), tuple(range(len(batch_dims))))
    extrinsics = np.tile(extrinsics, batch_dims + (1, 1))
    extrinsics[..., :3, :3] = rotations[..., :3, :3]
    extrinsics[..., 3, :3] = translation

    opengl_extrinsics_np = _opencv_extrinsics_to_opengl(extrinsics)
    opengl_extrinsics_tf = _opencv_extrinsics_to_opengl_tf(
        tf.convert_to_tensor(extrinsics, dtype=tf.float32)
    )

    np.testing.assert_allclose(
        opengl_extrinsics_np, opengl_extrinsics_tf.numpy()
    )

  def _rasterize_rectangle(
      self,
      rectangle_3d: np.ndarray,
      view_projection_matrix: np.ndarray,
  ) -> np.ndarray:
    """Rasterizes rectangle coordinates with a projection matrix."""
    if not _HAS_TF_GRAPHICS:
      self.skipTest("tensorflow_graphics not available on this platform.")
    buffers = triangle_rasterizer.rasterize(
        rectangle_3d,
        self.rectangle_triangles,
        {"vertex_colors": self.vertex_colors},
        view_projection_matrix=view_projection_matrix,
        image_size=self.image_size,
        backend=rasterization_backend.RasterizationBackends.CPU,
    )
    buffers = {k: np.flipud(v.numpy()) for k, v in buffers.items()}
    foreground = buffers["mask"] > 0
    rendered_colors = (buffers["vertex_colors"] * 255).astype(np.uint8)
    rendered_image = np.where(
        foreground, rendered_colors, self.background_image
    )

    return rendered_image


if __name__ == "__main__":
  absltest.main()
