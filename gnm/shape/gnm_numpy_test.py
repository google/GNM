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

"""Tests for GNM NumPy implementation."""

# pylint: disable=protected-access

from collections.abc import Sequence
import copy
import itertools
import re

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_data_schema
from gnm.shape import gnm_numpy
from gnm.shape import gnm_utils
from gnm.shape.data.versions import gnm_catalog
import numpy as np
from scipy.spatial import transform as transform_module
from tensorflow_graphics.geometry.representation.mesh import normals as tf_normals
import trimesh
from trimesh import transformations

_SUPPORTED_VARIANTS = frozenset([
    gnm_numpy.GNMVariant.HEAD,
])

_Rotation = transform_module.Rotation

_INVALID_SUFFIXES = []


_MAINTAINED_MAJOR_GNM_VERSIONS = gnm_catalog.MAINTAINED_MAJOR_VERSIONS
_MAJOR_VERSION_TO_VARIANTS_MAP = gnm_catalog.MAJOR_VERSION_TO_VARIANTS_MAP


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
  """Applies a 4x4 transform to 3D points.

  Args:
    transform: A 4x4 transformation matrix.
    points: 3D point data, shaped (N, 3).

  Returns:
    The transformed points, shaped (N, 3).
  """
  points_transformed_h = np.insert(points, 3, 1.0, 1).dot(transform.T)
  return points_transformed_h[:, :-1] / points_transformed_h[:, -1:]


def get_pose_correctives_test_cases(gnm_np: gnm_numpy.GNM):
  rotations = np.arange(gnm_np.num_joints * 3, dtype=np.float32).reshape(
      gnm_np.num_joints, 3
  )
  no_correctives = np.zeros([gnm_np.num_vertices, 3])

  rotation_matrices = _Rotation.from_rotvec(rotations).as_matrix()

  test_cases = [{
      'pose_correctives_regressor': None,
      'rotations': rotations,
      'expected_pose_correctives': no_correctives,
  }]

  pose_correctives_regressor = np.ones(
      [gnm_np.num_vertices * 3, gnm_np.num_joints * 9], dtype=np.float32
  )
  expected_pose_correctives = pose_correctives_regressor.dot(
      (rotation_matrices - np.eye(3)[None]).reshape(-1)
  )
  expected_pose_correctives = expected_pose_correctives.reshape(
      gnm_np.num_vertices, 3
  )
  test_cases.append({
      'pose_correctives_regressor': pose_correctives_regressor.T,
      'rotations': rotations,
      'expected_pose_correctives': expected_pose_correctives,
  })

  return test_cases


def get_joints_with_identity_thresholds(gnm_np: gnm_numpy.GNM):
  del gnm_np
  return 1e-3


def get_group_subsets_test_cases():
  cases = []
  for version in _MAINTAINED_MAJOR_GNM_VERSIONS:
    for group in ['skin', 'right_eye', 'lower_teeth']:
      cases.append(dict(version=version, variant='head', group_name=group))
    for group in ['skin', 'head', 'right_hand']:
      cases.append(dict(version=version, variant='body', group_name=group))
  return cases


def get_eyeball_test_cases():
  cases = []
  for version in _MAINTAINED_MAJOR_GNM_VERSIONS:
    for variant in ['head', 'body']:
      if variant == 'head':
        joints = ['left_eye', 'right_eye']
      elif variant == 'body':
        joints = ['L_Eye', 'R_Eye']
      else:
        continue
      for joint in joints:
        cases.append(dict(version=version, variant=variant, joint_name=joint))
  return cases


class GNMNumpyTest(parameterized.TestCase):
  gnms: dict[str, dict[str, gnm_numpy.GNM]]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gnms = {}
    for version in _MAINTAINED_MAJOR_GNM_VERSIONS:
      cls.gnms[version] = {}
      for variant in _MAJOR_VERSION_TO_VARIANTS_MAP[version]:
        if variant in _SUPPORTED_VARIANTS:
          cls.gnms[version][variant] = gnm_numpy.GNM.from_local(
              gnm_numpy.GNMMajorVersion(version.removeprefix('v')),
              gnm_numpy.GNMVariant(variant),
          )

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def _get_default_kwargs(
      self, gnm_np: gnm_numpy.GNM, batch_dims: Sequence[int] = tuple([])
  ) -> dict[str, np.ndarray]:
    """Arguments for zero identity, expression, rotation, and translation."""
    batch_dims = tuple(batch_dims)
    return {
        'identity': np.zeros(batch_dims + (gnm_np.identity_dim,)),
        'expression': np.zeros(batch_dims + (gnm_np.expression_dim,)),
        'rotations': np.zeros(batch_dims + (gnm_np.num_joints, 3)),
        'translation': np.zeros(batch_dims + (3,)),
    }

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_global_translation(self, version: str, variant: str):
    """Test that global translation shifts joints and vertices uniformly."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')

    gnm_np = self.gnms[version][variant]
    translation = np.array([1, 2, 3], dtype=np.float32)
    kwargs = self._get_default_kwargs(gnm_np)
    desired = gnm_np(**kwargs) + translation
    actual = gnm_np(**(kwargs | {'translation': translation}))
    np.testing.assert_allclose(actual, desired, atol=1e-5)

    desired_joint_transforms = gnm_np.get_posed_joint_transforms(
        identity=kwargs['identity'],
        rotations=kwargs['rotations'],
        translation=kwargs['translation'],
    )
    desired_joint_transforms[:, :3, 3] += translation
    actual_joint_transforms = gnm_np.get_posed_joint_transforms(
        identity=kwargs['identity'],
        rotations=kwargs['rotations'],
        translation=translation,
    )
    np.testing.assert_allclose(
        actual_joint_transforms, desired_joint_transforms, atol=1e-5
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_global_rotation(self, version: str, variant: str):
    """Test that a root joint rotation is correctly modelled."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    kwargs = self._get_default_kwargs(gnm_np)
    vertices_zero_pose = gnm_np(**kwargs)

    # Only set the first element of rotations parameter, controlling the root.
    rotations = kwargs['rotations'].copy()
    rotations[0] = np.array([1, 2, 3])

    angle = np.linalg.norm(rotations[0])
    direction = rotations[0] / angle

    # Build a matching rotation matrix to the one GNM should be using.
    r_matrix = transformations.rotation_matrix(angle, direction)

    # Determine location of the root joint.
    root_joint = gnm_np.joint_positions_bind_pose(kwargs['identity'])[0]
    t_matrix = transformations.translation_matrix(root_joint)
    t_matrix_inv = transformations.translation_matrix(-root_joint)

    # A rotation about the root joint's position.
    matrix = t_matrix @ r_matrix @ t_matrix_inv

    desired = transform_points(matrix, vertices_zero_pose)
    actual = gnm_np(**(kwargs | {'rotations': rotations}))

    np.testing.assert_allclose(actual, desired, atol=1e-5)

    default_joint_transforms = gnm_np.get_posed_joint_transforms(
        identity=kwargs['identity'],
        rotations=kwargs['rotations'],
        translation=kwargs['translation'],
    )

    # Apply the rotation transformation on all joints transformations.
    desired_joint_transforms = np.einsum(
        'mk,jkn->jmn', matrix, default_joint_transforms
    )
    actual_joint_transforms = gnm_np.get_posed_joint_transforms(
        identity=kwargs['identity'],
        rotations=rotations,
        translation=kwargs['translation'],
    )
    np.testing.assert_allclose(
        actual_joint_transforms, desired_joint_transforms, atol=1e-5
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape(self, version: str, variant: str):
    """Badly shaped parameter should throw a ValueError."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    bad_dimension = (
        gnm_np.expression_dim + gnm_np.identity_dim + gnm_np.num_joints
    )
    bad_input = np.zeros(bad_dimension)
    kwargs = self._get_default_kwargs(gnm_np)
    for key in kwargs:
      with self.assertRaises(ValueError):
        gnm_np(**(kwargs | {key: bad_input}))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_bad_shape_joint_transforms(self, version: str, variant: str):
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    bad_dimension = (
        gnm_np.expression_dim + gnm_np.identity_dim + gnm_np.num_joints
    )
    bad_inputs = [
        {'identity': np.zeros(bad_dimension)},
        {'rotations': np.zeros([gnm_np.num_joints + 1, 3])},
        {'rotations': np.zeros([gnm_np.num_joints, 4])},
        {'translation': np.zeros([4])},
    ]
    joint_transform_kwargs = self._get_default_kwargs(gnm_np)
    joint_transform_kwargs.pop('expression')

    for bad_input_dict in bad_inputs:
      with self.assertRaises(ValueError):
        gnm_np.get_posed_joint_transforms(
            **(joint_transform_kwargs | bad_input_dict)
        )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_identity(self, version: str, variant: str):
    """Test that vertices are different after applying the identity model."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    kwargs = self._get_default_kwargs(gnm_np)
    verts_zero_identity = gnm_np(**kwargs)
    identity = np.ones(gnm_np.identity_dim)
    verts_with_identity = gnm_np(**kwargs | {'identity': identity})
    self.assertTrue(np.all(verts_zero_identity != verts_with_identity))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
  )
  def test_expression(self, version: str, variant: str):
    """Test that vertices are different after applying the expression model."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    kwargs = self._get_default_kwargs(gnm_np)
    verts_zero_expression = gnm_np(**kwargs)
    expression = np.ones(gnm_np.expression_dim)
    verts_with_expression = gnm_np(**kwargs | {'expression': expression})
    self.assertTrue(np.any(verts_zero_expression != verts_with_expression))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=(gnm_numpy.GNMVariant.HEAD.value,),
      batch_size=[(), (2,), (2, 3)],
  )
  def test_vertices_and_landmarks(
      self, version: str, variant: str, batch_size: tuple[int, ...]
  ):
    """Test extracting vertices and landmarks."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    kwargs = self._get_default_kwargs(gnm_np, batch_dims=batch_size)
    verts, landmarks = gnm_np.vertices_and_landmarks(
        gnm_numpy.GNMLandmarksType.HEAD_SPARSE_68, **kwargs
    )
    self.assertEqual(verts.shape, (*batch_size, gnm_np.num_vertices, 3))
    self.assertEqual(landmarks.shape, (*batch_size, 68, 3))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_vertex_groups_exist(self, version: str, variant: str):
    """Tests that there is at least one vertex group defined."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    self.assertNotEmpty(self.gnms[version][variant].vertex_group_names)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_vertex_groups(self, version: str, variant: str):
    """Tests that we can retrieve values and indices for each vertex group."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]

    # Skip the finger tips, since they have zero weights.
    valid_group_names = [
        n
        for n in gnm_np.vertex_group_names
        if not any([n.lower().endswith(suffix) for suffix in _INVALID_SUFFIXES])
    ]

    if not valid_group_names:
      self.skipTest(f'No valid vertex groups for {version}/{variant}')

    # Concatenate all valid vertex groups into a large 2D array.
    all_groups = np.stack([gnm_np.vertex_group(n) for n in valid_group_names])

    # Each value in the group should be in the range [0, 1].
    self.assertTrue(np.all(0 <= all_groups) & np.all(all_groups <= 1.0))

    # Each vertex group should have at least one non-zero vertex.
    self.assertTrue(np.all(np.max(all_groups, axis=1) > 1e-4))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_uvs(self, version: str, variant: str):
    """Tests that texture coordinates have the correct format."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    self.assertEqual(gnm.quad_uvs.shape[:2], gnm.quads.shape)
    self.assertEqual(gnm.triangle_uvs.shape[:2], gnm.triangles.shape)
    self.assertEqual(gnm.vertex_uvs.shape, (gnm.num_vertices, 2))

    # All UV coordinates should be between [0, 1].
    self.assertTrue(np.all(0 <= gnm.quad_uvs) & np.all(gnm.quad_uvs <= 1.0))
    self.assertTrue(
        np.all(0 <= gnm.triangle_uvs) & np.all(gnm.triangle_uvs <= 1.0)
    )
    self.assertTrue(np.all(0 <= gnm.vertex_uvs) & np.all(gnm.vertex_uvs <= 1.0))

    # Check that each set of quad UVs matches two sets of triangle UVs.
    for i, quad_uvs in enumerate(gnm.quad_uvs):
      triangle_uvs_1 = gnm.triangle_uvs[i]
      triangle_uvs_2 = gnm.triangle_uvs[len(gnm.quads) + i]
      np.testing.assert_allclose(quad_uvs[:3], triangle_uvs_1)
      np.testing.assert_allclose(quad_uvs[np.array([2, 3, 0])], triangle_uvs_2)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_edge_list(self, version: str, variant: str):
    """Checks that the edge list matches the quad topology."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]

    # We build two adjacency matrices, one for the quads, and one for the edge
    # list, and check they are the same.
    num_vertices = gnm_np.num_vertices
    adjacency_matrix_quads = np.zeros((num_vertices, num_vertices), dtype=bool)
    adjacency_matrix_edge_list = adjacency_matrix_quads.copy()

    for v1, v2, v3, v4 in gnm_np.quads:
      adjacency_matrix_quads[v1, v2] = adjacency_matrix_quads[v2, v1] = True
      adjacency_matrix_quads[v2, v3] = adjacency_matrix_quads[v3, v2] = True
      adjacency_matrix_quads[v3, v4] = adjacency_matrix_quads[v4, v3] = True
      adjacency_matrix_quads[v4, v1] = adjacency_matrix_quads[v1, v4] = True

    for v1, v2 in gnm_np.edge_list:
      adjacency_matrix_edge_list[v1, v2] = True

    np.testing.assert_array_equal(
        adjacency_matrix_quads, adjacency_matrix_edge_list
    )

  @parameterized.parameters(get_group_subsets_test_cases())
  def test_group_subsets(self, version: str, variant: str, group_name: str):
    """Tests that the convenience functions for accessing data subsets work."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]

    group_indices = gnm_np.vertex_group_indices(group_name)
    triangles = gnm_np.triangles_group(group_name)
    triangle_uvs = gnm_np.triangle_uvs_group(group_name)

    # NumPy optimized vectorized subset check.
    mask = np.zeros(gnm_np.num_vertices, dtype=bool)
    mask[group_indices] = True
    self.assertTrue(np.all(mask[triangles]))

    quads = gnm_np.quads_group(group_name)
    quad_uvs = gnm_np.quad_uvs_group(group_name)
    self.assertTrue(np.all(mask[quads]))

    self.assertEqual(
        len(group_indices),
        len(gnm_np.vertex_uvs_group(group_name)),
    )
    self.assertEqual(len(triangles), len(triangle_uvs))
    self.assertEqual(len(quads), len(quad_uvs))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_parts_are_non_overlapping_and_complete(
      self, version: str, variant: str
  ):
    """Check that separate parts are non-overlapping and complete GNM."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    combined = np.stack([gnm.vertex_group(n) for n in gnm.mesh_component_names])
    np.testing.assert_array_equal(
        np.sum(combined, axis=0), np.ones(gnm.num_vertices)
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_mesh_component_names_include_skin(self, version: str, variant: str):
    """Check that the mesh_component_names include 'skin'."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    self.assertIn('skin', gnm.mesh_component_names)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_vertex_group_names_include_skin(self, version: str, variant: str):
    """Check that the vertex_group_names include 'skin'."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    self.assertIn('skin', gnm.vertex_group_names)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
  )
  def test_mirror_indices(self, version: str, variant: str):
    """Checks we have one mirror index for each vertex.

    See mirror_indices_test.
    """
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    self.assertCountEqual(gnm.mirror_indices, range(gnm.num_vertices))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
      vertex_group=['left_eye', 'right_eye'],
  )
  def test_eyes_dont_move(self, version: str, variant: str, vertex_group: str):
    """Eyes should not move even if every expression shape is applied."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    indices = gnm_np.vertex_group_indices(vertex_group)
    all_expressions_no_pupil = np.ones(gnm_np.expression_dim)
    regions = gnm_utils.expression_to_regions(all_expressions_no_pupil, gnm_np)
    regions['eyeballs'][:] = 0.0
    all_expressions_no_pupil = gnm_utils.regions_to_expression(regions, gnm_np)
    all_expressions_applied = gnm_np(expression=all_expressions_no_pupil)
    deltas = all_expressions_applied - gnm_np.template_vertex_positions
    np.testing.assert_allclose(0.0, deltas[indices], atol=1e-6)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_joints_with_identity_dont_match_template(
      self, version: str, variant: str
  ):
    """Given a random identity, joints should _not_ match the template."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]

    thresholds = get_joints_with_identity_thresholds(gnm_np)

    identity = self.rng.uniform(1, 2, gnm_np.identity_dim)
    joints_bind_pose = gnm_np.joint_positions_bind_pose(identity)
    deltas = gnm_np.template_joint_positions - joints_bind_pose
    self.assertTrue(np.all(np.linalg.norm(deltas, axis=-1) >= thresholds))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
      side=['&left', '&right'],  # Note '&' for group intersection.
  )
  def test_eyeball_interior_is_inside_exterior(
      self, version: str, variant: str, side: str
  ):
    """Tests eyeball interior is inside exterior for template and identities."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    interior_indices = gnm_np.vertex_group_indices('eyeball_interior', side)
    exterior_triangles = gnm_np.triangles_group('eyeball_exterior', side)

    # Pose GNM vertices for template and several identity dimensions.
    dims_to_test = 3
    test_names = ['template']
    identity = np.zeros((dims_to_test * 2 + 1, gnm_np.identity_dim))
    for i in range(dims_to_test):
      identity[i * 2 + 1 : i * 2 + 3, i] = [2, -2]
      test_names.extend([f'identity_{i}_positive', f'identity_{i}_negative'])

    all_vertices = gnm_np(identity=identity)

    for test_name, vertices in zip(test_names, all_vertices):

      # The points to check are the vertices of the interior group.
      interior_points = vertices[interior_indices]

      # Build a trimesh from the exterior group.
      exterior_mesh = trimesh.Trimesh(
          vertices=vertices, faces=exterior_triangles, process=False
      )

      # Check if all interior points are contained within the exterior mesh.
      with self.subTest(test_name):
        self.assertTrue(np.all(exterior_mesh.contains(interior_points)))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
  )
  def test_expression_basis_shape(self, version: str, variant: str):
    """Tests the expression basis is accessible and has the expected shape."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    self.assertEqual(
        gnm_np.expression_basis.shape,
        (gnm_np.expression_dim, gnm_np.num_vertices, 3),
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
      vertex_group=['left_pupil', 'right_pupil'],
  )
  def test_pupil_uvs_close_to_middle(
      self, version: str, variant: str, vertex_group: str
  ):
    """Tests the pupil UV coordinates are close to (0.5, 0.5)."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    uv = gnm_np.vertex_uvs_group(vertex_group)[0]
    self.assertLess(np.linalg.norm(uv - 0.5), 2e-3)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
      side=['left', 'right'],
      group=['eyeball_interior', 'eyeball_exterior'],
      axis=[0, 1],
  )
  def test_eyeball_uvs_not_flipped(
      self, version: str, variant: str, side: str, group: str, axis: int
  ):
    """Checks that model-space axes align with UV axes for eyeballs."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    side_indices = gnm_np.vertex_group_indices(side)
    group_indices = gnm_np.vertex_group_indices(group)
    side_group_indices = np.intersect1d(side_indices, group_indices)
    vertices = gnm_np.template_vertex_positions[:, axis]
    min_index = side_group_indices[np.argmin(vertices[side_group_indices])]
    max_index = side_group_indices[np.argmax(vertices[side_group_indices])]
    self.assertGreater(vertices[max_index], vertices[min_index])
    uvs = gnm_np.vertex_uvs[:, axis]
    self.assertGreater(uvs[max_index], uvs[min_index])

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_omit_all_parameters(self, version: str, variant: str):
    """If all parameters are omitted, GNM returns the template vertices."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    expected = gnm.template_vertex_positions
    np.testing.assert_allclose(expected, gnm(), atol=1e-6)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
      batch_dims=[[], [1], [1, 2], [2, 1, 2]],
      parameter_count=[1, 2, 3, 4],
  )
  def test_variable_batch_and_omitted_parameters(
      self,
      version: str,
      variant: str,
      batch_dims: list[int],
      parameter_count: int,
  ):
    """Exercise GNM with various batch dimensions and omitted parameters."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    parameters = self._get_default_kwargs(gnm, batch_dims)

    # Omit some parameters.
    for keys in itertools.combinations(parameters, parameter_count):
      sub_parameters = {k: parameters[k] for k in keys}
      gnm(**sub_parameters)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_pose_correctives(self, version: str, variant: str):
    """Test that the pose correctives are correct."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]

    test_cases = get_pose_correctives_test_cases(gnm_np)
    for case in test_cases:
      gnm = copy.deepcopy(gnm_np)
      gnm.pose_correctives_regressor = case['pose_correctives_regressor']

      pose_correctives = gnm.compute_pose_correctives(case['rotations'])

      np.testing.assert_allclose(
          pose_correctives, case['expected_pose_correctives'], atol=1e-6
      )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(
          v.value for v in _SUPPORTED_VARIANTS if 'hand' not in v.value
      ),
  )
  def test_multiple_vertex_groups(self, version: str, variant: str):
    """Test we can combine multiple vertex groups."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    # Compare two ways of combining both eyeballs fully.
    np.testing.assert_array_equal(
        gnm.vertex_group_indices('left_eye', 'right_eye'),
        gnm.vertex_group_indices('eyeball_interior', 'eyeball_exterior'),
    )

    # Check trivial set operations.
    self.assertEmpty(gnm.vertex_group_indices('left', '-left'))
    np.testing.assert_equal(gnm.vertex_group_mask('~left', 'left'), True)
    np.testing.assert_equal(gnm.vertex_group_mask('left', '~left'), True)
    np.testing.assert_equal(
        gnm.vertex_group_mask('~left'), ~gnm.vertex_group_mask('left')
    )

    # Check that left_eye and right_eye are disjoint.
    self.assertEmpty(gnm.vertex_group_indices('left_eye', '&right_eye'))

    # Exercise other vertex-group related functions with multiple groups.
    groups = ('hockey_mask', 'left_ear', 'right_ear', '&left')
    gnm.quad_indices_for_group(*groups)
    gnm.triangle_indices_for_group(*groups)
    gnm.quads_group(*groups)
    gnm.triangles_group(*groups)
    gnm.quad_uvs_group(*groups)
    gnm.triangle_uvs_group(*groups)
    gnm.vertex_uvs_group(*groups)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=('head',),
  )
  def test_left_and_right_have_equal_number_of_vertices(
      self, version: str, variant: str
  ):
    """Test the left and right groups have the same number of vertices."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    self.assertEqual(
        len(gnm.vertex_group_indices('left')),
        len(gnm.vertex_group_indices('right')),
    )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=('head',),
  )
  def test_face_identity_names_contain_all_groups(
      self, version: str, variant: str
  ):
    """Tests that the identity names contain all expected groups."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    # Only check for the groups existing in the GNM model.
    expected_groups = gnm_utils.identity_region_names(gnm)
    regexes = [f'^{n}_[0-9][0-9][0-9]$' for n in expected_groups]

    region_indices = []
    for regex in regexes:
      matcher = np.vectorize(lambda x: bool(re.match(regex, x)))  # pylint: disable=cell-var-from-loop
      region_indices.append(np.where(matcher(gnm.identity_names))[0])

    with self.subTest('Identity regions are nonempty with no duplicates.'):
      for indices in region_indices:
        self.assertNotEmpty(indices)
        np.testing.assert_array_equal(indices, np.unique(indices))

    with self.subTest('Identity regions have expected combined length.'):
      self.assertEqual(
          sum([region.size for region in region_indices]), gnm.identity_dim
      )
    with self.subTest('Identity regions exactly cover the identity basis.'):
      np.testing.assert_array_equal(
          np.unique(np.concatenate(region_indices)),
          np.arange(gnm.identity_dim),
      )

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=('head',),
  )
  def test_face_expression_names_contain_all_groups(
      self, version: str, variant: str
  ):
    """Tests that the expression names contain all expected groups."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    regexes = [
        f'^{n}_[0-9][0-9][0-9]$'
        for n in ['left_eye', 'right_eye', 'mouth', 'tongue', 'eyeballs']
    ] + ['^tongue_mean$']
    region_indices = []
    for regex in regexes:
      matcher = np.vectorize(lambda x: bool(re.match(regex, x)))  # pylint: disable=cell-var-from-loop
      region_indices.append(np.where(matcher(gnm.expression_names))[0])

    with self.subTest('Expression regions are nonempty with no duplicates.'):
      for indices in region_indices:
        self.assertNotEmpty(indices)
        np.testing.assert_array_equal(indices, np.unique(indices))

    with self.subTest('Expression regions have expected combined length.'):
      self.assertEqual(
          sum([region.size for region in region_indices]), gnm.expression_dim
      )
    with self.subTest('Expression regions exactly cover the expression basis.'):
      np.testing.assert_array_equal(
          np.unique(np.concatenate(region_indices)),
          np.arange(gnm.expression_dim),
      )

  @parameterized.product(
      batch_dims=[
          [],
          [4],
          [10, 4],
          [10, 64],
          [1],
          [1, 2],
          [2, 1, 2],
          [5, 4, 3, 2, 1],
      ],
      dtype=[np.float32, np.float64],
  )
  def test_axis_angle_to_rotation_matrix(
      self, batch_dims: Sequence[int], dtype: np.dtype
  ):
    """Tests that axis-angle to rotation matrix conversion is correct."""
    axis_angle = _Rotation.random(int(np.prod(batch_dims)), rng=self.rng)

    expected_rotmats = (
        np.asarray(axis_angle.as_matrix())
        .reshape(*batch_dims, 3, 3)
        .astype(dtype)
    )
    actual_rotmats = gnm_numpy._rotation_matrix(
        np.asarray(axis_angle.as_rotvec(), dtype=dtype).reshape(*batch_dims, 3)
    )

    # Check that the dtype is not changed.
    self.assertEqual(actual_rotmats.dtype, expected_rotmats.dtype)
    np.testing.assert_allclose(actual_rotmats, expected_rotmats, atol=1e-6)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_adding_and_removing_vertex_group_names_does_not_raise_error(
      self, version: str, variant: str
  ):
    """Tests that adding and removing vertex group names does not raise."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]
    gnm.vertex_group_names = [*gnm.vertex_group_names, 'new_group']
    gnm.vertex_group_names = gnm.vertex_group_names[:-1]

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_add_vertex_group(self, version: str, variant: str):
    """Tests adding a new vertex group."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    new_group_name = 'test_new_group'
    new_group_value = np.random.uniform(size=(gnm.num_vertices,))

    # Make a copy to avoid modifying the cached model for other tests!
    gnm_copy = copy.deepcopy(gnm)

    gnm_copy.add_vertex_group(new_group_name, new_group_value)

    self.assertIn(new_group_name, gnm_copy.vertex_group_names)
    np.testing.assert_array_equal(
        gnm_copy.vertex_group(new_group_name), new_group_value
    )

    # Test adding duplicate name raises error
    with self.assertRaises(ValueError):
      gnm_copy.add_vertex_group(new_group_name, new_group_value)

    # Test adding wrong shape raises error
    with self.assertRaises(ValueError):
      gnm_copy.add_vertex_group('another_group', np.zeros((10,)))

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=tuple(_SUPPORTED_VARIANTS),
  )
  def test_to_numpy_data_dict(self, version: str, variant: str):
    """Tests to_numpy_data_dict method."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

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
  def test_from_gnm(self, version: str, variant: str):
    """Tests from_gnm factory method."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm = self.gnms[version][variant]

    new_gnm = gnm_numpy.GNM.from_gnm(gnm)

    self.assertEqual(new_gnm.version, gnm.version)
    self.assertEqual(new_gnm.variant, gnm.variant)

    parameters = self._get_default_kwargs(gnm)
    vertices_orig = gnm(**parameters)
    vertices_new = new_gnm(**parameters)
    np.testing.assert_allclose(vertices_orig, vertices_new, atol=1e-5)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=('head',),
      batch_dims=[(5,), tuple()],
  )
  def test_compute_vertex_normals(
      self, version: str, variant: str, batch_dims: Sequence[int]
  ):
    """Tests that the vertex normals are computed correctly."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    parameters = self._get_default_kwargs(gnm_np, batch_dims=batch_dims)
    vertices = gnm_np(**parameters)
    vertex_normals = gnm_np.compute_vertex_normals(vertices)

    triangles_batch = np.broadcast_to(
        gnm_np.triangles, (*batch_dims, *gnm_np.triangles.shape)
    ).astype(np.int32)
    vertex_normals_tf = tf_normals.vertex_normals(
        vertices=vertices,
        indices=triangles_batch,
    ).numpy()
    np.testing.assert_allclose(vertex_normals, vertex_normals_tf, atol=1e-6)

  @parameterized.product(
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
      variant=('head',),
  )
  def test_compute_vertex_normals_with_zero_magnitude(
      self, version: str, variant: str
  ):
    """Tests that a warning is logged for vertex normals with zero magnitude."""
    if variant not in self.gnms[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')
    gnm_np = self.gnms[version][variant]
    vertices = gnm_np.template_vertex_positions

    # Collapse the faces around vertex 0.
    tmesh = trimesh.Trimesh(vertices, gnm_np.triangles, process=False)
    adjacent_vertices = tmesh.vertex_neighbors[0]
    vertices[adjacent_vertices] = vertices[0]

    with self.assertLogs(level='WARNING') as log_output:
      gnm_np.compute_vertex_normals(vertices)
    self.assertIn('zero magnitude', log_output.output[0])


class GNMNumpyFactoryMethodsTest(parameterized.TestCase):
  """Tests for instantiating GNM using factory methods."""

  @parameterized.product(
      variant=tuple(_SUPPORTED_VARIANTS),
      version=_MAINTAINED_MAJOR_GNM_VERSIONS,
  )
  def test_from_local_successful(self, variant, version):
    if variant not in _MAJOR_VERSION_TO_VARIANTS_MAP[version]:
      self.skipTest(f'variant {variant} not supported in {version}.')

    # Convert string version/variant to Enums.
    major_version = gnm_numpy.GNMMajorVersion(version[1:])
    gnm_variant = gnm_numpy.GNMVariant(variant)

    model = gnm_numpy.GNM.from_local(major_version, gnm_variant)
    self.assertIsInstance(model, gnm_numpy.GNM)


if __name__ == '__main__':
  absltest.main()
