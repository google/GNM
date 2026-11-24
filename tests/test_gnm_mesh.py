"""Integration tests against the official GNM v3 Head model on CPU."""

import os
from pathlib import Path
import subprocess
import sys

from gnm.shape import gnm_pytorch
import numpy as np
from personalab.gnm_mesh import generate_mesh, load_head_model
import pytest
from scipy.spatial.transform import Rotation
import torch
import trimesh


GEOMETRY_ATOL = 2e-7  # meters, for float32 geometry compared with a reference
MIN_DEFORMATION = 1e-4  # meters


@pytest.fixture(scope="module")
def model_file():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE")


@pytest.fixture(scope="module")
def model(model_file):
    return load_head_model(model_file)


@pytest.fixture(scope="module")
def neutral(model):
    return generate_mesh(model)


def test_loads_v3_head_on_pytorch_cpu(model):
    assert isinstance(model, gnm_pytorch.GNM)
    assert model.version == gnm_pytorch.GNMVersion.V3_0
    assert model.variant == gnm_pytorch.GNMVariant.HEAD
    assert model.template_vertex_positions.device.type == "cpu"
    assert model.template_vertex_positions.dtype == torch.float32


def test_neutral_vertices_and_triangles(model, neutral):
    assert neutral.vertices.shape == (model.num_vertices, 3)
    assert np.isfinite(neutral.vertices).all()
    np.testing.assert_allclose(
        neutral.vertices, model.template_vertex_positions.numpy(),
        rtol=0, atol=GEOMETRY_ATOL,
    )
    assert neutral.faces.shape == model.triangles.shape
    assert neutral.faces.ndim == 2 and neutral.faces.shape[1] == 3
    assert np.issubdtype(neutral.faces.dtype, np.integer)
    assert 0 <= neutral.faces.min() <= neutral.faces.max() < model.num_vertices
    assert np.all(neutral.area_faces > 0)
    np.testing.assert_array_equal(neutral.faces, model.triangles.numpy())


def test_identity_changes_face_by_the_selected_basis(model, neutral):
    index = model.identity_names.index("head_000")
    identity = model.template_vertex_positions.new_zeros(model.identity_dim)
    identity[index] = 1.5

    changed = generate_mesh(model, identity=identity)
    delta = changed.vertices - neutral.vertices
    skin = model.vertex_group_indices("skin")
    assert np.linalg.norm(delta[skin], axis=1).max() > MIN_DEFORMATION
    expected = model.vertex_identity_basis[index].numpy() * 1.5
    np.testing.assert_allclose(delta, expected, rtol=0, atol=GEOMETRY_ATOL)
    np.testing.assert_array_equal(changed.faces, neutral.faces)


def test_lower_face_expression_moves_lips_and_leaves_eyes_in_place(model, neutral):
    index = model.expression_names.index("lower_face_region_000")
    expression = model.template_vertex_positions.new_zeros(model.expression_dim)
    expression[index] = 2.0

    changed = generate_mesh(model, expression=expression)
    delta = changed.vertices - neutral.vertices
    lips = model.vertex_group_indices("upper_lip", "lower_lip")
    eyes = model.vertex_group_indices("eye_interiors")
    assert np.linalg.norm(delta[lips], axis=1).max() > MIN_DEFORMATION
    np.testing.assert_allclose(delta[eyes], 0, rtol=0, atol=GEOMETRY_ATOL)
    expected = model.expression_basis[index].numpy() * 2.0
    np.testing.assert_allclose(delta, expected, rtol=0, atol=GEOMETRY_ATOL)
    np.testing.assert_array_equal(changed.faces, neutral.faces)


def test_head_joint_rotates_its_vertices_around_the_joint(model, neutral):
    head = model.joint_names.index("head")
    neck = model.joint_names.index("neck")
    rotations = model.template_vertex_positions.new_zeros((model.num_joints, 3))
    rotations[head, 1] = 0.35

    posed = generate_mesh(model, rotations=rotations)
    head_vertices = model.skinning_weights[head].numpy() == 1.0
    neck_vertices = model.skinning_weights[neck].numpy() == 1.0
    assert head_vertices.any() and neck_vertices.any()
    pivot = model.template_joint_positions[head].numpy()
    rotation = Rotation.from_rotvec(rotations[head].numpy())
    expected = rotation.apply(neutral.vertices[head_vertices] - pivot) + pivot
    displacement = np.linalg.norm(expected - neutral.vertices[head_vertices], axis=1)
    assert displacement.max() > MIN_DEFORMATION
    np.testing.assert_allclose(
        posed.vertices[head_vertices], expected, rtol=0, atol=GEOMETRY_ATOL,
    )
    np.testing.assert_allclose(
        posed.vertices[neck_vertices], neutral.vertices[neck_vertices],
        rtol=0, atol=GEOMETRY_ATOL,
    )
    np.testing.assert_array_equal(posed.faces, neutral.faces)


def _export_from_cli(output, model_file, *parameters):
    command = [sys.executable, "-m", "personalab.gnm_mesh", "--output", str(output)]
    if model_file is not None:
        command += ["--model-file", model_file]
    subprocess.run(
        command + list(parameters), cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    return trimesh.load_mesh(output, process=False, maintain_order=True)


@pytest.mark.parametrize("suffix,atol", [(".obj", 5.1e-9), (".ply", 0.0)])
def test_cli_neutral_export(model_file, neutral, tmp_path, suffix, atol):
    exported = _export_from_cli(tmp_path / f"neutral{suffix}", model_file)
    np.testing.assert_allclose(exported.vertices, neutral.vertices, rtol=0, atol=atol)
    np.testing.assert_array_equal(exported.faces, neutral.faces)


def test_cli_controls_and_exact_repeat_in_separate_processes(model, model_file, tmp_path):
    parameters = (
        "--identity", "head_000=1.5",
        "--expression", "lower_face_region_000=2.0",
        "--rotation", "head=0,0.35,0",
        "--translation=0.01,-0.02,0.03",
    )
    first = _export_from_cli(tmp_path / "first.ply", model_file, *parameters)
    second = _export_from_cli(tmp_path / "second.ply", model_file, *parameters)

    template = model.template_vertex_positions
    identity = template.new_zeros(model.identity_dim)
    expression = template.new_zeros(model.expression_dim)
    rotations = template.new_zeros((model.num_joints, 3))
    identity[model.identity_names.index("head_000")] = 1.5
    expression[model.expression_names.index("lower_face_region_000")] = 2.0
    rotations[model.joint_names.index("head"), 1] = 0.35
    translation = template.new_tensor([0.01, -0.02, 0.03])
    reference = generate_mesh(
        model, identity=identity, expression=expression, rotations=rotations,
        translation=translation,
    )
    np.testing.assert_array_equal(first.vertices, reference.vertices)
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.faces, reference.faces)
    np.testing.assert_array_equal(first.faces, second.faces)
