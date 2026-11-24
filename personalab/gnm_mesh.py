"""Generate and export a GNM v3 head mesh using PyTorch."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from gnm.shape import gnm_data_loader, gnm_pytorch
import numpy as np
import torch
import trimesh


def load_head_model(model_file: str | Path | None = None) -> gnm_pytorch.GNM:
    """Load the v3 Head model on CPU from the GNM cache/CDN or a local NPZ."""
    major = gnm_pytorch.GNMMajorVersion.V3
    variant = gnm_pytorch.GNMVariant.HEAD
    if model_file is None:
        model = gnm_pytorch.GNM.from_remote(version=major, variant=variant)
    else:
        model = gnm_pytorch.GNM.from_custom_file(model_file)

    if (
        gnm_data_loader.full_version_to_major(model.version) != major
        or model.variant != variant
    ):
        raise ValueError(f"Expected GNM v3 Head, got {model.version} {model.variant}")
    return model.cpu().eval()


@torch.no_grad()
def generate_mesh(
    model: gnm_pytorch.GNM,
    *,
    identity: torch.Tensor | None = None,
    expression: torch.Tensor | None = None,
    rotations: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
) -> trimesh.Trimesh:
    """Evaluate one mesh with native GNM tensors and preserve its topology."""
    vertices = model(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    ).cpu().numpy()
    faces = model.triangles.detach().cpu().numpy()

    if (
        vertices.shape != (model.num_vertices, 3)
        or vertices.size == 0
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Expected one mesh with finite (V, 3) vertex coordinates")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or faces.size == 0
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise ValueError("Expected a nonempty (T, 3) array of integer triangle indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle indices are outside the vertex array")
    if np.any(faces == np.roll(faces, 1, axis=1)):
        raise ValueError("A triangle references the same vertex more than once")

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _vector(template: torch.Tensor, text: str, size: int) -> torch.Tensor:
    parts = text.split(",")
    if len(parts) != size:
        raise ValueError(f"Expected {size} comma-separated values, got {text!r}")
    values = template.new_tensor([float(part) for part in parts])
    if not torch.isfinite(values).all():
        raise ValueError(f"Expected finite parameter values, got {text!r}")
    return values


def _named_parameters(
    template: torch.Tensor,
    shape: tuple[int, ...],
    names: Sequence[str],
    assignments: Sequence[str],
) -> torch.Tensor | None:
    if not assignments:
        return None

    parameters = template.new_zeros(shape)
    for assignment in assignments:
        name, separator, text = assignment.partition("=")
        if not separator or name not in names:
            raise ValueError(
                f"Invalid named parameter {assignment!r}. Use NAME=VALUE and "
                "see --list-parameters for model names."
            )
        parameter = parameters[names.index(name)]
        values = _vector(template, text, parameter.numel())
        parameter.copy_(values.reshape(parameter.shape))
    return parameters


def _output_path(text: str) -> Path:
    path = Path(text)
    if path.suffix.lower() not in {".obj", ".ply"}:
        raise argparse.ArgumentTypeError("output must have a .obj or .ply extension")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """Run the mesh export CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=_output_path, help="mesh path (.obj or .ply)")
    action.add_argument(
        "--list-parameters", action="store_true", help="show model names",
    )
    parser.add_argument("--model-file", type=Path, help="local GNM v3 Head .npz file")
    parser.add_argument("--identity", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--expression", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--rotation",
        action="append",
        default=[],
        metavar="JOINT=X,Y,Z",
        help="joint axis-angle vector in radians",
    )
    parser.add_argument(
        "--translation", metavar="X,Y,Z", help="root translation in meters",
    )
    args = parser.parse_args(argv)

    try:
        model = load_head_model(args.model_file)
        if args.list_parameters:
            print(f"GNM {model.version} {model.variant} (PyTorch CPU)")
            print(f"Identity ({model.identity_dim}): {', '.join(model.identity_names)}")
            print(f"Expression ({model.expression_dim}): {', '.join(model.expression_names)}")
            print(f"Joints ({model.num_joints}): {', '.join(model.joint_names)}")
            return

        template = model.template_vertex_positions
        mesh = generate_mesh(
            model,
            identity=_named_parameters(
                template, (model.identity_dim,), model.identity_names, args.identity,
            ),
            expression=_named_parameters(
                template, (model.expression_dim,), model.expression_names, args.expression,
            ),
            rotations=_named_parameters(
                template, (model.num_joints, 3), model.joint_names, args.rotation,
            ),
            translation=(
                _vector(template, args.translation, 3)
                if args.translation is not None else None
            ),
        )
        mesh.export(args.output, file_type=args.output.suffix[1:].lower())
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    print(
        f"GNM {model.version} {model.variant}: {len(mesh.vertices)} vertices, "
        f"{len(mesh.faces)} triangles -> {args.output}"
    )


if __name__ == "__main__":
    main()
