"""Verify that a non-editable GNM Shape install includes its landmark data."""

from importlib import metadata, resources
from pathlib import Path
import subprocess
import sys


def test_installed_wheel_can_load_sparse_landmarks(tmp_path):
    subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve())], cwd=tmp_path,
        check=True,
    )


def _verify_installed_landmarks():
    from gnm.shape import gnm_landmarks
    import numpy as np

    data_path = "data/landmarks/head_sparse_68.txt"
    wheel_path = f"gnm/shape/{data_path}"
    distribution = metadata.distribution("gnm-shape")
    assert wheel_path in {entry.as_posix() for entry in distribution.files or ()}, (
        f"{wheel_path} is missing from the installed GNM wheel"
    )
    resource = resources.files("gnm.shape").joinpath(data_path)
    with resources.as_file(resource) as resource_path:
        assert resource_path.resolve() == distribution.locate_file(wheel_path).resolve()

    landmarks = gnm_landmarks.load_landmarks(
        gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68,
    )
    assert landmarks.indices.shape == landmarks.weights.shape == (68, 3)
    assert landmarks.indices.dtype == np.int32
    assert landmarks.weights.dtype == np.float32
    assert np.all(landmarks.indices >= 0)
    assert np.isfinite(landmarks.weights).all()


if __name__ == "__main__":
    _verify_installed_landmarks()
