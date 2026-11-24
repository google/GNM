# GNM: Generative aNthropometric Model and Ecosystem

![GNM Teaser Image](assets/readme/gnm_logo.png)

Welcome to the **GNM Ecosystem** repository. GNM - pronounced as genome
(/ˈdʒiː.noʊm/) in reference to the human genome - strives to be the most
accurate and complete 3D parametric human model.

3D Morphable Models (3DMMs) are widely used across computer vision, computer
graphics, and generative AI for representing human geometry and appearance. GNM
introduces a state-of-the-art family of parametric statistical human models and
its associated perception stack.

Our roadmap includes releasing a comprehensive suite of statistical models
complemented by perception and analysis technology. To facilitate early
community research and open development, we are beginning our open-source
release with **GNM Head**, our high-fidelity statistical 3D model of the human
head.

The ecosystem is released under the Apache 2.0 permissive license suitable for
both non-commercial and commercial applications.

## News

  - [2026-07-28]: Official [technical report](https://arxiv.org/abs/2607.23687) for the GNM Head released! 🚀🚀🚀

## PersonaLab workflow

Docker Engine and Docker Compose v2 are the PersonaLab execution environment.
No host Python installation or virtual environment is required.
Run these commands from the repository root on Linux (x86-64):

```sh
mkdir -p datasets models outputs .gnm-cache
export PERSONALAB_UID=$(id -u) PERSONALAB_GID=$(id -g)
docker compose build personalab
docker compose run --rm personalab python -m personalab.gnm_mesh --output outputs/neutral.obj
docker compose run --rm personalab python -m pytest -v -p no:cacheprovider tests
```

The official GNM model is downloaded by GNM on first use and cached in
`.gnm-cache/`. To use a specific local model, place its NPZ in `models/` and
pass `--model-file models/gnm_head.npz`. For the tests, pass
`-e PERSONALAB_GNM_MODEL_FILE=/app/models/gnm_head.npz` to `docker compose run`.
The datasets and local models are mounted read-only; cache and outputs are
persistent and writable. Rebuild the image after source or dependency changes.

The Issue #1 validation used GNM v3.0 Head from Hugging Face repository
`google/gnm-v3`, revision `c01e90d298d82301f9fd18f54806be751775cb7c`, file
`v3_0/gnm_head.npz`, with SHA-256
`61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb`.
Use this exact asset for comparisons with that validation.

For CUDA workloads, Docker Compose 2.30+ and a working NVIDIA Container
Toolkit installation are required. Build and check the explicitly selected
CUDA 12.8 PyTorch environment:

```sh
docker compose --profile cuda build personalab-cuda
docker compose --profile cuda run --rm personalab-cuda python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

The CUDA service refuses to start when PyTorch cannot access CUDA. The existing
Issue #1 mesh CLI intentionally executes on CPU; GPU fitting is not implemented
by this container configuration. The image pins PyTorch and the direct
numerical/test dependencies reported in Issue #1, but does not lock every
transitive dependency or the base image digest.

## GNM Ecosystem Packages

Here we list all the available GNM packages:

| Package | Description | Status | Teaser |
| :---: | :--- | :---: | :---: |
| **[GNM Head](gnm/shape/README.md)**  | Parametric 3D statistical human head and face geometry model providing fine-grained, disentangled control over identity, expressions, and head pose. The model contains controllable internal anatomy including eyeballs, teeth and tongue. Includes multi-framework backend support for **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**, along with semantic parameter sampling. <br><br>*Technical report*: [GNM Head: A Generative aNthropometric Model of the human head](https://arxiv.org/abs/2607.23687) | [![CI Linux](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml)<br>[![CI macOS](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml)<br>[![CI Windows](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml)<br>[![Lint](https://github.com/google/gnm/actions/workflows/lint.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/lint.yml) | ![GNM Head Teaser](gnm/shape/assets/readme/teaser_heads_cropped.gif) ![GNM Head demo teaser](gnm/shape/assets/readme/gnm_head_demo.gif)

## Citation
If you use any part of the GNM Ecosystem in your work, please consider citing
the corresponding package. Relevant bibtex entries are listed below as well as
within the individual packages.

**GNM Head**

```bash
@article{ploumpis2026gnmhead,
  title={GNM Head: A Generative aNthropometric Model of the human head},
  author={Ploumpis, S. and Bednarik, J. and Zoss, G. and Guseinov, R. and Prasso, L. and Chandran, P. and Boyne, O. and Choutas, V. and Bolkart, T. and Wang, D. and Chai, M. and Qiu, D. and Winberg, S. and Rainer, G. and Bridgeman, L. and Vicini, D. and Riviere, J. and Boetzel, Y. and Koumis, A. and Busch, J. and Herrera, C. and Still, J. and Ysebert, S. and Lincoln, P. and Escolano, S. O. and Rhemann, C. and Wood, E. and Beeler, T. and Zafeiriou, S.},
  year={2026},
  eprint={2607.23687},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.23687},
}
```

## Contributing
We'd love to accept your patches and contributions to this project! See
[CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to get started
and how we handle external contributions.

## License
This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.
