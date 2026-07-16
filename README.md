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

The ecosystem is released under a permissive license suitable for both
non-commercial and commercial applications.


## GNM Ecosystem Packages

Here we list all the available GNM packages:

| Name | Description | Chips | Teaser |
| :--- | :--- | :--- | :---: |
| **[GNM Head](gnm/shape/README.md)** | Parametric 3D statistical human head and face geometry model providing fine-grained, disentangled control over identity, expressions, and head pose. The model contains controllable internal anatomy including eyeballs, teeth and tongue. Includes multi-framework backend support for **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**, along with semantic parameter sampling. | [![CI Linux](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml)<br>[![CI macOS](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml)<br>[![CI Windows](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml)<br>[![Lint](https://github.com/google/gnm/actions/workflows/lint.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/lint.yml) | ![GNM Head Teaser](gnm/shape/assets/readme/teaser_heads_cropped.gif) ![GNM Head demo teaser](gnm/shape/assets/readme/gnm_head_demo.gif)

## Citation
If you use any part of the GNM Ecosystem in your work, please consider citing
the corresponding package. Relevant bibtex entries are listed below as well as
within the individual packages.

**GNM Head**

```bash
coming soon
```

## Contributing
We'd love to accept your patches and contributions to this project! See
[CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to get started
and how we handle external contributions.

## License
This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.
