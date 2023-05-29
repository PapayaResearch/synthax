# SynthAX: A fast modular synthesizer in JAX ⚡️
[![Pyversions](https://img.shields.io/pypi/pyversions/synthax.svg?style=flat-square)](https://pypi.python.org/pypi/synthax)
[![PyPI version](https://badge.fury.io/py/synthax.svg)](https://badge.fury.io/py/synthax)
[![Paper](http://img.shields.io/badge/paper-arxiv.2212.04180-B31B1B.svg)]()

Do you want to leverage massive vectorization and high-throughput accelerators for modular synthesizers? `synthax` allows you to leverage JAX, XLA compilation and auto-vectorization/parallelization with your favorite accelerators. You can get started here 👉 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## Basic `synthax` API Usage

```python
import jax
import synthax

# TODO
rng = jax.random.PRNGKey(42)
```

## Installation

The latest `synhtax` release can directly be installed from PyPI:

```
pip install synthax
```

If you want to get the most recent commit, please install directly from the repository:

```
pip install git+https://github.com/PapayaResearch/synthax.git@main
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Acknowledgements & Citing

If you use `synthax` in your research, please cite the following [paper](https://arxiv.org/abs/2212.04180):

```
@software{synthax2023,
  author = {Cherep, Manuel and Singh, Nikhil},
  title = {SynthAX: A fast modular synthesizer in JAX},
  url = {https://github.com/PapayaResearch/synthax},
  version = {0.1.0},
  year = {2023},
}
```

This project is inspired by [torchsynth](https://github.com/torchsynth/torchsynth). We acknowledge financial support by Fulbright Spain.

## Development

You can run the test suite via `python -m pytest -vv --all`. Feel free to create an issue and/or start [contributing](CONTRIBUTING.md).
