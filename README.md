# SynthAX: A Fast Modular Synthesizer in JAX ⚡️
[![Pyversions](https://img.shields.io/pypi/pyversions/synthax.svg?style=flat-square)](https://pypi.python.org/pypi/synthax)
[![PyPI version](https://badge.fury.io/py/synthax.svg)](https://badge.fury.io/py/synthax)
![PyPI - License](https://img.shields.io/pypi/l/synthax)

<h1 align="center">
  <a href="https://github.com/PapayaResearch/synthax/blob/main/media/logo.png">
    <img src="https://github.com/PapayaResearch/synthax/blob/main/media/logo.png?raw=true" width="400" /></a>
</h1>

Accelerating audio synthesis far beyond realtime speeds has a significant role to play in advancing intelligent audio production techniques. SynthAX is a fast virtual modular synthesizer written in JAX. At its peak, SynthAX generates audio over 60,000 times faster than realtime, and significantly faster than the state-of-the-art in accelerated sound synthesis. It leverages massive vectorization and high-throughput accelerators. You can get started here [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PapayaResearch/synthax/blob/main/examples/walkthrough.ipynb)

## Basic `synthax` API Usage

```python
import jax
from synthax.config import SynthConfig
from synthax.synth import ParametricSynth

# Generate PRNG key
key = jax.random.PRNGKey(42)
config = SynthConfig(
    batch_size=16,
    sample_rate=44100,
    buffer_size_seconds=4.0
)

key, subkey = jax.random.split(key)
# Instantiate synthesizer
synth = ParametricSynth(
    PRNG_key=subkey,
    config=config,
    sine=1,
    square_saw=1,
    fm_sine=1,
    fm_square_saw=0
)

key, subkey = jax.random.split(key)
# Initialize and run
params = synth.init(subkey)
audio = jax.jit(synth.apply)(params)
```

## Installation

The latest `synthax` release can directly be installed from PyPI:

```
pip install synthax
```

If you want to get the most recent commit, please install directly from the repository:

```
pip install git+https://github.com/PapayaResearch/synthax.git@main
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Acknowledgements & Citing

If you use `synthax` in your research, please cite the following:

```
@software{synthax2023,
  author = {Cherep, Manuel and Singh, Nikhil},
  title = {SynthAX: A Fast Modular Synthesizer in JAX},
  url = {https://github.com/PapayaResearch/synthax},
  version = {0.1.0},
  year = {2023},
}
```

This project is based on [torchsynth](https://github.com/torchsynth/torchsynth). We acknowledge financial support by Fulbright Spain.
