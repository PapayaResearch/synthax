# Copyright (c) 2023
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import jax
import jax.numpy as jnp
import dataclasses
from synthax.modules.base import SynthModule
from synthax.parameter import ModuleParameterRange, from_0to1
from synthax.functional import normalize_if_clipping
from synthax.types import Signal, ParameterSpec
from typing import Optional

class ModulationMixer(SynthModule):
    """
    A modulation matrix that combines :math:`N` input modulation signals to make
    :math:`M` output modulation signals. Each output is a linear combination of
    all in input signals, as determined by an :math:`N \times M` mixing matrix.

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        n_input (int): Number of input signals to module mix.
        n_output (int): Number of output signals to generate.
        mod (ParameterSpec): Accepts a parameter range, initial values or both.
    """

    n_input: int
    n_output: int
    mod: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
        curve=0.5
    )

    def setup(self):
        # The default parameter range applies to all inputs
        default_ranges = {f.name: f.default for f in dataclasses.fields(self)}
        default_range = default_ranges["mod"]
        initializer = lambda PRNG_key, range_: jax.random.uniform(
            PRNG_key,
            shape=(self.config.batch_size, self.n_output, self.n_input),
            minval=range_.minimum,
            maxval=range_.maximum
        )
        # Initialize the parameter
        self._init_param("mod", default_range, initializer)

    def __call__(self, *signals: Signal):
        """
        Performs mixture of modulation signals.
        """

        signals = jnp.stack(signals, axis=1)

        modulation = jnp.array_split(
            jnp.matmul(from_0to1(self._mod, self._mod_range), signals),
            self.n_output,
            axis=1
        )

        # TODO: avoid loop to improve jitted performance.
        return tuple(m.squeeze(1) for m in modulation)


class AudioMixer(SynthModule):
    """
    Sums together N audio signals and applies range-normalization if the
    resulting signal is outside of [-1, 1].

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        n_input (int): Number of inputs.
        level (ParameterSpec): Accepts a parameter range, initial values or both, for all inputs equally.
    """

    n_input: int
    level: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
        curve=1.0
    )

    def setup(self):
        # The default parameter range applies to all inputs
        default_ranges = {f.name: f.default for f in dataclasses.fields(self)}
        default_range = default_ranges["level"]
        initializer = lambda PRNG_key, range_: jax.random.uniform(
            PRNG_key,
            shape=(self.config.batch_size, self.n_input),
            minval=range_.minimum,
            maxval=range_.maximum
        )
        # Initialize the parameter
        self._init_param("level", default_range, initializer)

    def __call__(self, *signals: Signal):
        """
        Returns a mixed signal from an array of input signals.
        """
        # TODO: improve performance
        signals = jnp.stack(signals, axis=1)

        # TODO: improve performance
        mixed_signal = normalize_if_clipping(
            jnp.matmul(
                jnp.expand_dims(from_0to1(self._level, self._level_range), 1),
                signals
            ).squeeze(1)
        )

        return mixed_signal
