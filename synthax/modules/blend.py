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
import chex
import dataclasses
from flax import linen as nn
from synthax.modules.base import SynthModule
from synthax.parameter import ModuleParameterRange, from_0to1
from synthax.config import SynthConfig
from synthax.types import ParameterSpec
from typing import Optional


class CrossfadeKnob(SynthModule):
    """
    Crossfade knob parameter with no signal generation

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        ratio (synthax.types.ParameterSpec): Accepts a parameter range, initial values or both.
    """

    ratio: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )


class SoftModeSelector(SynthModule):
    """
    A soft mode selector.
    If there are n different modes, return a probability distribution over them.

    Args:
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
        n_modes (int): Number of modes.
        exponent (chex.Array): determines how strongly to scale each [0,1] value prior to normalization.
        mode (synthax.types.ParameterSpec): Accepts a parameter range, initial values or both, for all modes equally.
    """

    n_modes: int
    exponent: chex.Array = jnp.exp(1), # e
    mode: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        # The default parameter range applies to all modes
        default_ranges = {f.name: f.default for f in dataclasses.fields(self)}
        default_range = default_ranges["mode"]
        initializer = lambda PRNG_key, range_: jax.random.uniform(
            PRNG_key,
            shape=(self.n_modes, self.config.batch_size),
            minval=range_.minimum,
            maxval=range_.maximum
        )
        # Initialize the parameter
        self._init_param("mode", default_range, initializer)

    def __call__(self):
        # Normalize all mode weights so they sum to 1.0
        values_exp = jnp.power(
            from_0to1(self._mode, self._mode_range),
            self.exponent[0]
        )
        return values_exp / jnp.sum(values_exp, axis=0)


class HardModeSelector(SynthModule):
    """
    A hard mode selector.
    NOTE: This is non-differentiable.

    Args:
        config (SynthConfig): Global configuration.
        n_modes (int): Number of modes.
        mode (synthax.types.ParameterSpec): Accepts a parameter range, initial values or both, for all modes equally.
    """

    n_modes: int
    mode: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        # The default parameter range applies to all modes
        default_ranges = {f.name: f.default for f in dataclasses.fields(self)}
        default_range = default_ranges["mode"]
        initializer = lambda PRNG_key, range_: jax.random.uniform(
            PRNG_key,
            shape=(self.n_modes, self.config.batch_size),
            minval=range_.minimum,
            maxval=range_.maximum
        )
        # Initialize the parameter
        self._init_param("mode", default_range, initializer)

    def __call__(self):
        mode = from_0to1(self._mode, self._mode_range)
        idx = jnp.argmax(mode, axis=0)
        return jax.nn.one_hot(idx, num_classes=len(mode)).T
