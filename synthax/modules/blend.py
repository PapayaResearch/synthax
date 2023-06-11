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
from flax import linen as nn
from synthax.modules.base import SynthModule
from synthax.parameter import ModuleParameter, ModuleParameterRange
from synthax.config import SynthConfig
from typing import Optional


class CrossfadeKnob(SynthModule):
    """
    Crossfade knob parameter with no signal generation

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    parameter_ranges: Optional[ModuleParameterRange] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        # TODO: Refactor if possible
        self.parameters = {
            "ratio": ModuleParameter(
                name="ratio",
                range=self.parameter_ranges,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
        }

    def __call__(self):
        # TODO: Implement?
        raise NotImplementedError("Must override the __call__ method")


class SoftModeSelector(SynthModule):
    """
    A soft mode selector.
    If there are n different modes, return a probability distribution over them.

    Args:
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
        n_modes (int): TODO
        exponent (chex.Array): determines how strongly to scale each [0,1] value prior to normalization.
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    n_modes: int
    exponent: chex.Array = jnp.exp(1), # e
    parameter_ranges: Optional[ModuleParameterRange] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        # TODO: Refactor if possible
        self.parameters = {
            f"mode{i}weight": ModuleParameter(
                name=f"mode{i}weight",
                range=self.parameter_ranges,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for i in range(self.n_modes)
        }

    def __call__(self):
        # Normalize all mode weights so they sum to 1.0
        # TODO: .value?
        # TODO: Refactor get all values in SynthModule
        parameter_values = [p.value for k, p in self.parameters.items()]
        parameter_values_exp = jnp.power(parameter_values, exponent=self.exponent)
        return parameter_values_exp / jnp.sum(parameter_values_exp, axis=0)


class HardModeSelector(SynthModule):
    """
    A hard mode selector.
    NOTE: This is non-differentiable.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): Global configuration.
        n_modes (int): TODO
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    n_modes: int
    # TODO: Pass a Dict with a key "mode_weight" and then add "_{i}"
    # to keep a consistent API
    parameter_ranges: Optional[ModuleParameterRange] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        # TODO: Refactor if possible
        self.parameters = {
            f"mode{i}weight": ModuleParameter(
                name=f"mode{i}weight",
                range=self.parameter_ranges,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for i in range(self.n_modes)
        }

    def __call__(self):
        # TODO: .value?
        # TODO: Refactor get all values in SynthModule
        parameter_values = [p.value for k, p in self.parameters.items()]
        idx = jnp.argmax(parameter_values, axis=0)
        return jax.nn.one_hot(idx, num_classes=len(parameter_values)).T