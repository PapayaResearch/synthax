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
from synthax.parameter import ModuleParameter, ModuleParameterRange
from synthax.config import SynthConfig
from synthax.types import ParameterSpec
from typing import Optional


class CrossfadeKnob(SynthModule):
    """
    Crossfade knob parameter with no signal generation

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        ratio (ParameterSpec): TODO
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
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        n_modes (int): TODO
        exponent (chex.Array): determines how strongly to scale each [0,1] value prior to normalization.
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    n_modes: int
    exponent: chex.Array = jnp.exp(1), # e
    mode: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        param_names = []
        for i in range(self.n_modes):
            param_name = f"mode{i}weight"
            param_names.append(param_name)
            setattr(self, param_name, self.mode)
        # The default parameter range applies to all modes
        default_values = {f.name: f.default for f in dataclasses.fields(self)}
        default_rng = default_values["mode"]
        self.parameters = {
            name: self._init_param(name, default_rng)
            for name in param_names
        }

    def __call__(self):
        # Normalize all mode weights so they sum to 1.0
        # TODO: Refactor get all values in SynthModule
        parameter_values = [p.from_0to1() for k, p in self.parameters.items()]
        parameter_values_exp = jnp.power(parameter_values, exponent=self.exponent)
        return parameter_values_exp / jnp.sum(parameter_values_exp, axis=0)


class HardModeSelector(SynthModule):
    """
    A hard mode selector.
    NOTE: This is non-differentiable.

    Args:
        config (SynthConfig): Global configuration.
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        n_modes (int): TODO
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    n_modes: int
    mode: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        param_names = []
        for i in range(self.n_modes):
            param_name = f"mode{i}weight"
            param_names.append(param_name)
            setattr(self, param_name, self.mode)
        # The default parameter range applies to all modes
        default_values = {f.name: f.default for f in dataclasses.fields(self)}
        default_rng = default_values["mode"]
        self.parameters = {
            name: self._init_param(name, default_rng)
            for name in param_names
        }

    def __call__(self):
        # TODO: Refactor get all values in SynthModule
        parameter_values = [p.from_0to1() for k, p in self.parameters.items()]
        idx = jnp.argmax(parameter_values, axis=0)
        return jax.nn.one_hot(idx, num_classes=len(parameter_values)).T
