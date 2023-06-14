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
from synthax.parameter import ModuleParameter, ModuleParameterRange
from synthax.config import SynthConfig
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
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        n_input (int): Number of input signals to module mix.
        n_output (int): Number of output signals to generate.
        input_names (List(str)): TODO
        output_names (List(str)): TODO
        mod (ParameterSpec): TODO.
    """

    n_input: int
    n_output: int
    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
    mod: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
        curve=0.5,
    )

    def setup(self):
        param_names = []
        for i in range(self.n_input):
            for j in range(self.n_output):
                # Apply custom param name if it was passed in
                if self.input_names is not None:
                    param_name = f"{self.input_names[i]}->{self.output_names[j]}"
                else:
                    param_name = f"{i}->{j}"
                param_names.append(param_name)
                setattr(self, param_name, self.mod)
        # The default parameter range applies to all modes
        default_values = {f.name: f.default for f in dataclasses.fields(self)}
        default_rng = default_values["mod"]
        self.parameters = {
            name: self._init_param(name, default_rng)
            for name in param_names
        }

    def __call__(self, *signals: Signal):
        """
        Performs mixture of modulation signals.
        """

        parameter_values = jnp.array([p.from_0to1() for k, p in self.parameters.items()])
        parameter_values = jnp.reshape(
            parameter_values,
            (self.batch_size, self.n_input, self.n_output)
        )
        parameter_values = jnp.swapaxes(parameter_values, 1, 2)

        signals = jnp.stack(signals, axis=1)

        modulation = jnp.array_split(
            jnp.matmul(parameter_values, signals),
            self.n_output,
            axis=1
        )
        return tuple(m.squeeze(1) for m in modulation)


class AudioMixer(SynthModule):
    """
    Sums together N audio signals and applies range-normalization if the
    resulting signal is outside of [-1, 1].

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        n_input (int): TODO
        names (List[str]): TODO
        level (ParameterSpec): TODO
    """

    n_input: int
    names: Optional[list[str]] = None
    level: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
        curve=1.0,
    )

    def setup(self):
        param_names = []
        for i in range(self.n_input):
            param_name = f"level{i}" if self.names is None else self.names[i]
            param_names.append(param_name)
            setattr(self, param_name, self.level)
        # The default parameter range applies to all modes
        default_values = {f.name: f.default for f in dataclasses.fields(self)}
        default_rng = default_values["level"]
        self.parameters = {
            name: self._init_param(name, default_rng)
            for name in param_names
        }

    def __call__(self, *signals: Signal):
        """
        Returns a mixed signal from an array of input signals.
        """
        parameter_values = jnp.stack([p.from_0to1() for k, p in self.parameters.items()], axis=1)
        signals = jnp.stack(signals, axis=1)
        mixed_signal = normalize_if_clipping(
            jnp.matmul(
                jnp.expand_dims(parameter_values, 1),
                signals
            ).squeeze(1)
        )
        return mixed_signal
