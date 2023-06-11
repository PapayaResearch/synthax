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
from synthax.modules.base import SynthModule
from synthax.parameter import ModuleParameter, ModuleParameterRange
from synthax.config import SynthConfig
from synthax.functional import normalize_if_clipping
from synthax.types import Signal

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
        input_names (List(str)): TODO
        output_names (List(str)): TODO
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    n_input: int
    n_output: int
    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
    parameter_ranges: Optional[ModuleParameterRange] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
        curve=0.5,
    )

    def setup(self):
        # TODO: Refactor if possible
        parameters = {}
        for i in range(self.n_input):
            for j in range(self.n_output):
                # Apply custom param name if it was passed in
                if self.input_names is not None:
                    name = f"{self.input_names[i]}->{self.output_names[j]}"
                else:
                    name = f"{i}->{j}"

                parameters[name] = ModuleParameter(
                    name=name,
                    range=self.parameter_ranges,
                    value=jax.random.uniform(
                        self.PRNG_key,
                        shape=(self.config.batch_size,)
                    )
                )
        self.parameters = parameters

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
        n_input (int): TODO
        curves (List[float]): TODO
        names (List[str]): TODO
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    n_input: int
    names: Optional[list[str]] = None
    parameter_ranges: Optional[ModuleParameterRange] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
        curve=1.0,
    )

    def setup(self):
        names = [f"level{i}" if self.names is None else self.names[i]
                      for i in range(self.n_input)]

        parameter_ranges = [self.parameter_ranges] * self.n_input

        # TODO: Refactor if possible
        self.parameters = {
            name: ModuleParameter(
                name=name,
                range=parameter_range,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for name, parameter_range in zip(names, parameter_ranges)
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
        return self.to_buffer_size(mixed_signal)