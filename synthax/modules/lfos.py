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
from synthax.modules.base import ControlRateModule
from synthax.parameter import ModuleParameterRange, from_0to1
from synthax.config import SynthConfig
from synthax.types import ParameterSpec, Signal
from typing import Optional


class LFO(ControlRateModule):
    """
    Low Frequency Oscillator.

    The LFO shape can be any mixture of sine, triangle, saw, reverse saw, and
    square waves. Contributions of each base-shape are determined by the
    :attr:`~synthax.module.LFO.lfo_types` values, which are between 0 and 1.

    Args:
        config (SynthConfig): See :class:`~synthax.module.SynthConfig`.
        exponent: A non-negative value that determines the discrimination of the
            soft-max selector for LFO shapes. Higher values will tend to favour
            one LFO shape over all others. Lower values will result in a more
            even blend of LFO shapes.
        frequency (ParameterSpec): Accepts a parameter range, initial values or both.
        mod_depth (ParameterSpec): Accepts a parameter range, initial values or both.
        initial_phase (ParameterSpec): Accepts a parameter range, initial values or both.
        sin (ParameterSpec): Accepts a parameter range, initial values or both.
        tri (ParameterSpec): Accepts a parameter range, initial values or both.
        saw (ParameterSpec): Accepts a parameter range, initial values or both.
        rsaw (ParameterSpec): Accepts a parameter range, initial values or both.
        sqr (ParameterSpec): Accepts a parameter range, initial values or both.
    """

    exponent: chex.Array = jnp.exp(1) # e
    frequency: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=20.0,
        curve=0.25,
    )
    mod_depth: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=-10.0,
        maximum=20.0,
        curve=0.5,
        symmetric=True
    )
    initial_phase: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=-jnp.pi,
        maximum=jnp.pi,
    )
    sin: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )
    tri: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )
    saw: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )
    rsaw: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )
    sqr: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        super().setup()
        self.lfo_types = ["_sin", "_tri", "_saw", "_rsaw", "_sqr"]

    def __call__(self, mod_signal: Optional[Signal] = None):
        """
        Generates low frequency oscillator control signal.

        Args:
            mod_signal (Signal):  LFO rate modulation signal in Hz. To modulate the
                depth of the LFO, use :class:`synthax.module.ControlRateVCA`.
        """
        # Create frequency signal
        frequency = self.make_control(mod_signal)
        argument = jnp.cumsum(2 * jnp.pi * frequency / self.control_rate, axis=1)
        argument += jnp.expand_dims(
            from_0to1(self._initial_phase, self._initial_phase_range),
            axis=1
        )

        # Get LFO shapes
        shapes = jnp.stack(self.make_lfo_shapes(argument), axis=1)

        # Apply mode selection to the LFO shapes
        # TODO: avoid loop to improve jitted performance.
        mode = jnp.stack(
            [getattr(self, lfo) for lfo in self.lfo_types],
            axis=1
        )
        mode = jnp.power(mode, self.exponent)
        mode /= jnp.sum(mode, axis=1, keepdims=True)

        signal = jnp.matmul(jnp.expand_dims(mode, axis=1), shapes).squeeze(1)
        return signal

    def make_control(self, mod_signal: Optional[Signal] = None) -> Signal:
        """
        Applies the LFO-rate modulation signal to the LFO base frequency.

        Args:
            mod_signal (Signal): Modulation signal in Hz. Positive values increase the
                LFO base rate; negative values decrease it.
        """
        frequency = jnp.expand_dims(
            from_0to1(self._frequency, self._frequency_range),
            axis=1
        )

        # If no modulation, then return a view of the frequency of this
        # LFO expanded to the control buffer size
        if mod_signal is None:
            return jnp.broadcast_to(frequency, (frequency.shape[0], self.control_buffer_size))

        modulation = jnp.expand_dims(
            from_0to1(self._mod_depth, self._mod_depth_range),
            axis=1
        ) * mod_signal

        return jnp.maximum(frequency + modulation, 0.0)

    def make_lfo_shapes(
        self,
        argument: Signal
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Generates five separate signals for each LFO shape and returns them as a
        tuple, to be mixed by :func:`synthax.module.LFO.__call__`.

        Args:
            argument (Signal): Time-varying phase to generate LFO signals.
        """
        cos = jnp.cos(argument + jnp.pi)
        square = jnp.sign(cos)

        cos = (cos + 1.0) / 2.0
        square = (square + 1.0) / 2.0

        saw = jnp.remainder(argument, 2 * jnp.pi) / (2 * jnp.pi)
        rev_saw = 1.0 - saw

        triangle = 2 * saw
        triangle = jnp.where(triangle > 1.0, 2.0 - triangle, triangle)

        return cos, triangle, saw, rev_saw, square
