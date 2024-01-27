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
from synthax.modules.base import SynthModule
from synthax.parameter import ModuleParameterRange, from_0to1
from synthax.config import SynthConfig
from synthax.functional import midi_to_hz
from synthax.types import Signal, ParameterSpec
from typing import Optional

class VCO(SynthModule):
    """
    Base class for voltage controlled oscillators.

    Think of this as a VCO on a modular synthesizer. It has a base pitch
    (specified here as a midi value), and a pitch modulation depth. Its call
    accepts a modulation signal between [-1, 1]. An array of 0's returns a
    stationary audio signal at its base pitch.

    Args:
        config (SynthConfig): See :class:`~synthax.module.SynthModule`
        tuning (ParameterSpec): Accepts a parameter range, initial values or both.
        mod_depth (ParameterSpec): Accepts a parameter range, initial values or both.
        initial_phase (ParameterSpec): Accepts a parameter range, initial values or both.
    """

    tuning: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=-24.0,
        maximum=24.0,
    )
    mod_depth: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=-96.0,
        maximum=96.0,
        curve=0.2,
        symmetric=True
    )
    initial_phase: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=-jnp.pi,
        maximum=jnp.pi,
    )

    def __call__(
        self,
        midi_f0: chex.Array,
        mod_signal: Optional[Signal] = None
    ) -> Signal:
        """
        Generates audio signal from modulation signal.

        Args:
            midi_f0 (chex.Array): Fundamental of note in midi note value (0 - 127).
            mod_signal (synthax.types.Signal): Modulation signal to apply to the pitch.
        """

        # TODO: improve performance
        control_as_frequency = self.make_control_as_frequency(midi_f0, mod_signal)
        # TODO: improve performance
        cosine_argument = self.make_argument(control_as_frequency)
        cosine_argument += jnp.expand_dims(
            from_0to1(self._initial_phase, self._initial_phase_range),
            axis=1
        )
        # TODO: improve performance
        signal = self.oscillator(cosine_argument, midi_f0)
        return signal

    def make_control_as_frequency(
        self,
        midi_f0: chex.Array,
        mod_signal: Optional[Signal] = None
    ) -> Signal:
        """
        Generates a time-varying control signal in frequency (Hz) from a midi
        fundamental pitch and pitch-modulation signal.

        Args:
            midi_f0 (chex.Array): Fundamental pitch value in midi.
            mod_signal (synthax.types.Signal): Pitch modulation signal in midi.
        """
        midi_f0 = jnp.expand_dims(
            midi_f0 + from_0to1(self._tuning, self._tuning_range),
            axis=1
        )

        # If there is no modulation, then convert the midi_f0 values to
        # frequency and return an expanded view that contains buffer size
        # number of values
        if mod_signal is None:
            control_hz = midi_to_hz(midi_f0)
            return jnp.broadcast_to(control_hz, (control_hz.shape[0], self.buffer_size))

        # If there is modulation, then add that to the fundamental,
        # clamp to a range [0.0, 127.0], then return in frequency Hz.
        modulation = jnp.expand_dims(
            from_0to1(self._mod_depth, self._mod_depth_range),
            axis=1
        ) * mod_signal
        control = jax.lax.clamp(0.0, midi_f0 + modulation, 127.0)
        return midi_to_hz(control)

    def make_argument(self, freq: Signal) -> Signal:
        """
        Generates the phase argument to feed an oscillating function to
        generate an audio signal.

        Args:
            freq (synthax.types.Signal): Time-varying instantaneous frequency in Hz.
        """
        return jnp.cumsum(2 * jnp.pi * freq / self.sample_rate, axis=1)

    def oscillator(self, argument: Signal, *args: tuple, **kwargs: dict) -> Signal:
        """
        This function accepts a phase argument and generates output audio. It is
        implemented by the child class.

        Args:
            argument (synthax.types.Signal): The phase of the oscillator at each time sample.
        """
        raise NotImplementedError("Derived classes must override this method")


class SineVCO(VCO):
    """
    Simple VCO that generates a pitched sinusoid.

    Args:
        config (SynthConfig): See :class:`~synthax.module.SynthModule`
    """

    def oscillator(self, argument: Signal, midi_f0: chex.Array) -> Signal:
        """
        A cosine oscillator.

        Args:
            argument (synthax.types.Signal): The phase of the oscillator at each time sample.
        """
        return jnp.cos(argument)


class FmVCO(VCO):
    """
    Frequency modulation VCO. Takes a modulation signal as instantaneous
    frequency (in Hz) rather than as a midi value.

    Typical modulation is calculated in pitch-space (midi). For FM to work,
    we have to change the order of calculations. Here the modulation depth is
    re-interpreted as the "modulation index" which is tied to the fundamental of
    the oscillator being modulated:

        :math:`I = \\Delta f / f_m`

    where :math:`I` is the modulation index, :math:`\\Delta f` is the frequency
    deviation imparted by the modulation, and :math:`f_m` is the modulation
    frequency, both in Hz.

    Args:
        config (SynthConfig): See :class:`~synthax.module.SynthModule`
    """

    # We include this override to output to make mod_signal non-optional
    def __call__(self, midi_f0: chex.Array, mod_signal: Signal) -> Signal:
        """
        Args:
            midi_f0 (chex.Array): note value in midi
            mod_signal (synthax.types.Signal): audio rate frequency modulation signal
        """
        return super().__call__(midi_f0, mod_signal)

    def make_control_as_frequency(
        self,
        midi_f0: chex.Array,
        mod_signal
    ) -> Signal:
        """
        Creates a time-varying control signal in instantaneous frequency (Hz).

        Args:
            midi_f0 (chex.Array): Fundamental frequency in midi.
            mod_signal (synthax.types.Signal): FM modulation signal (interpreted as modulation index).
        """
        # Compute modulation in Hz space (rather than midi-space).
        f0_hz = jnp.expand_dims(
            midi_to_hz(
                midi_f0 + from_0to1(self._tuning, self._tuning_range),
            ),
            axis=1
        )
        fm_depth = jnp.expand_dims(
            from_0to1(self._mod_depth, self._mod_depth_range),
            axis=1
        ) * f0_hz
        modulation_hz = fm_depth * mod_signal
        return jax.lax.clamp(0.0, f0_hz + modulation_hz, self.nyquist)

    def oscillator(self, argument: Signal, midi_f0: chex.Array) -> Signal:
        """
        A cosine oscillator. ...Good ol' cosine.

        Args:
            argument (synthax.types.Signal): The phase of the oscillator at each time sample.
        """
        return jnp.cos(argument)


class SquareSawVCO(VCO):
    """
    An oscillator that can take on either a square or a sawtooth waveshape, and
    can sweep continuously between them, as determined by the
    :attr:`~synthax.module.SquareSawVCO.shape` parameter. A shape value of 0
    makes a square wave; a shape of 1 makes a saw wave.

    With apologies to Lazzarini and Timoney (2010).
    `"New perspectives on distortion synthesis for virtual analog oscillators."
    <https://doi.org/10.1162/comj.2010.34.1.28>`_
    Computer Music Journal 34, no. 1: 28-40.

    Args:
        config (SynthConfig): See :class:`~synthax.module.SynthModule`
        tuning (ParameterSpec): Accepts a parameter range, initial values or both.
        mod_depth (ParameterSpec): Accepts a parameter range, initial values or both.
        initial_phase (ParameterSpec): Accepts a parameter range, initial values or both.
        shape (ParameterSpec): Accepts a parameter range, initial values or both.
    """

    shape: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def oscillator(self, argument: Signal, midi_f0: chex.Array) -> Signal:
        """
        Generates output square/saw audio given a phase argument.

        Args:
            argument (synthax.types.Signal): The phase of the oscillator at each time sample.
            midi_f0 (chex.Array): Fundamental frequency in midi.
        """
        partials = jnp.expand_dims(self.partials_constant(midi_f0), axis=1)
        square = jnp.tanh(jnp.pi * partials * jnp.sin(argument) / 2)
        shape = jnp.expand_dims(
            from_0to1(self._shape, self._shape_range),
            axis=1
        )
        return (1 - shape / 2) * square * (1 + shape * jnp.cos(argument))

    def partials_constant(self, midi_f0):
        """
        Calculates a value to determine the number of overtones in the resulting
        square / saw wave, in order to keep aliasing at an acceptable level.
        Higher fundamental frequencies require fewer partials for a rich sound;
        lower-frequency sounds can safely have more partials without causing
        audible aliasing.

        Args:
            midi_f0 (chex.Array): Fundamental frequency in midi.
        """
        tuning = from_0to1(self._tuning, self._tuning_range)
        mod_depth = from_0to1(self._mod_depth, self._mod_depth_range)

        max_pitch = (
            midi_f0 + tuning + jnp.maximum(mod_depth, 0)
        )
        max_f0 = midi_to_hz(max_pitch)
        return 12000 / (max_f0 * jnp.log10(max_f0))


class Noise(SynthModule):
    """
    Generates white noise that is the same length as the buffer.

    `Note`: If you have multiple `Noise` modules in the same
    :class:`~synthax.synth.BaseSynth`, make sure you instantiate
    each `Noise` with a unique seed.

    Args:
        config (SynthConfig): See :class:`~synthax.module.SynthModule`
        seed (int): Seed for the random number generator.
    """

    seed: int = 0

    def setup(self):
        def make_noise():
            return jnp.broadcast_to(
                jax.random.uniform(
                    jax.random.PRNGKey(self.seed),
                    shape=(1, self.buffer_size),
                    minval=-1.0,
                    maxval=1.0
                ),
                (self.batch_size, self.buffer_size)
            )
        self._noise = make_noise()

    def __call__(self) -> Signal:
        return self._noise
