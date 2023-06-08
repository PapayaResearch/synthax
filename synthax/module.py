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

"""
Synth modules in JAX.
"""

import jax
import jax.numpy as jnp
import chex
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple, NewType
from flax import linen as nn
from synthax.config import SynthConfig
from synthax.parameter import ModuleParameter, ModuleParameterRange, ParameterName

# TODO:
# - Consider __call__ to take *args and **kwargs
# - Build docs and improve documentation
# - Store info in the config if needed since setup cannot take **kwargs
# - Move previous util functions somwhere else, but avoid util.py
# - Include chex and types everywhere
# - Include examples for each function to include automatically in the docs
# - Include tests

Signal = NewType('Signal', chex.Array)

def midi_to_hz(midi: chex.Array) -> chex.Array:
    """
    Convert from midi (linear pitch) to frequency in Hz.

    Args:
        midi (TODO): Linear pitch on the MIDI scale.

    Return:
        Frequency in Hz.
    """
    return 440.0 * (jnp.exp2((midi - 69.0) / 12.0))


def fix_length(signal: Signal, length: int) -> Signal:
    """
    Pad or truncate (TODO) to specified length.
    """
    num_samples = signal.shape[1]
    if num_samples < length:
        signal = jnp.pad(signal, (0, length - num_samples))
    elif num_samples > length:
        signal = signal[:, :length]
    return signal


def normalize_if_clipping(signal: Signal) -> Signal:
    """
    Only normalize invidiaul signals in batch that have samples
    less than -1.0 or greater than 1.0
    """
    max_sample = jnp.max(jnp.abs(signal), axis=1, keepdims=True)[0]
    return jnp.where(max_sample > 1.0, signal / max_sample, signal)


def normalize(signal: Signal) -> Signal:
    """
    Normalize every individual signal in batch.
    """
    max_sample = jnp.max(jnp.abs(signal), axis=1, keepdims=True)[0]
    return signal / max_sample


class SynthModule(nn.Module):
    """
    A base class for synthesis modules. A :class:`~.SynthModule`
    optionally takes input from other :class:`~.SynthModule` instances.
    The :class:`~.SynthModule` uses its (optional) input and its
    set of :class:`~synthax.parameter.ModuleParameter` to generate
    output.

    All :class:`~.SynthModule` objects should be atomic, i.e., they
    should not contain other :class:`~.SynthModule` objects. This
    design choice is in the spirit of modular synthesis.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): An object containing synthesis settings
            that are shared across all modules.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    @property
    def batch_size(self):
        """ Size of the batch to be generated. """
        return self.config.batch_size

    @property
    def sample_rate(self):
        """ Sample rate frequency in Hz. """
        return self.config.sample_rate

    @property
    def nyquist(self):
        """ Convenience property for the highest frequency that can be
        represented at :attr:`~.sample_rate` (as per Shannon-Nyquist). """
        return self.sample_rate / 2.0

    @property
    def eps(self):
        """ A very small value used to avoid computational errors. """
        return self.config.eps

    @property
    def buffer_size(self):
        """ Size of the module output in samples. """
        return self.config.buffer_size

    def to_buffer_size(self, signal: Signal) -> Signal:
        """
        Fixes the length of a signal to the default buffer size of this module,
        as specified by :attr:`~.SynthModule.buffer_size`. Longer signals are
        truncated to length; shorter signals are zero-padded.

        Args:
            signal (TODO): A signal to pad or truncate.
        """
        return fix_length(signal, self.buffer_size)

    def seconds_to_samples(self, seconds: float):
        """
        Convenience function to calculate the number of samples corresponding to
        given a time value and :attr:`~.sample_rate`. Returns a possibly
        fractional value.

        Args:
            seconds (float): Time value in seconds.
        """
        return seconds * self.sample_rate

    def set_parameter(self, parameter: ModuleParameter, value: chex.Array):
        """
        Updates a parameter value in a parameter-specific non-normalized range.

        Args:
            parameter_id: Id of the parameter to update.
            value:  Value to assign to the parameter.
        """
        parameter.to_0to1(value)


class ControlRateModule(SynthModule):
    """
    An abstract base class for non-audio modules that adapts the functions of
    :class:`.~SynthModule` to run at :attr:`~.ControlRateModule.control_rate`.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): An object containing synthesis settings
            that are shared across all modules.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    @property
    def sample_rate(self):
        raise NotImplementedError("This module operates at control rate")

    @property
    def buffer_size(self):
        raise NotImplementedError("This module uses control buffer size")

    @property
    def control_rate(self):
        """ Control rate frequency in Hz. """
        return self.config.control_rate

    @property
    def control_buffer_size(self):
        """ Size of the module output in samples. """
        return self.config.control_buffer_size

    def to_buffer_size(self, signal: Signal) -> Signal:
        """
        Fixes the length of a signal to the control buffer size of this module,
        as specified by :attr:`~.ControlRateModule.control_buffer_size`. Longer
        signals are truncated to length; shorter signals are zero-padded.

        Args:
            signal: A signal to pad or truncate.
        """
        return fix_length(signal, self.control_buffer_size)

    def seconds_to_samples(self, seconds: float):
        """
        Convenience function to calculate the number of samples corresponding to
        given a time value and :attr:`~.control_rate`. Returns a possibly
        fractional value.

        Args:
            seconds (float): Time value in seconds.
        """
        return seconds * self.control_rate

    def __call__(self) -> Signal:
        """
        Each :class:`~.ControlRateModule` should override this.
        """
        raise NotImplementedError("Must override the __call__ method")


class ADSR(ControlRateModule):
    """
    Envelope class for building a control-rate ADSR signal.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
        parameter_ranges (Dict[ParameterName, :class:`~synthax.parameter.ModuleParameterRange`]): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    # TODO: Allow parameter_ranges to be partial
    parameter_ranges: Optional[Dict[ParameterName, ModuleParameterRange]] = field(
        default_factory = lambda: {
            "attack": ModuleParameterRange(
                minimum=0.0,
                maximum=2.0,
                curve=0.5,
            ),
            "decay": ModuleParameterRange(
                minimum=0.0,
                maximum=2.0,
                curve=0.5,
            ),
            "sustain": ModuleParameterRange(
                minimum=0.0,
                maximum=1.0,
            ),
            "release": ModuleParameterRange(
                minimum=0.0,
                maximum=5.0,
                curve=0.5,
            ),
            "alpha": ModuleParameterRange(
                minimum=0.1,
                maximum=6.0,
            )
        })

    def setup(self):
        # TODO: Refactor if possible
        self.parameters = {
            name: ModuleParameter(
                name,
                parameter_range,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for name, parameter_range in self.parameter_ranges.items()
        }

    def __call__(self, note_on_duration) -> Signal:
        """
        Generate an ADSR envelope.

        By default, this envelope reacts as if it was triggered with midi, for
        example playing a keyboard. Each midi event has a beginning and end:
        note-on, when you press the key down; and note-off, when you release the
        key. `note_on_duration` is the amount of time that the key is depressed.

        During the note-on, the envelope moves through the attack and decay
        sections of the envelope. This leads to musically-intuitive, but
        programatically-counterintuitive behaviour:

        Example:
            Assume attack is .5 seconds, and decay is .5 seconds. If a note is
            held for .75 seconds, the envelope won't pass through the entire
            attack-and-decay (specifically, it will execute the entire attack,
            and only .25 seconds of the decay).

        If this is confusing, don't worry about it. ADSR's do a lot of work
        behind the scenes to make the playing experience feel natural.

        Args:
            note_on_duration (TODO): Duration of note on event in seconds.
        """

        # Calculations to accommodate attack/decay phase cut by note duration.
        attack = self.parameters["attack"].from_0to1()
        decay = self.parameters["decay"].from_0to1()

        new_attack = jnp.minimum(attack, note_on_duration)
        new_decay = jnp.maximum(note_on_duration - attack, 0)
        new_decay = jnp.minimum(new_decay, decay)

        attack_signal = self.make_attack(new_attack)
        decay_signal = self.make_decay(new_attack, new_decay)
        release_signal = self.make_release(note_on_duration)

        signal = attack_signal * decay_signal * release_signal

        return self.to_buffer_size(signal)

    def ramp(
            self,
            duration: float,
            start: Optional[float] = None,
            inverse: Optional[bool] = False
    ) -> Signal:
        """
        Makes a ramp of a given duration in seconds.

        The construction of this matrix is rather cryptic. Essentially, this
        method works by tilting and clipping ramps between 0 and 1, then
        applying a scaling factor :attr:`~alpha`.

        Args:
            duration (TODO): Length of the ramp in seconds.
            start (TODO): Initial delay of ramp in seconds.
            inverse (TODO): Toggle to flip the ramp from ascending to descending.
        """

        duration = self.seconds_to_samples(duration)

        # Convert to number of samples.
        if start is not None:
            start = self.seconds_to_samples(start)
        else:
            start = 0.0

        # Build ramps template.
        range_ = jnp.arange(self.control_buffer_size)
        ramp = jnp.broadcast_to(
            range_,
            (self.batch_size, range_.shape[0])
        )

        # Shape ramps.
        ramp = ramp - start
        ramp = jnp.maximum(ramp, 0)
        ramp = (ramp + self.eps) / duration + self.eps
        ramp = jnp.minimum(ramp, 1)

        # The following is a workaround. In inverse mode, a ramp with 0 duration
        # (that is all 1's) becomes all 0's, which is a problem for the
        # ultimate calculation of the ADSR signal (a * d * r => 0's). So this
        # replaces only rows who sum to 0 (i.e., all components are zero).

        if inverse:
            ramp = jnp.where(duration > 0.0, 1.0 - ramp, ramp)

        # Apply scaling factor.
        ramp = jnp.power(ramp, self.parameters["alpha"].from_0to1())
        return ramp

    def make_attack(self, attack_time) -> Signal:
        """
        Builds the attack portion of the envelope.

        Args:
            attack_time (TODO): Length of the attack in seconds.
        """
        return self.ramp(attack_time)

    def make_decay(self, attack_time, decay_time) -> Signal:
        """
        Creates the decay portion of the envelope.

        Args:
            attack_time (TODO): Length of the attack in seconds.
            decay_time (TODO): Length of the decay time in seconds.
        """
        sustain = self.parameters["sustain"].from_0to1()
        a = 1.0 - sustain
        b = self.ramp(decay_time, start=attack_time, inverse=True)
        return jnp.squeeze(a * b + sustain)

    def make_release(self, note_on_duration) -> Signal:
        """
        Creates the release portion of the envelope.

        Args:
            note_on_duration (TODO): Duration of midi note in seconds (release starts
                when the midi note is released).
        """
        return self.ramp(
            self.parameters["release"].from_0to1(),
            start=note_on_duration,
            inverse=True
        )


class VCO(SynthModule):
    """
    Base class for voltage controlled oscillators.

    Think of this as a VCO on a modular synthesizer. It has a base pitch
    (specified here as a midi value), and a pitch modulation depth. Its call
    accepts a modulation signal between [-1, 1]. An array of 0's returns a
    stationary audio signal at its base pitch.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
        parameter_ranges (Dict[ParameterName, :class:`~synthax.parameter.ModuleParameterRange`]): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    # TODO: Allow parameter_ranges to be partial
    parameter_ranges: Optional[Dict[ParameterName, ModuleParameterRange]] = field(
        default_factory = lambda: {
            "tuning": ModuleParameterRange(
                minimum=-24.0,
                maximum=24.0,
            ),
            "mod_depth": ModuleParameterRange(
                minimum=-96.0,
                maximum=96.0,
                curve=0.2,
                symmetric=True,
            ),
            "initial_phase": ModuleParameterRange(
                minimum=-jnp.pi,
                maximum=jnp.pi,
            )
        })

    def setup(self):
        # TODO: Refactor if possible
        self.parameters = {
            name: ModuleParameter(
                name,
                parameter_range,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for name, parameter_range in self.parameter_ranges.items()
        }

    def __call__(
            self,
            midi_f0: chex.Array,
            mod_signal: Optional[Signal] = None
    ) -> Signal:
        """
        Generates audio signal from modulation signal.

        Args:
            midi_f0 (TODO): Fundamental of note in midi note value (0 - 127).
            mod_signal (TODO): Modulation signal to apply to the pitch.
        """

        control_as_frequency = self.make_control_as_frequency(midi_f0, mod_signal)
        cosine_argument = self.make_argument(control_as_frequency)
        cosine_argument += self.parameters["initial_phase"].from_0to1()
        signal = self.oscillator(cosine_argument, midi_f0)
        return self.to_buffer_size(signal)

    def make_control_as_frequency(
            self,
            midi_f0: chex.Array,
            mod_signal: Optional[Signal] = None
    ) -> Signal:
        """
        Generates a time-varying control signal in frequency (Hz) from a midi
        fundamental pitch and pitch-modulation signal.

        Args:
            midi_f0 (TODO): Fundamental pitch value in midi.
            mod_signal (TODO): Pitch modulation signal in midi.
        """
        midi_f0 = (midi_f0 + self.parameters["tuning"].from_0to1())

        # If there is no modulation, then convert the midi_f0 values to
        # frequency and return an expanded view that contains buffer size
        # number of values
        if mod_signal is None:
            control_hz = midi_to_hz(midi_f0)
            return jnp.broadcast_to(control_hz, (-1, self.buffer_size))

        # If there is modulation, then add that to the fundamental,
        # clamp to a range [0.0, 127.0], then return in frequency Hz.
        modulation = self.parameters["mod_depth"].from_0to1() * mod_signal
        control = jax.lax.clamp(0.0, midi_f0 + modulation, 127.0)
        return midi_to_hz(control)

    def make_argument(self, freq: Signal) -> Signal:
        """
        Generates the phase argument to feed an oscillating function to
        generate an audio signal.

        Args:
            freq (TODO): Time-varying instantaneous frequency in Hz.
        """
        return jnp.cumsum(2 * jnp.pi * freq / self.sample_rate, axis=1)

    def oscillator(self, argument: Signal, *args: Tuple, **kwargs: Dict) -> Signal:
        """
        This function accepts a phase argument and generates output audio. It is
        implemented by the child class.

        Args:
            argument (TODO): The phase of the oscillator at each time sample.
        """
        raise NotImplementedError("Derived classes must override this method")


class SineVCO(VCO):
    """
    Simple VCO that generates a pitched sinusoid.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    def oscillator(self, argument: Signal, midi_f0: chex.Array) -> Signal:
        """
        A cosine oscillator.

        Args:
            argument (TODO): The phase of the oscillator at each time sample.
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
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    # TODO: We include this override to output to make mod_signal non-optional
    def __call__(self, midi_f0: chex.Array, mod_signal: Signal) -> Signal:
        """
        Args:
            midi_f0 (TODO): note value in midi
            mod_signal (TODO): audio rate frequency modulation signal
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
            midi_f0 (TODO): Fundamental frequency in midi.
            mod_signal (TODO): FM modulation signal (interpreted as modulation index).
        """
        # Compute modulation in Hz space (rather than midi-space).
        f0_hz = midi_to_hz(midi_f0 + self.parameters["tuning"].from_0to1())
        fm_depth = self.parameters["mod_depth"].from_0to1() * f0_hz
        modulation_hz = fm_depth * mod_signal
        return jax.lax.clamp(0.0, f0_hz + modulation_hz, self.nyquist)

    def oscillator(self, argument: Signal, midi_f0: chex.Array) -> Signal:
        """
        A cosine oscillator. ...Good ol' cosine.

        Args:
            argument (TODO): The phase of the oscillator at each time sample.
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
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
        parameter_ranges (Dict[ParameterName, :class:`~synthax.parameter.ModuleParameterRange`]): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    # TODO: Allow parameter_ranges to be partial
    parameter_ranges: Optional[Dict[ParameterName, ModuleParameterRange]] = field(
        default_factory = lambda: {
            "tuning": ModuleParameterRange(
                minimum=-24.0,
                maximum=24.0,
            ),
            "mod_depth": ModuleParameterRange(
                minimum=-96.0,
                maximum=96.0,
                curve=0.2,
                symmetric=True,
            ),
            "initial_phase": ModuleParameterRange(
                minimum=-jnp.pi,
                maximum=jnp.pi,
            ),
            "shape": ModuleParameterRange(
                minimum=0.0,
                maximum=1.0,
            )
        })

    def oscillator(self, argument: Signal, midi_f0: chex.Array) -> Signal:
        """
        Generates output square/saw audio given a phase argument.

        Args:
            argument (TODO): The phase of the oscillator at each time sample.
            midi_f0 (TODO): Fundamental frequency in midi.
        """
        partials = self.partials_constant(midi_f0)
        square = jnp.tanh(jnp.pi * partials * jnp.sin(argument) / 2)
        shape = self.parameters["shape"].from_0to1()
        return (1 - shape / 2) * square * (1 + shape * jnp.cos(argument))

    def partials_constant(self, midi_f0):
        """
        Calculates a value to determine the number of overtones in the resulting
        square / saw wave, in order to keep aliasing at an acceptable level.
        Higher fundamental frequencies require fewer partials for a rich sound;
        lower-frequency sounds can safely have more partials without causing
        audible aliasing.

        Args:
            midi_f0 (TODO): Fundamental frequency in midi.
        """
        max_pitch = (
            midi_f0 + self.parameters["tuning"].from_0to1() + jnp.maximum(self.parameters["mod_depth"].from_0to1(), 0)
        )
        max_f0 = midi_to_hz(max_pitch)
        return 12000 / (max_f0 * jnp.log10(max_f0))


class VCA(SynthModule):
    """
    Voltage controlled amplifier.

    The VCA shapes the amplitude of an audio input signal over time, as
    determined by a control signal. To shape control-rate signals, use
    :class:`synthax.module.ControlRateVCA`.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    def __call__(self, audio_in: Signal, control_in: Signal):
        signal = audio_in * control_in
        return self.to_buffer_size(signal)


class ControlRateVCA(ControlRateModule):
    """
    Voltage controlled amplifier.

    The VCA shapes the amplitude of a control input signal over time, as
    determined by another control signal. To shape audio-rate signals, use
    :class:`synthax.module.VCA`.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    def __call__(self, audio_in: Signal, control_in: Signal):
        signal = audio_in * control_in
        return self.to_buffer_size(signal)


class Noise(SynthModule):
    """
    Generates white noise that is the same length as the buffer.

    TODO:

    For performance noise is pre-computed. In order to maintain
    reproducibility noise must be computed on the CPU and then transferred
    to the GPU, if a GPU is being used. We pre-compute
    :attr:`~synthax.config.BASE_REPRODUCIBLE_BATCH_SIZE`
    samples of noise and then repeat those for larger batch sizes.

    To keep things fast we only support multiples of
    :attr:`~synthax.config.BASE_REPRODUCIBLE_BATCH_SIZE`
    when reproducibility mode is enabled. For example, if you batch size
    is 4 times :attr:`~synthax.config.BASE_REPRODUCIBLE_BATCH_SIZE`, then
    you get the same noise signals repeated 4 times.

    `Note`: If you have multiple `Noise` modules in the same
    :class:`~synthax.synth.BaseSynth`, make sure you instantiate
    each `Noise` with a unique seed.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    def setup(self):
        self.parameters = {
            "noise": jax.random.uniform(
                self.PRNG_key,
                shape=(self.config.batch_size, self.buffer_size),
                minval=-1.0,
                maxval=1.0
            )
        }

    def __call__(self) -> Signal:
        signal = self.parameters["noise"]
        return self.to_buffer_size(signal)

class LFO(ControlRateModule):
    """
    Low Frequency Oscillator.

    The LFO shape can be any mixture of sine, triangle, saw, reverse saw, and
    square waves. Contributions of each base-shape are determined by the
    :attr:`~synthax.module.LFO.lfo_types` values, which are between 0 and 1.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (SynthConfig): See :class:`~synthax.module.SynthConfig`.
        exponent: A non-negative value that determines the discrimination of the
            soft-max selector for LFO shapes. Higher values will tend to favour
            one LFO shape over all others. Lower values will result in a more
            even blend of LFO shapes.
        parameter_ranges (Dict[ParameterName, :class:`~synthax.parameter.ModuleParameterRange`]): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    exponent: chex.Array = jnp.exp(1) # e
    # TODO: Allow parameter_ranges to be partial
    parameter_ranges: Optional[Dict[ParameterName, ModuleParameterRange]] = field(
        default_factory = lambda: {
            "frequency": ModuleParameterRange(
                minimum=0.0,
                maximum=20.0,
                curve=0.25,
            ),
            "mod_depth": ModuleParameterRange(
                minimum=-10.0,
                maximum=20.0,
                curve=0.5,
                symmetric=True,
            ),
            "initial_phase": ModuleParameterRange(
                minimum=-jnp.pi,
                maximum=jnp.pi,
            ),
            "sin": ModuleParameterRange(
                minimum=0.0,
                maximum=1.0,
            ),
            "tri": ModuleParameterRange(
                minimum=0.0,
                maximum=1.0,
            ),
            "saw": ModuleParameterRange(
                minimum=0.0,
                maximum=1.0,
            ),
            "rsaw": ModuleParameterRange(
                minimum=0.0,
                maximum=1.0,
            ),
            "sqr": ModuleParameterRange(
                minimum=0.0,
                maximum=1.0,
            )
        })

    def setup(self):
        # TODO: Refactor
        self.lfo_types = ["sin", "tri", "saw", "rsaw", "sqr"]
        self.parameters = {
            name: ModuleParameter(
                name,
                parameter_range,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for name, parameter_range in self.parameter_ranges.items()
        }

    def __call__(self, mod_signal: Optional[Signal] = None):
        """
        Generates low frequency oscillator control signal.

        Args:
            mod_signal (TODO):  LFO rate modulation signal in Hz. To modulate the
                depth of the LFO, use :class:`synthax.module.ControlRateVCA`.
        """
        # Create frequency signal
        frequency = self.make_control(mod_signal)
        argument = jnp.cumsum(2 * jnp.pi * frequency / self.control_rate, axis=1)
        argument += self.parameters["initial_phase"].from_0to1()

        # Get LFO shapes
        shapes = jnp.stack(self.make_lfo_shapes(argument), axis=1)

        # Apply mode selection to the LFO shapes
        mode = jnp.stack(
            [self.parameters[lfo].from_0to1() for lfo in self.lfo_types],
            axis=1
        )
        mode = jnp.power(mode, self.exponent)
        mode /= jnp.sum(mode, axis=1, keepdims=True)

        signal = jnp.matmul(mode, shapes).squeeze(1)
        return self.to_buffer_size(signal)

    def make_control(self, mod_signal: Optional[Signal] = None) -> Signal:
        """
        Applies the LFO-rate modulation signal to the LFO base frequency.

        Args:
            mod_signal (TODO): Modulation signal in Hz. Positive values increase the
                LFO base rate; negative values decrease it.
        """
        frequency = self.parameters["initial_phase"].from_0to1()

        # If no modulation, then return a view of the frequency of this
        # LFO expanded to the control buffer size
        if mod_signal is None:
            return jnp.broadcast_to(frequency, (-1, self.control_buffer_size))

        modulation = self.parameters["mod_depth"].from_0to1() * mod_signal
        return jnp.maximum(frequency + modulation, 0.0)

    def make_lfo_shapes(
            self,
            argument: Signal
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Generates five separate signals for each LFO shape and returns them as a
        tuple, to be mixed by :func:`synthax.module.LFO.__call__`.

        Args:
            argument (TODO): Time-varying phase to generate LFO signals.
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


class ModulationMixer(SynthModule):
    """
    A modulation matrix that combines :math:`N` input modulation signals to make
    :math:`M` output modulation signals. Each output is a linear combination of
    all in input signals, as determined by an :math:`N \times M` mixing matrix.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        n_input (int): Number of input signals to module mix.
        n_output (int): Number of output signals to generate.
        input_names (List(str)): TODO
        output_names (List(str)): TODO
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    n_input: int
    n_output: int
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
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
                    name,
                    self.parameter_ranges,
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
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        n_input (int): TODO
        curves (List[float]): TODO
        names (List[str]): TODO
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    n_input: int
    curves: Optional[List[float]] = None
    names: Optional[List[str]] = None
    parameter_ranges: Optional[ModuleParameterRange] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
        curve=1.0,
    )

    def setup(self):
        names = [f"level{i}" if self.names is None else self.names[i]
                      for i in range(self.n_input)]
        if self.curves is None:
            self.curves = [1.0] * self.n_input

        parameter_ranges = [self.parameter_ranges] * self.n_input
        for i, parameter_range in enumerate(parameter_ranges):
            parameter_range.curve = self.curves[i]

        # TODO: Refactor if possible
        self.parameters = {
            name: ModuleParameter(
                name,
                parameter_range,
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


class ControlRateUpsample(SynthModule):
    """
    Upsample control signals to the global sampling rate

    Uses linear interpolation to resample an input control signal to the
    audio buffer size set in config.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Configuration.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    def __call__(self, signal):
        upsampled_signal = jax.image.resize(
            signal,
            (1, self.config.buffer_size),
            method="linear"
        )
        return self.to_buffer_size(upsampled_signal)


class CrossfadeKnob(SynthModule):
    """
    Crossfade knob parameter with no signal generation

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    parameter_ranges: Optional[ModuleParameterRange] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )

    def setup(self):
        # TODO: Refactor if possible
        self.parameters = {
            "ratio": ModuleParameter(
                "ratio",
                self.parameter_ranges,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
        }

    def __call__(self):
        # TODO: Implement?
        raise NotImplementedError("Must override the __call__ method")


class MonophonicKeyboard(SynthModule):
    """
    A keyboard controller module. Mimics a mono-synth keyboard and contains
    parameters that output a midi_f0 and note duration.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        parameter_ranges (Dict[ParameterName, :class:`~synthax.parameter.ModuleParameterRange`]): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    # TODO: Allow parameter_ranges to be partial (e.g. only midi_f0 and default duration)
    parameter_ranges: Optional[Dict[ParameterName, ModuleParameterRange]] = field(
        default_factory = lambda: {
            "midi_f0": ModuleParameterRange(
                minimum=0.0,
                maximum=127.0,
                curve=1.0,
            ),
            "duration": ModuleParameterRange(
                minimum=0.01,
                maximum=4.0,
                curve=0.5,
            )
        })

    def setup(self):
        # TODO: Refactor if possible
        self.parameters = {
            name: ModuleParameter(
                name,
                parameter_range,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for name, parameter_range in self.parameter_ranges.items()
        }

    def __call__(self):
        # TODO: Refactor
        return self.parameters["midi_f0"].from_0to1(), self.parameters["duration"].from_0to1()


class SoftModeSelector(SynthModule):
    """
    A soft mode selector.
    If there are n different modes, return a probability distribution over them.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
        n_modes (int): TODO
        exponent (chex.Array): determines how strongly to scale each [0,1] value prior to normalization.
        parameter_ranges (:class:`~synthax.parameter.ModuleParameterRange`): TODO.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
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
                f"mode{i}weight",
                self.parameter_ranges,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for i in range(n_modes)
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

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
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
                f"mode{i}weight",
                self.parameter_ranges,
                value=jax.random.uniform(
                    self.PRNG_key,
                    shape=(self.config.batch_size,)
                )
            )
            for i in range(n_modes)
        }

    def __call__(self):
        # TODO: .value?
        # TODO: Refactor get all values in SynthModule
        parameter_values = [p.value for k, p in self.parameters.items()]
        idx = jnp.argmax(parameter_values, axis=0)
        return jax.nn.one_hot(idx, num_classes=len(parameter_values)).T
