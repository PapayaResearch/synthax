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


class ADSR(ControlRateModule):
    """
    Envelope class for building a control-rate ADSR signal.

    Args:
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
        attack (ParameterSpec): Accepts a parameter range, initial values or both.
        decay (ParameterSpec): Accepts a parameter range, initial values or both.
        sustain (ParameterSpec): Accepts a parameter range, initial values or both.
        release (ParameterSpec): Accepts a parameter range, initial values or both.
        alpha (ParameterSpec): Accepts a parameter range, initial values or both.
    """

    attack: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=2.0,
        curve=0.5,
    )
    decay: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=2.0,
        curve=0.5,
    )
    sustain: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=1.0,
    )
    release: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=5.0,
        curve=0.5,
    )
    alpha: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.1,
        maximum=6.0,
    )

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
            note_on_duration (chex.Array): Duration of note on event in seconds.
        """

        attack = from_0to1(self._attack, self._attack_range)
        decay = from_0to1(self._decay, self._decay_range)

        # Calculations to accommodate attack/decay phase cut by note duration.
        new_attack = jnp.minimum(attack, note_on_duration)
        new_decay = jnp.maximum(note_on_duration - attack, 0)
        new_decay = jnp.minimum(new_decay, decay)

        attack_signal = self.make_attack(new_attack)
        decay_signal = self.make_decay(new_attack, new_decay)
        release_signal = self.make_release(note_on_duration)

        signal = attack_signal * decay_signal * release_signal

        return signal

    def ramp(
            self,
            duration: chex.Array,
            start: Optional[chex.Array] = None,
            inverse: Optional[bool] = False
    ) -> Signal:
        """
        Makes a ramp of a given duration in seconds.

        The construction of this matrix is rather cryptic. Essentially, this
        method works by tilting and clipping ramps between 0 and 1, then
        applying a scaling factor :attr:`~alpha`.

        Args:
            duration (chex.Array): Length of the ramp in seconds.
            start (chex.Array): Initial delay of ramp in seconds.
            inverse (bool): Toggle to flip the ramp from ascending to descending.
        """

        duration = jnp.expand_dims(self.seconds_to_samples(duration), axis=1)

        # Convert to number of samples.
        if start is not None:
            start = jnp.expand_dims(self.seconds_to_samples(start), axis=1)
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
        ramp = jnp.power(
            ramp,
            jnp.expand_dims(from_0to1(self._alpha, self._alpha_range), axis=1)
        )
        return ramp

    def make_attack(self, attack_time) -> Signal:
        """
        Builds the attack portion of the envelope.

        Args:
            attack_time (chex.Array): Length of the attack in seconds.
        """
        return self.ramp(attack_time)

    def make_decay(self, attack_time, decay_time) -> Signal:
        """
        Creates the decay portion of the envelope.

        Args:
            attack_time (chex.Array): Length of the attack in seconds.
            decay_time (chex.Array): Length of the decay time in seconds.
        """
        sustain = jnp.expand_dims(
            from_0to1(self._sustain, self._sustain_range),
            axis=1
        )
        a = 1.0 - sustain
        b = self.ramp(decay_time, start=attack_time, inverse=True)
        return jnp.squeeze(a * b + sustain)

    def make_release(self, note_on_duration) -> Signal:
        """
        Creates the release portion of the envelope.

        Args:
            note_on_duration (chex.Array): Duration of midi note in seconds (release starts
                when the midi note is released).
        """
        return self.ramp(
            from_0to1(self._release, self._release_range),
            start=note_on_duration,
            inverse=True
        )
