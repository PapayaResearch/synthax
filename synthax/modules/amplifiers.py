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
from flax import linen as nn
from synthax.modules.base import SynthModule, ControlRateModule
from synthax.config import SynthConfig
from synthax.types import Signal


class VCA(SynthModule):
    """
    Voltage controlled amplifier.

    The VCA shapes the amplitude of an audio input signal over time, as
    determined by a control signal. To shape control-rate signals, use
    :class:`synthax.module.ControlRateVCA`.

    Args:
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
    """

    def __call__(self, audio_in: Signal, control_in: Signal):
        signal = audio_in * control_in
        return signal


class ControlRateVCA(ControlRateModule):
    """
    Voltage controlled amplifier.

    The VCA shapes the amplitude of a control input signal over time, as
    determined by another control signal. To shape audio-rate signals, use
    :class:`synthax.module.VCA`.

    Args:
        config (SynthConfig): See :class:`~synhtax.module.SynthModule`
    """

    def __call__(self, audio_in: Signal, control_in: Signal):
        signal = audio_in * control_in
        return signal
