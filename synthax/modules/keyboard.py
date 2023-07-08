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
from synthax.modules.base import ControlRateModule
from synthax.parameter import ModuleParameterRange, from_0to1
from synthax.config import SynthConfig
from synthax.types import ParameterSpec
from typing import Optional


class MonophonicKeyboard(ControlRateModule):
    """
    A keyboard controller module. Mimics a mono-synth keyboard and contains
    parameters that output a midi_f0 and note duration.

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        midi_f0 (ParameterSpec): Accepts a parameter range, initial values or both.
        duration (ParameterSpec): Accepts a parameter range, initial values or both.
    """

    midi_f0: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.0,
        maximum=127.0,
        curve=1.0,
    )
    duration: Optional[ParameterSpec] = ModuleParameterRange(
        minimum=0.01,
        maximum=4.0,
        curve=0.5,
    )

    def __call__(self):
        midi_f0 = from_0to1(self._midi_f0, self._midi_f0_range)
        duration = from_0to1(self._duration, self._duration_range)
        return midi_f0, duration
