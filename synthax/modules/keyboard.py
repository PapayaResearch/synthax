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
from dataclasses import field
from flax import linen as nn
from synthax.modules.base import ControlRateModule
from synthax.parameter import ModuleParameter, ModuleParameterRange
from synthax.config import SynthConfig
from synthax.types import ParameterName
from typing import Optional


class MonophonicKeyboard(ControlRateModule):
    """
    A keyboard controller module. Mimics a mono-synth keyboard and contains
    parameters that output a midi_f0 and note duration.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Configuration.
        parameter_ranges (Dict[ParameterName, :class:`~synthax.parameter.ModuleParameterRange`]): TODO.
    """

    # TODO: Allow parameter_ranges to be partial (e.g. only midi_f0 and default duration)
    parameter_ranges: Optional[dict[ParameterName, ModuleParameterRange]] = field(
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
                name=name,
                range=parameter_range,
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