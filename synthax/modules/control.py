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
from flax import linen as nn
from synthax.modules.base import SynthModule
from synthax.config import SynthConfig
from synthax.types import Signal


class ControlRateUpsample(SynthModule):
    """
    Upsample control signals to the global sampling rate

    Uses linear interpolation to resample an input control signal to the
    audio buffer size set in config.

    Args:
        config (:class:`~synthax.config.SynthConfig`): Configuration.
    """

    def __call__(self, signal):
        def vmap_interp(s: Signal):
            return jnp.interp(
                jnp.linspace(0, self.config.control_buffer_size, self.config.buffer_size),
                jnp.linspace(0, self.config.control_buffer_size, self.config.control_buffer_size),
                s
            )
        upsampled_signal = jax.vmap(vmap_interp)(signal)
        return upsampled_signal
