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
from synthax.types import Signal


class LPF(SynthModule):
    """
    Simple time-veritying low-pass filter.

    Args:
        order (int): Order of the filter.
    """
    order: int

    def __call__(self, audio_in: Signal, control_in: Signal):
        def tv_lpf(input_signal: Signal, input_control: Signal) -> Signal:
            dt = 1/self.config.sample_rate
            taus = 1.0 / (2 * jnp.pi * input_control)

            alphas = dt / (taus + dt)

            state = jnp.zeros(self.order)

            def one_step(state, inputs):
                alpha, x = inputs
                new_state = alpha * x + (1 - alpha) * state
                return new_state, new_state

            _, outputs = jax.lax.scan(
                one_step,
                state,
                jnp.stack([alphas, input_signal], axis=-1)
            )
            return outputs[:, -1]

        return jax.vmap(tv_lpf)(audio_in, control_in)
