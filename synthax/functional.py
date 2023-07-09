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
import flax
import chex
from typing import Tuple
from synthax.types import Signal

def midi_to_hz(midi: chex.Array) -> chex.Array:
    """
    Convert from midi (linear pitch) to frequency in Hz.

    Args:
        midi (chex.Array): Linear pitch on the MIDI scale.

    Return:
        Frequency in Hz.
    """
    return 440.0 * (jnp.exp2((midi - 69.0) / 12.0))

def normalize_if_clipping(signal: Signal) -> Signal:
    """
    Only normalize individual signals in batch that have samples
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
