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
Global configuration for :class:`~synthax.synth.BaseSynth` and its
components :class:`~synthax.module.SynthModule`.
"""

import jax
import jax.numpy as jnp
from typing import Optional

class SynthConfig:
    """
    Any :class:`~synthax.module.SynthModule` and
    :class:`~synthax.synth.BaseSynth` might use these global
    configuration values.

    Args:
        batch_size (int): Positive number of sounds to generate.
        sample_rate (int): Sample rate for audio generation (e.g. 16000, 44100, 48000).
        buffer_size (float): Duration of the sound in seconds.
        control_rate (int): Sample rate for control signal generation.
        device (str): Device to allocate tensors.
    """

    def __init__(
            self,
            batch_size: int,
            sample_rate: Optional[int] = 44100,
            buffer_size_seconds: Optional[float] = 3.0,
            control_rate: Optional[int] = 441,
            eps: float = 1e-6,
            **kwargs
    ):
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.buffer_size_seconds = buffer_size_seconds
        self.buffer_size = jnp.ceil(buffer_size_seconds * sample_rate).astype(int)
        self.control_rate = control_rate
        self.control_buffer_size = jnp.ceil(buffer_size_seconds * control_rate).astype(int)
        self.eps = eps

        # Store any additional metadata
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"SynthConfig({self.__dict__})"
