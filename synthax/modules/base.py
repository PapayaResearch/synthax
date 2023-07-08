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
import dataclasses
from flax import linen as nn
from synthax.config import SynthConfig
from synthax.parameter import ModuleParameterRange, ModuleParameterSpec, to_0to1
from synthax.types import Signal, is_parameter_spec


class SynthModule(nn.Module):
    """
    A base class for synthesis modules. A :class:`~.SynthModule`
    optionally takes input from other :class:`~.SynthModule` instances.

    All :class:`~.SynthModule` objects should be atomic, i.e., they
    should not contain other :class:`~.SynthModule` objects. This
    design choice is in the spirit of modular synthesis.

    Args:
        config (SynthConfig): An object containing synthesis settings
            that are shared across all modules.
    """

    config: SynthConfig

    def setup(self):
        # Filter all ParameterSpec default ranges
        default_ranges = {f.name: f.default
                          for f in dataclasses.fields(self) if is_parameter_spec(f.type)}
        for name, default_range in default_ranges.items():
            self._init_param(name, default_range)

    def _init_param(self, name, default_range, initializer=None):
        def init_param(PRNG_key: jax.random.PRNGKey):
            parameter_spec = getattr(self, name)
            if isinstance(parameter_spec, jnp.ndarray):
                # Values are assumed to be within the parameter range
                parameter_range = default_range
                value = parameter_spec
            if isinstance(parameter_spec, ModuleParameterRange):
                parameter_range = parameter_spec
                if initializer is None:
                    # Random initialization uniformly within the parameter range
                    value = jax.random.uniform(
                        PRNG_key,
                        shape=(self.config.batch_size,),
                        minval=parameter_range.minimum,
                        maxval=parameter_range.maximum
                    )
                else:
                    # Custom initialization
                    value = initializer(PRNG_key, parameter_range)

            if isinstance(parameter_spec, ModuleParameterSpec):
                # Values are assumed to be within the parameter range
                parameter_range = parameter_spec.range_
                value = parameter_spec.value

            setattr(
                self,
                "_"+name+"_range",
                parameter_range
            )

            # Save value in 0-1 range
            return to_0to1(value, parameter_range)

        setattr(
            self,
            "_"+name,
            self.param(
                name,
                lambda rng: init_param(rng)
            )
        )

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

    def seconds_to_samples(self, seconds: float):
        """
        Convenience function to calculate the number of samples corresponding to
        given a time value and :attr:`~.sample_rate`. Returns a possibly
        fractional value.

        Args:
            seconds (float): Time value in seconds.
        """
        return seconds * self.sample_rate


class ControlRateModule(SynthModule):
    """
    An abstract base class for non-audio modules that adapts the functions of
    :class:`.~SynthModule` to run at :attr:`~.ControlRateModule.control_rate`.

    Args:
        config (SynthConfig): An object containing synthesis settings
            that are shared across all modules.
    """

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
