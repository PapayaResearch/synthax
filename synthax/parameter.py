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
Parameters for DDSP Modules
"""

import chex
import jax.numpy as jnp
from typing import Optional, NewType

ParameterName = NewType('ParameterName', str)

class ModuleParameterRange:
    """
    `ModuleParameterRange` class is a structure for keeping track of
    the specific range that a parameter might take on. Also handles
    functionality for converting between machine-readable range [0, 1] and a
    human-readable range [minimum, maximum].

    Args:
        minimum (float): minimum value in human-readable range
        maximum (float): maximum value in human-readable range
        curve (float): strictly positive shape of the curve
            values less than 1 place more emphasis on smaller
            values and values greater than 1 place more emphasis on
            larger values. 1 is linear.
        symmetric (bool): whether or not the parameter range is symmetric,
            allows for curves around a center point. When this is True,
            a curve value of one is linear, greater than one emphasizes
            the minimum and maximum, and less than one emphasizes values
            closer to :math:`(maximum - minimum)/2`.
    """

    def __init__(
            self,
            minimum: float,
            maximum: float,
            curve: float = 1.0,
            symmetric: bool = False,
    ):
        self.minimum = minimum
        self.maximum = maximum
        self.curve = curve
        self.symmetric = symmetric

    def __repr__(self):
        return f"ModuleParameterRange({self.__dict__})"

    def from_0to1(self, normalized: float) -> float:
        """
        Set value of this parameter using a normalized value in the range [0,1]

        Args:
            normalized (float): value within machine-readable range [0, 1] to convert to
                human-readable range [minimum, maximum].
        """
        if not self.symmetric:
            if self.curve != 1.0:
                normalized = jnp.exp2(jnp.log2(normalized) / self.curve)

            return self.minimum + (self.maximum - self.minimum) * normalized

        # Compute the curve for a symmetric curve
        dist = 2.0 * normalized - 1.0
        if self.curve != 1.0:
            normalized = jnp.where(
                dist == 0.0,
                dist,
                jnp.exp2(jnp.log2(jnp.abs(dist)) / self.curve) * jnp.sign(dist),
            )

        return self.minimum + (self.maximum - self.minimum) / 2.0 * (normalized + 1.0)

    def to_0to1(self, value: chex.Array) -> chex.Array:
        """
        Convert from human-readable range [minimum, maximum] to machine-range [0, 1].

        Args:
          value (chex.Array): value within the range defined by minimum and maximum
        """
        normalized = (value - self.minimum) / (self.maximum - self.minimum)

        if not self.symmetric:
            if self.curve != 1:
                normalized = jnp.power(normalized, self.curve)
            return normalized

        dist = 2.0 * normalized - 1.0
        return (1.0 + jnp.power(jnp.abs(dist), self.curve) * jnp.sign(dist)) / 2.0


class ModuleParameter:
    """
    Args:
        name (str): A name for this parameter
        range (ModuleParameterRange): Object that supports conversion
            between human-readable range and machine-readable [0,1] range.
        value (chex.Array): initial value of this parameter in the human-readable range.
    """

    def __init__(
            self,
            name: str,
            range: ModuleParameterRange,
            value: chex.Array,
    ):
        self.name = name
        self.range = range
        # TODO: Flax self.param
        self.to_0to1(value)

    def set_value(self, value: chex.Array):
        """
        Set the value of this parameter using an input that is
        in the machine-readable range.

        Args:
            v (chex.Array): Value to update this parameter
        """
        self.value = value

    def from_0to1(self) -> chex.Array:
        """
        Get the value of this parameter in the human-readable range.
        """
        return self.range.from_0to1(self.value)

    def to_0to1(self, value: chex.Array):
        """
        Set the value of this parameter using an input that is
        in the human-readable range.

        Args:
            v (chex.Array): Value to update this parameter
        """
        self.value = self.range.to_0to1(value)

    def __repr__(self):
        return f"ModuleParameter({self.__dict__})"
