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

import jax
import chex
from flax import linen as nn
from flax import struct
import jax.numpy as jnp
from typing import Callable, NamedTuple

@struct.dataclass
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

    minimum: float
    maximum: float
    curve: float = 1.0
    symmetric: bool = False


def from_0to1(normalized: jax.typing.ArrayLike, range: ModuleParameterRange) -> jax.typing.ArrayLike:
    """
    Set value of this parameter using a normalized value in the range [0,1]

    Args:
        normalized (float): value within machine-readable range [0, 1] to convert to
            human-readable range [minimum, maximum].
    """
    if not range.symmetric:
        if range.curve != 1.0:
            normalized = jnp.exp2(jnp.log2(normalized) / range.curve)

        return range.minimum + (range.maximum - range.minimum) * normalized

    # Compute the curve for a symmetric curve
    dist = 2.0 * normalized - 1.0
    if range.curve != 1.0:
        normalized = jnp.where(
            dist == 0.0,
            dist,
            jnp.exp2(jnp.log2(jnp.abs(dist)) / range.curve) * jnp.sign(dist),
        )

    return range.minimum + (range.maximum - range.minimum) / 2.0 * (normalized + 1.0)


def to_0to1(value: jax.typing.ArrayLike, range: ModuleParameterRange) -> jax.typing.ArrayLike:
    """
    Convert from human-readable range [minimum, maximum] to machine-range [0, 1].

    Args:
        value (chex.Array): value within the range defined by minimum and maximum
    """
    normalized = (value - range.minimum) / (range.maximum - range.minimum)

    if not range.symmetric:
        if range.curve != 1:
            normalized = jnp.power(normalized, range.curve)
        return normalized

    dist = 2.0 * normalized - 1.0
    return (1.0 + jnp.power(jnp.abs(dist), range.curve) * jnp.sign(dist)) / 2.0


class ModuleParameterSpec(NamedTuple):
    range: ModuleParameterRange
    value: jax.typing.ArrayLike

class ModuleParameter(nn.Module):
    """
    Args:
        name (str): A name for this parameter
        range (ModuleParameterRange): Object that supports conversion
            between human-readable range and machine-readable [0,1] range.
        value (chex.Array): initial value of this parameter in the human-readable range.
    """

    name: str
    range: ModuleParameterRange
    value: chex.Array

    def setup(self):
        self._value = self.param(
            self.name,
            lambda _: self.value
        )

    def from_0to1(self) -> chex.Array:
        """
        Get the value of this parameter in the human-readable range.
        """
        return from_0to1(self._value, self.range)

    def __repr__(self):
        return f"ModuleParameter({self.__dict__})"
