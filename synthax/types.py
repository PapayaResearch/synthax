import chex
from typing import NewType


Signal = NewType("Signal", chex.Array)
ParameterName = NewType("ParameterName", str)