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

import json
import yaml
import jax
import jax.numpy as jnp
import flax
from synthax.synth import BaseSynth
from synthax.config import SynthConfig
from typing import Optional, NamedTuple, Mapping


class SynthSpec(NamedTuple):
    config: Optional[SynthConfig]
    params: Optional[flax.core.frozen_dict.FrozenDict]
    PRNGkey: Optional[jax.random.PRNGKey]


def write_synthspec(
    outfile: str,
    synth: BaseSynth,
    params: Optional[dict] = None,
    PRNGkey: Optional[jax.random.PRNGKey] = None,
    format: str = "yaml",
    **kwargs
):
    """
    Write a synth spec to disk.

    Args:
        output_path (str): Path to write synth to.
        synth (:class:`~synthax.synth.BaseSynth`): Synth to write.
        config (:class:`~synthax.config.SynthConfig`): Configuration to write.
        params (dict): Parameters to write.
        format (str): Format to write synth in.
    """

    outd = {**kwargs}

    def params_tovals(d):
        if isinstance(d, Mapping):
            for k, v in d.items():
                d = dict(d)
                d[k] = params_tovals(v)
        if isinstance(d, jnp.ndarray):
            return d.tolist()
        return d

    params = params_tovals(params)

    if params is not None:
        outd = {"params": params} if "params" not in params else dict(params)

    if synth is not None:
        outd["config"] = synth.config.__dict__

    if PRNGkey is not None:
        outd["prng_key"] = PRNG_key.tolist()[-1]

    outdata = None
    if format in ("yaml", "yml"):
        outdata = yaml.dump(outd, Dumper=yaml.CDumper)
    elif format == "json":
        outdata = json.dumps(outd, indent=4)
    else:
        raise ValueError("Format should bs yaml or json.")

    if not outfile.endswith(format):
        outfile = "%s.%s" % (outfile, format)

    with open(outfile, "w") as outfile:
        outfile.write(outdata)


def read_synthspec(filename: str) -> SynthSpec:
    """Read a synth spec from disk.

    Args:
        infile (str): Path to synth spec.
    """

    with open(filename, "r") as infile:
        data = infile.read()

    if filename.endswith("yaml") or filename.endswith("yml"):
        data = yaml.load(data, Loader=yaml.CLoader)
    elif filename.endswith("json"):
        data = json.loads(data)
    else:
        raise ValueError("Format should bs yaml or json.")

    if "config" in data:
        config = SynthConfig(**data["config"])
    else:
        config = None

    if "prng_key" in data:
        prng_key = jax.random.PRNGKey(data["prng_key"])
    else:
        prng_key = None

    params = data.get("params", None)

    def vals_toparams(d):
        if isinstance(d, Mapping):
            for k, v in d.items():
                d[k] = vals_toparams(v)
        if isinstance(d, (list, tuple)):
            return jnp.array(d)
        return d

    if params is not None:
        params = flax.core.frozen_dict.freeze({"params": vals_toparams(params)})

    return SynthSpec(config=config, params=params, PRNGkey=prng_key)
