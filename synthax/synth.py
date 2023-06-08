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
:class:`~synthax.module.SynthModule` wired together form a modular synthesizer.
"""

import os
import yaml
import jax
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import OrderedDict as OrderedDictType
from flax import linen as nn
from synthax.config import SynthConfig
from synthax.parameter import ModuleParameter
from synthax.module import (
    ADSR,
    LFO,
    VCA,
    AudioMixer,
    ControlRateUpsample,
    ControlRateVCA,
    ModulationMixer,
    MonophonicKeyboard,
    Noise,
    SineVCO,
    FmVCO,
    SquareSawVCO,
    SynthModule,
)


class BaseSynth(nn.Module):
    """
    Base class for synthesizers that combine one or more
    :class:`~synthax.module.SynthModule` to create a full synth
    architecture.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig

    def setup(self):
        """
        Each `BaseSynth` should override this.
        """
        raise NotImplementedError("Must override the setup method")

    def __call__(self):
        """
        Each `BaseSynth` should override this.
        """
        raise NotImplementedError("Must override the __call__ method")

    @property
    def batch_size(self):
        return self.config.batch_size

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @property
    def buffer_size(self):
        return self.config.buffer_size

    @property
    def buffer_size_seconds(self):
        return self.config.buffer_size_seconds

    def get_parameters(self) -> OrderedDictType[Tuple[str, str], ModuleParameter]:
        """
        Returns a dictionary of
        :class:`~synthax.parameter.ModuleParameterRange` for this synth,
        keyed on a tuple of the :class:`~synthax.module.SynthModule` name
        and the parameter name.
        """

        # Each parameter in this synth will have a unique combination of module name
        # and parameter name -- create a dictionary keyed on that.
        parameters = []
        for name, module in self:
            # TODO: module.parameters() doesn't exist by default
            for parameter in module.params():
                parameters.append(
                    ((module_name, parameter.parameter_name),
                     parameter)
                )

        return OrderedDict(parameters)

    def set_parameters(self, params: Dict[Tuple[str, str], float]):
        """
        Set various :class:`~synthax.parameter.ModuleParameter` for this synth.

        Args:
            params: Module and parameter strings, with the corresponding value.
        """
        for (module_name, param_name), value in params.items():
            module = getattr(self, module_name)
            module.set_parameter(param_name, value)

    @property
    def hyperparameters(self) -> OrderedDictType[Tuple[str, str, str], Any]:
        """
        Returns a dictionary of curve and symmetry hyperparameter values keyed
        on a tuple of the module name, parameter name, and hyperparameter name
        """
        hparams = []
        # TODO: Get parameters directly from the Module
        for (module_name, parameter_name), parameter in self.get_parameters().items():
            hparams.append(
                (
                    (module_name, parameter_name, "curve"),
                    parameter.parameter_range.curve,
                )
            )
            hparams.append(
                (
                    (module_name, parameter_name, "symmetric"),
                    parameter.parameter_range.symmetric,
                )
            )

        return OrderedDict(hparams)

    def set_hyperparameter(self, hyperparameter: Tuple[str, str, str], value: Any):
        """
        Set a hyperparameter. Pass in the module name, parameter name, and
        hyperparameter to set, and the value to set it to.

        Args:
            hyperparameter (TODO): Hyperparameter defined as the module name,
                parameter name and the type (curve/symmetric)
            value (float | bool): Value of the hyperparameter
        """
        #  TODO: Find a name for "hp" to define curve or symmetric
        # e.g. "adsr_1", "attack", "curve"
        module_name, parameter_name, hp  = hyperparameter
        module = getattr(self, module_name)
        parameter = module.get_parameter(parameter_name)
        setattr(parameter.parameter_range, hp, value)

    def save_synth(self, filename: str = 'conf/voice.yaml', indent: int = 4):
        """
        Save the synthesizer to a YAML file

        Args:
            filename (str): Saving path
            indent (int): Number of spaces to use as indentation
        """

        # e.g. key = ["adsr_1", "attack", "curve"]
        # hp = [{"name": key, "value": val} for key, val in self.hyperparameters.items()]

        with open(os.path.abspath(filename), 'w') as f:
            yaml.dump(self.modules, f, indent)

    def load_hyperparameters(self, filename: str) -> None:
        """
        Load hyperparameters from a YAML file

        Args:
            filename (str): YAML file containing the hyperparameters of a synth.
        """
        # TODO: Load from YAML file
        with open(os.path.abspath(filename), "r") as f:
            hyperparameters = json.load(f)

        # Update all hyperparameters in this synth
        for hp in hyperparameters:
            # e.g. {"name": ["adsr_1", "attack", "curve"], "value": 0.5}
            self.set_hyperparameter(hp["name"], hp["value"])

class Voice(BaseSynth):
    """
    The Voice architecture comprises the following modules: a
    :class:`~synthax.module.MonophonicKeyboard`, two
    :class:`~synthax.module.LFO`, six :class:`~synthax.module.ADSR`
    envelopes (each :class:`~synthax.module.LFO` module includes
    two dedicated :class:`~synthax.module.ADSR`: one for rate
    modulation and another for amplitude modulation), one
    :class:`~synthax.module.SineVCO`, one
    :class:`~synthax.module.SquareSawVCO`, one
    :class:`~synthax.module.Noise` generator,
    :class:`~synthax.module.VCA`, a
    :class:`~synthax.module.ModulationMixer` and an
    :class:`~synthax.module.AudioMixer`. Modulation signals
    generated from control modules (:class:`~synthax.module.ADSR`
    and :class:`~synthax.module.LFO`) are upsampled to the audio
    sample rate before being passed to audio rate modules.

    Args:
        PRNG_key (jax.random.PRNGKey): PRNG key already split.
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
        load_file (str): Filename for the default hyperparameters
    """

    PRNG_key: jax.random.PRNGKey
    config: SynthConfig
    load_file: str

    # TODO: Define dynamically
    def setup(self):
        # Define modules
        modules_spec = [
            ("keyboard", MonophonicKeyboard, {}),
            ("adsr_1", ADSR, {}),
            ("adsr_2", ADSR, {}),
            ("adsr_3", ADSR, {}),
            ("lfo_1", LFO, {}),
            ("lfo_2", LFO, {}),
            ("lfo_3", LFO, {}),
            ("lfo_1_amp_adsr", ADSR, {}),
            ("lfo_2_amp_adsr", ADSR, {}),
            ("lfo_3_amp_adsr", ADSR, {}),
            ("lfo_1_rate_adsr", ADSR, {}),
            ("lfo_2_rate_adsr", ADSR, {}),
            ("lfo_3_rate_adsr", ADSR, {}),
            ("control_vca", ControlRateVCA, {}),
            ("control_upsample", ControlRateUpsample, {}),
            ("mod_matrix", ModulationMixer, {
                "n_input": 6,
                "n_output": 7,
                "input_names": [
                    "adsr_1",
                    "adsr_2",
                    "adsr_3",
                    "lfo_1",
                    "lfo_2",
                    "lfo_3"
                ],
                "output_names": [
                    "vco_1_pitch",
                    "vco_1_amp",
                    "vco_2_pitch",
                    "vco_2_amp",
                    "vco_3_pitch",
                    "vco_3_amp",
                    "noise_amp",
                ]
            }
            ),
            ("vco_1", SineVCO, {}),
            ("vco_2", SquareSawVCO, {}),
            ("vco_3", SineVCO, {}),
            ("vco_4", FmVCO, {}),
            ("noise", Noise, {}),
            ("vca", VCA, {}),
            ("mixer", AudioMixer, {
                "n_input": 4,
                "curves": [1.0, 1.0, 1.0, 0.025],
                "names": ["vco_1", "vco_2", "vco_4", "noise"]
            }
            )
        ]

        key = self.PRNG_key
        modules = {}
        for name, module, params in modules_spec:
            key, subkey = jax.random.split(key)
            modules[name] = module(subkey, self.config, **params)

        # TODO: Remove
        # Initialize all submodules
        # for module in modules.values():
        #     key, subkey = jax.random.split(key)
        #     module.init(subkey)

        self.modules = modules

        # Load the parameters
        # self.load_hyperparameters(self.load_file)

    def __call__(self, *args, **kwargs) -> Any:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        midi_f0, note_on_duration = self.modules["keyboard"]()

        # ADSRs for modulating LFOs
        lfo_1_rate = self.modules["lfo_1_rate_adsr"](note_on_duration)
        lfo_2_rate = self.modules["lfo_2_rate_adsr"](note_on_duration)
        lfo_3_rate = self.modules["lfo_3_rate_adsr"](note_on_duration)
        lfo_1_amp = self.modules["lfo_1_amp_adsr"](note_on_duration)
        lfo_2_amp = self.modules["lfo_2_amp_adsr"](note_on_duration)
        lfo_3_amp = self.modules["lfo_3_amp_adsr"](note_on_duration)

        # Compute LFOs with envelopes
        lfo_1 = self.modules["control_vca"](self.modules["lfo_1"](lfo_1_rate), lfo_1_amp)
        lfo_2 = self.modules["control_vca"](self.modules["lfo_2"](lfo_2_rate), lfo_2_amp)
        lfo_3 = self.modules["control_vca"](self.modules["lfo_3"](lfo_3_rate), lfo_3_amp)

        # ADSRs for Oscillators and noise
        adsr_1 = self.modules["adsr_1"](note_on_duration)
        adsr_2 = self.modules["adsr_2"](note_on_duration)
        adsr_3 = self.modules["adsr_3"](note_on_duration)

        # Mix all modulation signals
        (vco_1_pitch,
         vco_1_amp,
         vco_2_pitch,
         vco_2_amp,
         vco_3_pitch,
         vco_3_amp,
         noise_amp) = self.modules["mod_matrix"](adsr_1, adsr_2, adsr_3,
                                                 lfo_1, lfo_2, lfo_3)

        # Create signal and with modulations and mix together
        vco_1_out = self.modules["vca"](
            self.modules["vco_1"](midi_f0, self.modules["control_upsample"](vco_1_pitch)),
            self.modules["control_upsample"](vco_1_amp),
        )
        vco_2_out = self.modules["vca"](
            self.modules["vco_2"](midi_f0, self.modules["control_upsample"](vco_2_pitch)),
            self.modules["control_upsample"](vco_2_amp),
        )
        vco_3_out = self.modules["vca"](
            self.modules["vco_3"](midi_f0, self.modules["control_upsample"](vco_3_pitch)),
            self.modules["control_upsample"](vco_3_amp),
        )
        vco_4_out = self.modules["vco_4"](midi_f0, vco_3_out)
        noise_out = self.modules["vca"](
            self.modules["noise"](),
            self.modules["control_upsample"](noise_amp)
        )

        return self.modules["mixer"](vco_1_out, vco_2_out, vco_4_out, noise_out)
