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

import jax.numpy as jnp
import chex
from flax import linen as nn
from synthax.config import SynthConfig
from synthax.parameter import ModuleParameterRange

from synthax.modules.oscillators import SineVCO, FmVCO, SquareSawVCO, Noise
from synthax.modules.envelopes import ADSR
from synthax.modules.lfos import LFO
from synthax.modules.amplifiers import VCA, ControlRateVCA
from synthax.modules.mixers import AudioMixer, ModulationMixer
from synthax.modules.control import ControlRateUpsample
from synthax.modules.keyboard import MonophonicKeyboard


class BaseSynth(nn.Module):
    """
    Base class for synthesizers that combine one or more
    :class:`~synthax.module.SynthModule` to create a full synth
    architecture.

    Args:
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
    """

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


class VoiceExpanded(BaseSynth):
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
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
    """

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
                "n_output": 7
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
            })
        ]

        modules = {}
        for name, module, params in modules_spec:
            modules[name] = module(config=self.config, **params)

        self.modules = modules

    def __call__(self, *args, **kwargs) -> chex.Array:
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
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
    """

    def setup(self):
        # Define modules
        modules_spec = [
            ("keyboard", MonophonicKeyboard, {}),
            ("adsr_1", ADSR, {}),
            ("adsr_2", ADSR, {}),
            ("lfo_1", LFO, {}),
            ("lfo_2", LFO, {}),
            ("lfo_1_amp_adsr", ADSR, {}),
            ("lfo_2_amp_adsr", ADSR, {}),
            ("lfo_1_rate_adsr", ADSR, {}),
            ("lfo_2_rate_adsr", ADSR, {}),
            ("control_vca", ControlRateVCA, {}),
            ("control_upsample", ControlRateUpsample, {}),
            ("mod_matrix", ModulationMixer, {
                "n_input": 4,
                "n_output": 5
            }
            ),
            ("vco_1", SineVCO, {}),
            ("vco_2", SquareSawVCO, {}),
            ("noise", Noise, {}),
            ("vca", VCA, {}),
            ("mixer", AudioMixer, {
                "n_input": 3,
                "level": ModuleParameterRange(
                    minimum=0,
                    maximum=1,
                    curve=jnp.array([1, 1, 0.025]) # 0.025 is for noise
                )
            })
        ]

        modules = {}
        for name, module, params in modules_spec:
            modules[name] = module(config=self.config, **params)

        self.modules = modules

    def __call__(self, *args, **kwargs) -> chex.Array:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        midi_f0, note_on_duration = self.modules["keyboard"]()

        # ADSRs for modulating LFOs
        lfo_1_rate = self.modules["lfo_1_rate_adsr"](note_on_duration)
        lfo_2_rate = self.modules["lfo_2_rate_adsr"](note_on_duration)
        lfo_1_amp = self.modules["lfo_1_amp_adsr"](note_on_duration)
        lfo_2_amp = self.modules["lfo_2_amp_adsr"](note_on_duration)

        # Compute LFOs with envelopes
        lfo_1 = self.modules["control_vca"](self.modules["lfo_1"](lfo_1_rate), lfo_1_amp)
        lfo_2 = self.modules["control_vca"](self.modules["lfo_2"](lfo_2_rate), lfo_2_amp)

        # ADSRs for Oscillators and noise
        adsr_1 = self.modules["adsr_1"](note_on_duration)
        adsr_2 = self.modules["adsr_2"](note_on_duration)

        # Mix all modulation signals
        (vco_1_pitch,
         vco_1_amp,
         vco_2_pitch,
         vco_2_amp,
         noise_amp) = self.modules["mod_matrix"](adsr_1, adsr_2, lfo_1, lfo_2,)

        # Create signal and with modulations and mix together
        vco_1_out = self.modules["vca"](
            self.modules["vco_1"](midi_f0, self.modules["control_upsample"](vco_1_pitch)),
            self.modules["control_upsample"](vco_1_amp),
        )
        vco_2_out = self.modules["vca"](
            self.modules["vco_2"](midi_f0, self.modules["control_upsample"](vco_2_pitch)),
            self.modules["control_upsample"](vco_2_amp),
        )
        noise_out = self.modules["vca"](
            self.modules["noise"](),
            self.modules["control_upsample"](noise_amp)
        )

        return self.modules["mixer"](vco_1_out, vco_2_out, noise_out)

class ParametricSynth(BaseSynth):
    """
    Define a synthesizer given the number of VCOs by type.

    Args:
        config (:class:`~synthax.config.SynthConfig`): Global configuration.
        sine (int): Number of :class:`~synthax.modules.oscillators.SineVCO`
        square_saw (int): Number of :class:`~synthax.modules.oscillators.SquareSawVCO`
        fm_sine (int): Number of :class:`~synthax.modules.oscillators.FmVCO` and
            :class:`~synthax.modules.oscillators.SineVCO`
        fm_square_saw (int): Number of :class:`~synthax.modules.oscillators.FmVCO` and
            :class:`~synthax.modules.oscillators.SquareSawVCO`
    """

    sine: int = 1
    square_saw: int = 1
    fm_sine: int = 1
    fm_square_saw: int = 0

    def setup(self):
        # Store total number of VCOs
        self.n = self.sine + self.square_saw + self.fm_sine + self.fm_square_saw

        # Define modules
        modules_spec = [
            ("control_vca", ControlRateVCA, {}),
            ("control_upsample", ControlRateUpsample, {}),
            ("noise", Noise, {}),
            ("vca", VCA, {}),
        ]

        mod_input = []
        mod_output = []
        for i in range(1, self.n + 1):
            # Modules (keyboards, ADSRs and LFOs)
            modules_spec.append((f"keyboard_{i}", MonophonicKeyboard, {}))
            modules_spec.append((f"adsr_{i}", ADSR, {}))
            modules_spec.append((f"lfo_{i}", LFO, {}))
            modules_spec.append((f"lfo_{i}_amp_adsr", ADSR, {}))
            modules_spec.append((f"lfo_{i}_rate_adsr", ADSR, {}))
            # Params for ModulationMixer
            mod_input.append(f"adsr_{i}")
            mod_input.append(f"lfo_{i}")
            mod_output.append(f"vco_{i}_pitch")
            mod_output.append(f"vco_{i}_amp")

        mixable_vcos = []
        i = 1
        # SineVCO
        for _ in range(self.sine):
            modules_spec.append((f"vco_{i}", SineVCO, {}))
            mixable_vcos.append(f"vco_{i}")
            i += 1

        # SquareSawVCO
        for _ in range(self.square_saw):
            modules_spec.append((f"vco_{i}", SquareSawVCO, {}))
            mixable_vcos.append(f"vco_{i}")
            i += 1

        # FM SineVCO
        fm_vcos_idxs = []
        for _ in range(self.fm_sine):
            modules_spec.append((f"vco_{i}", SineVCO, {}))
            fm_vcos_idxs.append(i)
            i += 1

        # FM SquareSawVCO
        for _ in range(self.fm_square_saw):
            modules_spec.append((f"vco_{i}", SquareSawVCO, {}))
            fm_vcos_idxs.append(i)
            i += 1

        fm_idxs = []
        for _ in range(self.fm_sine + self.fm_square_saw):
            modules_spec.append((f"vco_{i}", FmVCO, {}))
            mixable_vcos.append(f"vco_{i}")
            fm_idxs.append(i)
            i += 1

        self.fm_pair_idxs = list(zip(fm_vcos_idxs, fm_idxs))

        # Modulation mixer
        modules_spec.append(
            (
                "mod_matrix",
                ModulationMixer,
                {
                    "n_input": len(mod_input),
                    "n_output": len(mod_output) + 1
                },
             )
        )

        # Audio mixer
        modules_spec.append(
            (
                "mixer",
                AudioMixer,
                {
                    "n_input": len(mixable_vcos) + 1, # VCOs + noise
                },
            )
        )

        modules = {}
        for name, module, params in modules_spec:
            modules[name] = module(config=self.config, **params)

        self.modules = modules

    def __call__(self, *args, **kwargs) -> chex.Array:
        # Keyboard for each VCO
        keyboards = [] # [(midi_f0, note_on_duration)]
        for i in range(1, self.n + 1):
            keyboards.append(self.modules[f"keyboard_{i}"]())

        # ADSRs for modulating LFOs
        lfos_rate = []
        lfos_amp = []
        for i in range(1, self.n + 1):
            lfos_rate.append(self.modules[f"lfo_{i}_rate_adsr"](keyboards[i-1][1]))
            lfos_amp.append(self.modules[f"lfo_{i}_amp_adsr"](keyboards[i-1][1]))

        # Compute LFOs with envelopes
        lfos = []
        for i in range(1, self.n + 1):
            lfos.append(
                self.modules["control_vca"](
                    self.modules[f"lfo_{i}"](lfos_rate[i-1]),
                    lfos_amp[i-1]
                )
            )

        # ADSRs for oscillators and noise
        adsrs = []
        for i in range(1, self.n + 1):
            adsrs.append(self.modules[f"adsr_{i}"](keyboards[i-1][1]))

        # Mix all modulation signals
        mods = self.modules["mod_matrix"](*adsrs, *lfos)

        # Create signal and with modulations and mix together
        vcos = []
        for i, (vco_pitch, vco_amp) in enumerate(zip(mods[::2], mods[1::2])):
            # Iterate pairwise (e.g. [1,2,3,4] -> [(1,2), (3,4)])
            vcos.append(
                self.modules["vca"](
                    self.modules[f"vco_{i+1}"](
                        keyboards[i][0],
                        self.modules["control_upsample"](vco_pitch)
                    ),
                    self.modules["control_upsample"](vco_amp)
                )
            )

        # Input VCOs into FM VCOs
        for fm_vco_idx, fm_idx in self.fm_pair_idxs:
            vcos.append(
                self.modules[f"vco_{fm_idx}"](
                    keyboards[fm_vco_idx - 1][0],
                    vcos[fm_vco_idx - 1]
                )
            )

        # Add noise
        noise_out = self.modules["vca"](
            self.modules["noise"](),
            self.modules["control_upsample"](mods[-1])
        )

        # Get only the mixable VCOs (i.e. no VCOs going into FM)
        mixable_vcos = [vcos[i - 1]
                        for i in range(1, len(vcos) + 1)
                        if i not in [t[0] for t in self.fm_pair_idxs]]

        return self.modules["mixer"](*mixable_vcos, noise_out)
