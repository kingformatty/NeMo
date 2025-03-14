# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Tuple
import numpy as np
import torch.utils.data
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples, uuid4
from lhotse import SupervisionSet, SupervisionSegment, MonoCut, Recording, CutSet

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_hidden_length_from_sample_length,
)

from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    get_separator_audio,
    get_query_cut,
    speaker_to_target_w_query,
    mix_noise
)


from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset


class LhotseAudioToSpeechE2ESpkDiarWQueryDataset(torch.utils.data.Dataset):
    """
    This dataset is a Lhotse version of diarization dataset in audio_to_diar_label.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define the output types of the dataset."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'targets': NeuralType(('B', 'T', 'N'), LabelsType()),
            'target_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg):
        super().__init__()
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero',False)
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.add_separater_audio = self.cfg.get('add_separater_audio', True)
        self.separater_freq = self.cfg.get('separater_freq', 500)
        self.separater_duration = self.cfg.get('separater_duration',1)
        self.separater_unvoice_ratio = self.cfg.get('separater_unvoice_ratio', 0.3)
        if self.add_separater_audio:
            self.separater_audio = get_separator_audio(self.separater_freq, self.cfg.sample_rate, self.separater_duration, self.separater_unvoice_ratio)
        self.query_noise_path = self.cfg.get('query_noise_path',None)
        if self.query_noise_path:
            self.query_noise_cut = guess_parse_cutset(self.query_noise_path)
            self.query_noise_mix_prob = self.cfg.get('query_noise_mix_prob', 0.3)
            self.query_snr = tuple(self.cfg.get('query_snr',(2.5, 12.5)))

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        query_cuts = CutSet.from_cuts(get_query_cut(c) for c in cuts)
        if self.query_noise_path:
            query_cuts = mix_noise(
                query_cuts,
                self.query_noise_cut,
                snr = self.query_snr,
                mix_prob = self.query_noise_mix_prob,
            )
        spk_targets = [torch.transpose(torch.as_tensor(speaker_to_target_w_query(
            c, q, 
            self.add_separater_audio,
            self.separater_duration,
            self.num_speakers, 
            self.num_sample_per_mel_frame, 
            self.num_mel_frame_per_asr_frame, 
            self.spk_tar_all_zero), 
            dtype=torch.float32), 0, 1) for c, q in zip(cuts,query_cuts)]
        audio, audio_lens, cuts = self.load_audio(cuts)
        query_audio, query_audio_lens, query_cuts = self.load_audio(query_cuts)
        if self.add_separater_audio:
            concat_list = []
            for i in range(len(audio)):
                concat_list.append(torch.cat([query_audio[i,:query_audio_lens[i]],torch.tensor(self.separater_audio).to(audio.dtype),audio[i,:audio_lens[i]]]))
            audio = collate_vectors(concat_list, padding_value = 0)
            audio_lens = audio_lens + query_audio_lens + self.separater_duration * self.cfg.sample_rate
        else:
            concat_list = []
            for i in range(len(audio)):
                concat_list.append(torch.cat([query_audio[i,:query_audio_lens[i]],audio[i,:audio_lens[i]]]))
            audio = collate_vectors(concat_list, padding_value = 0)
            audio_lens = audio_lens + query_audio_lens
        spk_targets = collate_matrices(spk_targets)
        target_lens_list = []
        for audio_len in audio_lens:
            target_fr_len = get_hidden_length_from_sample_length(
                audio_len, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame
            )
            target_lens_list.append(target_fr_len)
        import ipdb; ipdb.set_trace()
        target_lens = torch.tensor(target_lens_list)
        return audio, audio_lens, spk_targets, target_lens