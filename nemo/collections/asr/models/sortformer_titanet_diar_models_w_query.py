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

import itertools
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import inf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.data.audio_to_diar_label_w_query_lhotse import LhotseAudioToSpeechE2ESpkDiarWQueryDataset
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, SpkDiarizationMixin
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_ats_targets, get_pil_targets
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging
from nemo.collections.asr.parts.submodules.tdnn_attention import (
    TDNNModule,
    get_statistics_with_mask,
    lens_to_mask,
)


__all__ = ['SortformerEncLabelModel']


class AttentivePoolLayer(nn.Module):

    def __init__(
        self,
        inp_filters: int,
        attention_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        eps: float = 1e-10,
        stride: int = 1,
    ):
        super().__init__()

        self.feat_in = inp_filters

        self.attention_layer = nn.Sequential(
            TDNNModule(inp_filters * 3, attention_channels, kernel_size=kernel_size, dilation=dilation),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels,
                out_channels=inp_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
            ),
        )
        self.eps = eps

    def forward(self, x, length=None):
        max_len = x.size(2)
        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = lens_to_mask(length, max_len=max_len, device=x.device)

        # encoder statistics
        mean, std = get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        # attention statistics
        attn = self.attention_layer(attn)  # attention pass
        # attn = attn.masked_fill(mask == 0, -inf)
        return attn


class SortformerTitanetEncLabelWQueryModel(SortformerEncLabelModel):
    """
    Encoder class for Sortformer diarization model.
    Model class creates training, validation methods for setting up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * Transformer Encoder
        * FastConformer Encoder
        * Sortformer Modules
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an Sortformer Diarizer model and a pretrained NEST encoder.
        In this init function, training and validation datasets are prepared.
        """
        random.seed(42)
        self._trainer = trainer if trainer else None
        self._cfg = cfg

        # First initialize the base model
        super().__init__(cfg=self._cfg, trainer=trainer)
        self.__init_titanet_layers()

    def __init_titanet_layers(self):
                # Then load the TitaNet encoder
        titanet = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
        self.titanet_encoder = titanet.encoder
        self.titanet_encoder.freeze()
        del titanet

        # Register the TitaNet encoder as a submodule
        self.add_module('titanet_encoder', self.titanet_encoder)

        self.titanet_pooling = AttentivePoolLayer(
            inp_filters =  self.titanet_encoder._cfg['jasper'][-1]['filters'], # 3072
            kernel_size = 8,
            stride=8,
            attention_channels = 128)
        
        shapes = [self.titanet_pooling.feat_in, int(self._cfg.model_defaults.tf_d_model)] # [3072, 192]
        emb_layers = []
        for shape_in, shape_out in zip(shapes[:-1], shapes[1:]):
            layer = self.affine_layer(shape_in, shape_out, learn_mean=False, affine_type='conv')
            emb_layers.append(layer)

        self.emb_layers = torch.nn.ModuleList(emb_layers)

    def affine_layer(
        self,
        inp_shape,
        out_shape,
        learn_mean=True,
        affine_type='conv',
    ):
        if affine_type == 'conv':
            layer = nn.Sequential(
                nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True),
                nn.Conv1d(inp_shape, out_shape, kernel_size=1),
            )

        else:
            layer = nn.Sequential(
                nn.Linear(inp_shape, out_shape),
                nn.BatchNorm1d(out_shape, affine=learn_mean, track_running_stats=True),
                nn.ReLU(),
            )

        return layer

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle TitaNet encoder weights separately.
        """
        # Filter out TitaNet encoder weights from the state dict
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('titanet_encoder.')}
        
        # Load the filtered state dict
        return super().load_state_dict(filtered_state_dict, strict=False)

    def forward(self, audio_signal, audio_signal_length):
        """
        Forward pass for training and inference with Titanet encoder.

        Args:
            audio_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, num_speakers)
        """
        processed_signal, processed_signal_length = self.process_signal(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        # TitaNet encoder forward
        encoder_outputs = self.titanet_encoder(audio_signal=processed_signal, length=processed_signal_length)
        if isinstance(encoder_outputs, tuple):
            encoded, length = encoder_outputs
        else:
            encoded, length = encoder_outputs, None

        pool = self.titanet_pooling(encoded, length)
        for layer in self.emb_layers:
            pool = layer(pool)


        # Sortformer encoder forward
        emb_seq, emb_seq_length = self.frontend_encoder(processed_signal, processed_signal_length)

        # fix pool shape to match with emb_seq
        pool = pool.transpose(1, 2)
        if pool.shape[1] < emb_seq.shape[1]:
            last_emb = pool[:,-1,:].unsqueeze(1)
            additional_frames = emb_seq.shape[1] - pool.shape[1]
            last_repeats = last_emb.repeat(1, additional_frames, 1)
            extended_pool = torch.cat([pool, last_repeats], dim=1)
            pool = extended_pool
        elif pool.shape[1] > emb_seq.shape[1]:
            pool = pool[:, :emb_seq.shape[1], :]
        
        # Combine the features (you can choose how to combine them)
        # import ipdb; ipdb.set_trace()
        combined_features = emb_seq + pool  # Simple addition, you can modify this

        # Continue with the rest of the forward pass
        preds = self.forward_infer(combined_features, emb_seq_length)
        return preds

    def __setup_dataloader_from_config(self, config):
        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseAudioToSpeechE2ESpkDiarWQueryDataset(cfg=config),
            )
    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(
            config=train_data_config,
        )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(
            config=val_data_layer_config,
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(
            config=test_data_config,
        )

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl
        return None