# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


import copy
import os
from typing import Optional
import soundfile as sf
import librosa

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch
from nemo.collections.asr.parts.utils.streaming_utils import BatchedFeatureFrameBufferer
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.preprocessing.segment import get_samples, AudioSegment
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import LengthsType, MelSpectrogramType, NeuralType
from nemo.collections.asr.parts.utils.streaming_utils import *
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length

from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    get_separator_audio,
)

import torch.nn.functional as F


# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames

# audio buffer
class FrameBatchDiarizer_tgt_spk:
    """
    class for streaming frame-based ASR use reset() method to reset FrameASR's
    state call transcribe(frame) to do ASR on contiguous signal's frames
    """

    def __init__(
        self,
        diar_model,
        frame_len=1.6,
        total_buffer=4.0,
        batch_size=4,
        dynamic_query=False,
        pad_to_buffer_len=True,
        activation_ratio=0.85,
        new_query_max_len=10,
        new_query_min_len=13, #4 for strategy 2, 13 for strategy 3 buffer 4, 38 for strategy 3 buffer 8
        query_refresh_rate=1, # 1 for strategy 3 high throughput, 10 for strategy 3 low latency
        query_change_once=True,#change query once and then not update query
        non_target_spk_offset_threshold=0.3, #0.85
        target_spk_onset_threshold=0.3, #0.3
        start_replace_step=3, #wait for x steps before start replacing query, 3 for high throughput, 7 for low latency, overall 3s 
        diar_model_streaming_mode=False,

    ):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.frame_bufferer = AudioBufferer_tgt_spk(
            asr_model=diar_model,
            frame_len=frame_len,
            batch_size=batch_size,
            total_buffer=total_buffer,
            pad_to_buffer_len=pad_to_buffer_len,
        )

        self.asr_model = diar_model

        self.batch_size = batch_size
        self.all_logits = []
        self.all_preds = []
        self.dynamic_query = dynamic_query

        self.frame_buffers = []
        self.diar_model_streaming_mode = diar_model_streaming_mode
        self.reset()
        cfg = copy.deepcopy(diar_model._cfg)
        self.cfg = cfg
        self.frame_len = frame_len
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        # import ipdb; ipdb.set_trace()
        self.raw_preprocessor = ASRModel.from_config_dict(cfg.preprocessor)
        self.raw_preprocessor.to(diar_model.device)
        self.preprocessor = self.raw_preprocessor
        #dynamic query related variables
        self.all_diar_preds = None
        self.all_audio = None
        self.new_query_min_len = new_query_min_len
        self.activation_ratio = activation_ratio
        self.new_query_max_len = new_query_max_len
        self.target_spk_onset_threshold = target_spk_onset_threshold
        self.non_target_spk_offset_threshold = non_target_spk_offset_threshold
        self.query_refresh_rate = query_refresh_rate
        self.query_change_once = query_change_once
        self.start_replace_step = start_replace_step


    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        self.prev_char = ''
        self.unmerged = []
        self.data_layer = AudioBuffersDatalayer_tgt_spk()
        self.data_loader = DataLoader(self.data_layer, batch_size=self.batch_size, collate_fn=speech_collate_fn)
        self.all_logits = []
        self.all_preds = []
        self.frame_buffers = []
        self.frame_bufferer.reset()
        self.query_refresh_count = 0
        self.query_refresh_rate = 1
        if self.diar_model_streaming_mode:
            assert self.batch_size == 1, "Batch size must be 1 for streaming mode"
            self._reset_streaming_state(
                batch_size = self.batch_size,
                async_streaming = False,
                device = self.asr_model.device
            )
        else:
            self.asr_model.streaming_mode = False

    def _reset_streaming_state(self, batch_size, async_streaming, device):
        self.asr_model.streaming_mode = True
        self.asr_model.sortformer_modules.chunk_len = 376
        self.asr_model.sortformer_modules.fifo_len = 188
        self.asr_model.sortformer_modules.spkcache_len = 188
        self.asr_model.sortformer_modules.query_embs = None
        self.asr_model.sortformer_modules.query_len = None
        self.query_len = None
        self.query_pred = None
        self.streaming_state = self.asr_model.sortformer_modules.init_streaming_state(
            batch_size = batch_size,
            async_streaming = async_streaming,
            device = device
        )
        self.streaming_state.query_pred_len = torch.zeros((batch_size), device=self.asr_model.device, dtype = torch.int32)

        # self.asr_model._reset_streaming_state()

    def get_partial_samples(self, audio_file: str, offset: float, duration: float, target_sr: int = 16000, dtype: str = 'float32'):
        try:
            with sf.SoundFile(audio_file, 'r') as f:
                start = int(offset * f.samplerate)
                f.seek(start)
                end = int((offset + duration) * f.samplerate)
                samples = f.read(dtype=dtype, frames = end - start)
                if f.samplerate != target_sr:
                    samples = librosa.core.resample(samples, orig_sr=f.samplerate, target_sr=target_sr)
                samples = samples.transpose()
        except:
            raise ValueError('Frame exceed audio')
        return samples

    def read_audio_file(self, audio_filepath: str, offset, duration, query_audio_file, query_offset, query_duration, separater_freq, separater_duration, separater_unvoice_ratio,delay, model_stride_in_secs, tokens_per_chunk):
        # samples = get_samples(audio_filepath)
        # rewrite loading audio function to support partial audio
        samples = self.get_partial_samples(audio_filepath, offset, duration)
        # pad on the right side
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        # query related variables
        separater_audio = get_separator_audio(separater_freq, self.asr_model._cfg.sample_rate, separater_duration, separater_unvoice_ratio)
        self.separater_audio = separater_audio
        if query_duration > 0:
            query_samples = self.get_partial_samples(query_audio_file, query_offset, query_duration)
            query_samples = np.concatenate([query_samples, separater_audio])
        else:
            query_samples = separater_audio
        # import ipdb; ipdb.set_trace()
        frame_reader = AudioIterator_tgt_spk(samples, query_samples, self.frame_len, self.asr_model.device)
        self.query_pred_len = get_hidden_length_from_sample_length(len(query_samples), 160, 8)
        self.set_frame_reader(frame_reader)
        #reset all_audio all_diar_preds
        self.all_audio = None
        self.all_diar_preds = None
        self.selected_regions = set()
        self.change_query_action = []
        self.delay = delay
        self.tokens_per_chunk = tokens_per_chunk
        if self.diar_model_streaming_mode:
            self.asr_model.sortformer_modules.spkcache_len = self.query_pred_len

    def set_frame_reader(self, frame_reader):
        self.frame_bufferer.set_frame_reader(frame_reader)

    @torch.no_grad()
    def infer_logits(self, keep_logits=False):
        frame_buffers = self.frame_bufferer.get_buffers_batch()

        while len(frame_buffers) > 0:
            self.frame_buffers += frame_buffers[:]
            self.data_layer.set_signal(frame_buffers[:])
            self._get_batch_preds(keep_logits)
            frame_buffers = self.frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self, keep_logits=False):
        device = self.asr_model.device
        for batch in iter(self.data_loader):
            
            feat_signal, feat_signal_len = batch

            # # padding silence after each buffer
            # import numpy as np; import torch
            # # Add padding silence to each sample
            # batch_size = len(feat_signal)
            # padding_len = 1 # in s
            # batched_padding_silence =torch.tensor(np.array([np.zeros([16000*padding_len]) for _ in range(batch_size)]))
            # feat_signal = torch.cat([feat_signal, batched_padding_silence], axis=-1)
            # padding_len = padding_len*16000
            # feat_signal_len += padding_len



            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            # forward_outs = self.asr_model(processed_signal=feat_signal, processed_signal_length=feat_signal_len)
            # encoded, encoded_len, _, _ = self.asr_model.train_val_forward([feat_signal, feat_signal_len, None, None, None, None], 0)
            if self.all_audio is None:
                self.all_audio = feat_signal[:,int(self.frame_bufferer.frame_reader.query_audio_signal_len[0]):]
            else:
                self.all_audio = F.pad(self.all_audio, (0, self.frame_bufferer.feature_frame_len, 0, 0))
                self.all_audio[:, -self.frame_bufferer.feature_buffer_len:] = feat_signal[:,int(self.frame_bufferer.frame_reader.query_audio_signal_len[0]):]
            if not self.diar_model_streaming_mode:
                preds = self.asr_model.forward(audio_signal = feat_signal, audio_signal_length = feat_signal_len)
                self.asr_model.diar_preds = preds

                if self.all_diar_preds is None:
                    self.all_diar_preds = self.asr_model.diar_preds[:,self.query_pred_len-1: self.asr_model.diar_preds.shape[1] - 1 - self.delay + self.tokens_per_chunk]
                    self.query_pred = self.asr_model.diar_preds[:,:self.query_pred_len]
                else:
                    import numpy as np
                    # self.all_diar_preds = F.pad(self.all_diar_preds, (0, 0, 0, int(np.ceil(self.frame_bufferer.feature_frame_len/16000 * 12.5)), 0, 0))
                    # self.all_diar_preds[:, -get_hidden_length_from_sample_length(self.frame_bufferer.feature_buffer_len, 160, 8)+2:,:] = self.asr_model.diar_preds[:, -get_hidden_length_from_sample_length(self.frame_bufferer.feature_buffer_len, 160, 8)+2:,:]
                    self.all_diar_preds = torch.cat([self.all_diar_preds, self.asr_model.diar_preds[:, self.asr_model.diar_preds.shape[1] - 1 - self.delay : self.asr_model.diar_preds.shape[1] - 1 - self.delay + self.tokens_per_chunk]], dim=1)
                    

            else:
                import math
                left_context = 0
                if self.query_pred is None:
                    diar_input_signal_len = feat_signal_len
                    diar_input_signal = feat_signal
                else:
                    chunk_len = self.frame_bufferer.feature_frame_len
                    chunk_len += left_context
                    diar_input_signal = torch.empty((feat_signal.size(0), chunk_len), 
                    dtype=feat_signal.dtype,
                    device=feat_signal.device)
                    for i in range(feat_signal.size(0)):
                        diar_input_signal[i,:] = feat_signal[i,feat_signal_len[i] - int(chunk_len):feat_signal_len[i]]
                    diar_input_signal_len = torch.tensor([int(chunk_len)], device = feat_signal.device).expand(feat_signal.size(0))
                with torch.no_grad():
                    # diar_preds = self.forward_diar(signal, signal_len, is_raw_waveform_input)
                    processed_signal, processed_signal_len = self.asr_model.process_signal(
                        audio_signal = diar_input_signal,
                        audio_signal_length = diar_input_signal_len,
                    )
                    feat_len = processed_signal.shape[2]
                    num_chunks = math.ceil(
                        feat_len / (self.asr_model.sortformer_modules.chunk_len * self.asr_model.sortformer_modules.subsampling_factor)
                    )
                    assert num_chunks == 1, "Only one chunk should be used for streaming mode"
                    streaming_loader = self.asr_model.sortformer_modules.streaming_feat_loader(
                        feat_seq = processed_signal,
                        feat_seq_length = processed_signal_len,
                        feat_seq_offset = 0
                    )
                    for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
                        # import ipdb; ipdb.set_trace()
                        self.streaming_state, chunk_preds = self.asr_model.forward_streaming_step(
                        processed_signal=chunk_feat_seq_t,
                        processed_signal_length=feat_lengths,
                        streaming_state=self.streaming_state,
                        total_preds=None,
                        left_offset=left_offset,
                        right_offset=right_offset,
                        streaming_level = 'emb',
                        left_context = left_context,
                        tokens_per_chunk = self.tokens_per_chunk,
                        query_len = self.query_pred_len
                        )
                if self.all_diar_preds is None:
                    self.all_diar_preds = self.asr_model.spkcache_fifo_chunk_preds[:,self.query_pred_len-1: self.asr_model.spkcache_fifo_chunk_preds.shape[1] - 1 - self.delay + self.tokens_per_chunk]
                    self.query_pred = self.asr_model.spkcache_fifo_chunk_preds[:,:self.query_pred_len-1]
                else:
                    import numpy as np
                    # self.all_diar_preds = F.pad(self.all_diar_preds, (0, 0, 0, int(np.ceil(self.frame_bufferer.feature_frame_len/16000 * 12.5)), 0, 0))
                    # self.all_diar_preds[:, -get_hidden_length_from_sample_length(self.frame_bufferer.feature_buffer_len, 160, 8)+2:,:] = self.asr_model.spkcache_fifo_chunk_preds[:, -get_hidden_length_from_sample_length(self.frame_bufferer.feature_buffer_len, 160, 8)+2:,:]
                    self.all_diar_preds = torch.cat([self.all_diar_preds, self.asr_model.spkcache_fifo_chunk_preds[:, self.asr_model.spkcache_fifo_chunk_preds.shape[1] - 1 - self.delay : self.asr_model.spkcache_fifo_chunk_preds.shape[1] - 1 - self.delay + self.tokens_per_chunk]], dim=1)

                # mid_chunk_preds = self.all_diar_preds[0,self.all_diar_preds.shape[1] - 1 - self.delay : self.all_diar_preds.shape[1] - 1 - self.delay + self.tokens_per_chunk].clone()
                # # import ipdb; ipdb.set_trace()
                # # for i in range(mid_chunk_preds.shape[0]):
                # #     if sum(mid_chunk_preds[i]) < 1.2 and mid_chunk_preds[i,0] <=0.7:
                # #         mid_chunk_preds[i,0] = 0
                # self.all_diar_preds[:,self.all_diar_preds.shape[1] - 1 - self.delay : self.all_diar_preds.shape[1] - 1 - self.delay + self.tokens_per_chunk] = mid_chunk_preds
                # # take care of confusion
                # # for i in range(len(mid_chunk_preds[1])):
                # #     if 

            save_intermediate_var = False
            if save_intermediate_var:
                parent_dir = '/home/jinhanw/workdir/workdir_nemo_diarization/sortformer_infer/saved/temp'
                os.makedirs(parent_dir, exist_ok=True)
                import pickle; import numpy as np;
                with open(os.path.join(parent_dir, 'feat_signal.pickle'), 'wb') as f:
                    pickle.dump(feat_signal, f)
                with open(os.path.join(parent_dir, 'feat_signal_len.pickle'), 'wb') as f:
                    pickle.dump(feat_signal_len, f)
                # with open(os.path.join(parent_dir,'asr_model.cfg'), 'w') as f:
                    # f.write(OmegaConf.to_yaml(self.asr_model.diarization_model._cfg))
                with open(os.path.join(parent_dir, 'diar_preds.pickle'), 'wb') as f:
                    pickle.dump(self.asr_model.diar_preds, f)
                # with open(os.path.join(parent_dir, 'total_diar_preds.pickle'), 'wb') as f:
                    # pickle.dump(self.asr_model.total_preds, f)
                # if self.dynamic_query:
                with open(os.path.join(parent_dir, 'all_diar_preds.pickle'), 'wb') as f:
                    pickle.dump(self.all_diar_preds, f)
                with open(os.path.join(parent_dir, 'all_audio.pickle'), 'wb') as f:
                    pickle.dump(self.all_audio, f)
                if self.diar_model_streaming_mode:
                    with open(os.path.join(parent_dir, 'spkcache_fifo_chunk_preds.pickle'), 'wb') as f:
                        pickle.dump(self.asr_model.spkcache_fifo_chunk_preds, f)
                import ipdb; ipdb.set_trace()


class AudioIterator_tgt_spk(IterableDataset):
    def __init__(self, samples, query_samples, frame_len, device, pad_to_frame_len=True):
        self._samples = samples
        self._frame_len = frame_len
        self._start = 0
        self.output = True
        self.count = 0
        self.pad_to_frame_len = pad_to_frame_len
        self._feature_frame_len = frame_len * 16000
        self.audio_signal = torch.from_numpy(self._samples).unsqueeze_(0).to(device)
        self.audio_signal_len = torch.Tensor([self._samples.shape[0]]).to(device)
        self._query_samples = query_samples
        self.query_audio_signal = torch.from_numpy(self._query_samples).unsqueeze_(0).to(device)
        self.query_audio_signal_len = torch.Tensor([self._query_samples.shape[0]]).to(device)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        # import ipdb; ipdb.set_trace()
        last = int(self._start + self._feature_frame_len)
        if last <= self.audio_signal_len[0]:
            frame = self.audio_signal[:, self._start : last].cpu()
            self._start = last
        else:
            if not self.pad_to_frame_len:
                frame = self.audio_signal[:, self._start : self.audio_signal_len[0]].cpu()
            else:
                frame = np.zeros([self.audio_signal.shape[0], int(self._feature_frame_len)], dtype='float32')
                segment = self.audio_signal[:, self._start : int(self.audio_signal_len[0])].cpu()
                frame[:, : segment.shape[1]] = segment
            self.output = False
        self.count += 1
        return frame

class AudioBufferer_tgt_spk:
    """
    Class to append each feature frame to a buffer and return
    an array of buffers.
    """

    def __init__(self, asr_model, frame_len=1.6, batch_size=4, total_buffer=4.0, pad_to_buffer_len=True):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        if hasattr(asr_model.preprocessor, 'log') and asr_model.preprocessor.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = 0.0
        self.asr_model = asr_model
        self.sr = asr_model._cfg.sample_rate
        self.frame_len = frame_len
        self.feature_frame_len = int(frame_len * self.sr)
        # timestep_duration = asr_model._cfg.preprocessor.window_stride
        # self.n_frame_len = int(frame_len / timestep_duration)

        # total_buffer_len = int(total_buffer / timestep_duration)
        total_buffer_len = int(total_buffer * self.sr)
        # self.n_feat = asr_model._cfg.preprocessor.features
        
        # self.buffer = np.ones([self.n_feat, total_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.buffer = np.ones([1, total_buffer_len], dtype = np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.pad_to_buffer_len = pad_to_buffer_len
        self.batch_size = batch_size

        self.signal_end = False
        self.frame_reader = None
        self.feature_buffer_len = total_buffer_len

        # self.feature_buffer = (
        #     np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        # )
        self.feature_buffer = (
            np.ones([1, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.frame_buffers = []
        self.buffered_features_size = 0
        self.reset()
        self.buffered_len = 0

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = np.ones(shape=self.buffer.shape, dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.prev_char = ''
        self.unmerged = []
        self.frame_buffers = []
        self.buffered_len = 0
        # self.feature_buffer = (
        #     np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        # )
        self.feature_buffer = (
            np.ones([1, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )

    def get_batch_frames(self):
        if self.signal_end:
            return []
        batch_frames = []
        for frame in self.frame_reader:
            batch_frames.append(np.copy(frame))
            if len(batch_frames) == self.batch_size:
                return batch_frames
        self.signal_end = True

        return batch_frames

    def get_frame_buffers(self, frames):
        # Build buffers for each frame
        self.frame_buffers = []
        for frame in frames:
            curr_frame_len = frame.shape[1]
            self.buffered_len += curr_frame_len
            if curr_frame_len < self.feature_buffer_len and not self.pad_to_buffer_len:
                self.frame_buffers.append(np.copy(frame))
                continue
            self.buffer[:, :-curr_frame_len] = self.buffer[:, curr_frame_len:]
            self.buffer[:, -self.feature_frame_len :] = frame
            self.frame_buffers.append(np.copy(self.buffer))
        return self.frame_buffers

    def set_frame_reader(self, frame_reader):
        self.frame_reader = frame_reader
        self.signal_end = False

    def _update_feature_buffer(self, feat_frame):
        curr_frame_len = feat_frame.shape[1]
        if curr_frame_len < self.feature_buffer_len and not self.pad_to_buffer_len:
            self.feature_buffer = np.copy(feat_frame)  # assume that only the last frame is less than the buffer length
        else:
            self.feature_buffer[:, : -feat_frame.shape[1]] = self.feature_buffer[:, feat_frame.shape[1] :]
            self.feature_buffer[:, -feat_frame.shape[1] :] = feat_frame
        self.buffered_features_size += feat_frame.shape[1]

    def get_norm_consts_per_frame(self, batch_frames):
        norm_consts = []
        for i, frame in enumerate(batch_frames):
            self._update_feature_buffer(frame)
            mean_from_buffer = np.mean(self.feature_buffer, axis=1)
            stdev_from_buffer = np.std(self.feature_buffer, axis=1)
            norm_consts.append((mean_from_buffer.reshape(self.n_feat, 1), stdev_from_buffer.reshape(self.n_feat, 1)))
        return norm_consts

    def normalize_frame_buffers(self, frame_buffers, norm_consts):
        CONSTANT = 1e-5
        for i, frame_buffer in enumerate(frame_buffers):
            frame_buffers[i] = (frame_buffer - norm_consts[i][0]) / (norm_consts[i][1] + CONSTANT)

    def get_buffers_batch(self):
        batch_frames = self.get_batch_frames()
        query_features = np.copy(self.frame_reader._query_samples)
        while len(batch_frames) > 0:

            frame_buffers = self.get_frame_buffers(batch_frames)
            for i, frame_buffer in enumerate(frame_buffers):
                frame_buffers[i] = np.concatenate([query_features, frame_buffer[0,:]], axis = 0)
            # norm_consts = self.get_norm_consts_per_frame(batch_frames, query_features)
            if len(frame_buffers) == 0:
                continue
            # self.normalize_frame_buffers(frame_buffers, norm_consts)
            return frame_buffers
        return []
    
class AudioBuffersDatalayer_tgt_spk(AudioBuffersDataLayer):
    def __init__(self):
        super().__init__()

    def __next__(self):
        if self._buf_count == len(self.signal):
            raise StopIteration
        self._buf_count += 1
        return (
            torch.as_tensor(self.signal[self._buf_count - 1], dtype=torch.float32),
            torch.as_tensor(self.signal[self._buf_count - 1].shape[0], dtype=torch.int64),
        )

    

