import os
import re
import copy
import math
import random
import logging
import itertools
from copy import deepcopy
import concurrent.futures
from cytoolz import groupby
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import soundfile
from tqdm import tqdm
from scipy.stats import norm
from nltk.tokenize import SyllableTokenizer

import torch.utils.data
from lhotse.cut.set import mix
from lhotse.cut import Cut, CutSet, MixedCut, MonoCut, MixTrack
from lhotse import SupervisionSet, SupervisionSegment, dill_enabled, AudioSource, Recording
from lhotse.utils import uuid4, compute_num_samples
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_hidden_length_from_sample_length,
    find_segments_from_rttm,
    json_to_cut,
    get_bounded_segment
)

from lhotse.lazy import LazyIteratorChain, LazyJsonlIterator

def mix_noise(
    cuts,
    noise_manifests,
    snr,
    mix_prob,
):
    
    mixed_cuts = []
    assert 0.0 <= mix_prob <= 1.0, "mix_prob must be between 0.0 and 1.0"
    for cut in cuts:
        if random.uniform(0.0, 1.0) > mix_prob or cut.duration == 0:
            mixed_cuts.append(cut)
            continue
        to_mix_manifest = random.choice(noise_manifests)
        to_mix_cut = json_to_cut(to_mix_manifest)
        to_mix_cut = to_mix_cut.resample(16000)
        snr = random.uniform(*snr) if isinstance(snr, (list, tuple)) else snr
        mixed = cut.mix(to_mix_cut, snr = snr)
        mixed = mixed.truncate(duration=cut.duration)
        mixed_cuts.append(mixed) 
    return CutSet.from_cuts(mixed_cuts)

def speaker_to_target_w_query(
        a_cut, 
        query,
        add_separater_audio: bool = True,
        separater_duration: int = 1,
        num_speakers: int = 4, 
        num_sample_per_mel_frame: int = 160, 
        num_mel_frame_per_asr_frame: int = 8, 
        spk_tar_all_zero: bool = False, 
        boundary_segments: bool = False):
    '''
    Get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length)
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut): Lhotse Cut instance which is MonoCut or MixedCut instance.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int): number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
        spk_tar_all_zero (Tensor): set to True gives all zero "mask"
        boundary_segments (bool): set to True to include segments containing the boundary of the cut, False by default for multi-speaker ASR training
    
    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    '''
    # get cut-related segments from rttms
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
        offsets = [track.offset for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
        offsets = [0]
    else:
        raise ValueError(f"Unsupported cut type type{cut}: only MixedCut and MonoCut are supported")
    segments_total = []
    for i, cut in enumerate(cut_list):
        if hasattr(cut, 'rttm_filepath') and cut.rttm_filepath is not None:
            rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        elif hasattr(cut, 'speaker_id') and cut.speaker_id is not None:
            rttms = SupervisionSet.from_segments([SupervisionSegment(
                id=uuid4(),
                recording_id=cut.recording_id,
                start=0,
                duration=cut.duration,
                channel=1,
                speaker=cut.speaker_id,
                language=None
            )])
        else:
            raise ValueError(f"Cut {cut.id} does not have rttm_filepath or speaker_id")
        if boundary_segments: # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(recording_id=cut.recording_id, rttms=rttms, start_after=cut.start, end_before=cut.end, tolerance=0.0)
        else: # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True)

        for seg in segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > cut.duration:
                seg.duration -= seg.end - cut.duration
            seg.start += offsets[i]
            segments_total.append(seg)
    segments_total.sort(key = lambda rttm_sup: rttm_sup.start)
    seen = set()
    seen_add = seen.add
    if isinstance(a_cut, MixedCut):
        cut = a_cut
    if 'query_speaker_id' in cut.custom:
        speaker_lst = [cut.query_speaker_id] + [s.speaker for s in segments_total] #add query speaker as the first speaker
    else:
        speaker_lst = [s.speaker for s in segments_total]

    speaker_ats = [s for s in speaker_lst if not (s in seen or seen_add(s))]
    
    speaker_to_idx_map = {
            spk: idx
            for idx, spk in enumerate(speaker_ats)
    }
        #initialize mask matrices (num_speaker, encoder_hidden_len)
    if add_separater_audio:
        encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples +  query.num_samples + separater_duration * query.sampling_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

        separater_hidden_len = get_hidden_length_from_sample_length(separater_duration * query.sampling_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

        query_hidden_len = get_hidden_length_from_sample_length(query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame) if 'query_speaker_id' in cut.custom else 0

        mask = np.zeros((num_speakers, encoder_hidden_len))
        if hasattr(query, 'rttm_filepath') and query.rttm_filepath is not None:
            query_rttms = SupervisionSet.from_rttm(query.rttm_filepath)
            query_segments_iterator = find_segments_from_rttm(recording_id=query.recording_id, rttms=query_rttms, start_after=query.start, end_before=query.end, tolerance=0.0)
            query_segments_total = []
            for seg in query_segments_iterator:
                #truncate negative start
                if seg.start < 0:
                    seg.duration += seg.start
                    seg.start = 0
                #truncate exceed segment trailing
                #exceed duration = (seg.end - query.duration)
                if seg.end > query.duration:
                    seg.duration -= seg.end - query.duration
                query_segments_total.append(seg)
            for rttm_sup in query_segments_total:
                    st = (
                                compute_num_samples(rttm_sup.start, query.sampling_rate)
                                if rttm_sup.start > 0
                                else 0
                            )
                    et = (
                                compute_num_samples(rttm_sup.end, query.sampling_rate)
                                if rttm_sup.end < query.duration
                                else compute_num_samples(query.duration, query.sampling_rate)
                            ) 
                    st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    mask[0, st_encoder_loc: et_encoder_loc] = 1
        else:
            mask[0,:query_hidden_len] = 1                

        for rttm_sup in segments_total:
            speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
            #only consider the first <num_speakers> speakers
            if speaker_idx < 4:
                st = (
                            compute_num_samples(rttm_sup.start, cut.sampling_rate)
                            if rttm_sup.start > 0
                            else 0
                        )
                et = (
                            compute_num_samples(rttm_sup.end, cut.sampling_rate)
                            if rttm_sup.end < cut.duration
                            else compute_num_samples(cut.duration, cut.sampling_rate)
                        )                   
                
                #map start time (st) and end time (et) to encoded hidden location
                st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

                mask[speaker_idx, query_hidden_len + separater_hidden_len + st_encoder_loc: query_hidden_len + separater_hidden_len + et_encoder_loc] = 1

    else:
        encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples +  query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
        query_hidden_len = get_hidden_length_from_sample_length(query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame) if 'query_speaker_id' in cut.custom else 0
        mask = np.zeros((num_speakers, encoder_hidden_len))

        if hasattr(query, 'rttm_filepath') and query.rttm_filepath is not None:
            query_rttms = SupervisionSet.from_rttm(query.rttm_filepath)
            query_segments_iterator = find_segments_from_rttm(recording_id=query.recording_id, rttms=query_rttms, start_after=query.start, end_before=query.end, tolerance=0.0)
            query_segments_total = []
            for seg in query_segments_iterator:
                if seg.start < 0:
                    seg.duration += seg.start
                    seg.start = 0
                if seg.end > query.duration:
                    seg.duration -= seg.end - query.duration
                query_segments_total.append(seg)
            for rttm_sup in query_segments_total:
                    st = (
                                compute_num_samples(rttm_sup.start, query.sampling_rate)
                                if rttm_sup.start > 0
                                else 0
                            )
                    et = (
                                compute_num_samples(rttm_sup.end, query.sampling_rate)
                                if rttm_sup.end < query.duration
                                else compute_num_samples(query.duration, query.sampling_rate)
                            ) 
                    st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    mask[0, st_encoder_loc: et_encoder_loc] = 1
        else:
            mask[0,:query_hidden_len] = 1  

        mask[0,:query_hidden_len] = 1

        for rttm_sup in segments_total:
            speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
            #only consider the first <num_speakers> speakers
            if speaker_idx < 4:
                st = (
                            compute_num_samples(rttm_sup.start, cut.sampling_rate)
                            if rttm_sup.start > 0
                            else 0
                        )
                et = (
                            compute_num_samples(rttm_sup.end, cut.sampling_rate)
                            if rttm_sup.end < cut.duration
                            else compute_num_samples(cut.duration, cut.sampling_rate)
                        )                   
                
                #map start time (st) and end time (et) to encoded hidden location
                st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

                mask[speaker_idx, query_hidden_len + st_encoder_loc:query_hidden_len + et_encoder_loc] = 1

    return mask


def get_separator_audio(freq, sr, duration, ratio):
    # Generate time values
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Generate sine wave
    y = np.sin(2 * np.pi * freq * t) * 0.1

    y[:int(sr * duration * ratio )] = 0
    y[-int(sr * duration * ratio ):] = 0
    return y

def get_query_cut(cut):
    '''
    Extract query from the cut and saved as a separate cut

    Args:
        cut: An audio cut. The cut should contain keys "query_audio_filepath", "query_offet", "query_duration"

    Returns:
        query_cut: a cut containing query information
    '''    
    if 'query_audio_filepath' in cut.custom:
        #no query is provided for query cut
        #no query is provided for query cut
        #TODO use create_cut function in asr_multispeaker_utils.py
        if cut.query_audio_filepath.find('voxceleb')!= -1:
            #change recording id to be the same as rttm line's session format
            elements = cut.query_audio_path.split('/')
            recording_id = elements[-3]+'-'+elements[-2]+'-'+elements[-1][:-4]
            query_rec = Recording.from_file(cut.query_audio_path, recording_id = recording_id)
        else:
            query_rec = Recording.from_file(cut.query_audio_filepath)
        if query_rec.sampling_rate != 16000:
            query_rec = query_rec.resample(sampling_rate=16000)
        query_sups = [SupervisionSegment(id=query_rec.id+'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration), recording_id = query_rec.id, start = 0, duration = cut.query_duration, speaker = cut.query_speaker_id)]
        #additional information for query
        custom = {
            'rttm_filepath' : cut.custom.get('query_rttm_filepath', None)
        }
        query_cut = MonoCut(id = query_rec.id +'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration),
                            start = cut.query_offset,
                            duration = cut.query_duration,
                            channel = 0,
                            recording = query_rec,
                            supervisions = query_sups)
        query_cut.custom = custom
        return query_cut
    else:
        query_rec = cut.recording if isinstance(cut, MonoCut) else cut.tracks[0].cut.recording
        query_sups = [SupervisionSegment(id=cut.id+'_query_dummy', recording_id = query_rec.id, start = 0, duration = 0, speaker = None)]
        query_cut = MonoCut(id = cut.id +'_query_no_ts_'+str(cut.start)+'_'+str(cut.duration),
                            start = 0,
                            duration = 0,
                            channel = 0,
                            recording = query_rec,
                            supervisions = query_sups)
        return query_cut
class TargetSpeakerSimulator():
    """
    This class is used to simulate multi-speaker audio data,
    which can be used for multi-speaker ASR and speaker diarization training.
    """
    def __init__(
        self, 
        manifest_filepath, 
        num_speakers, 
        simulator_type,
        min_delay=0.5,
        query_duration: List[float] = [3, 10]
    ):
        """
        Args:
            cuts (CutSet): The cutset that contains single-speaker audio cuts.
                Please make sure that the cuts have the 'speaker_id' attribute.                    
            num_speakers (int): The number of speakers in the simulated audio.
                We only simulate the samples with the fixed number of speakers.
                The variation of the number of speakers is controlled by the weights in Lhotse dataloader config.
            simulator_type (str): The type of simulator to use.
                - 'lsmix': LibriSpeechMix-style training sample.
                - 'meeting': Meeting-style training sample.
                - 'conversation': Conversation-style training sample.
            speaker_distribution (list): The distribution of speakers in the simulated audio.
                The length of the list is the maximum number of speakers.
                The list elements are the weights for each speaker.
            min_delay (float): The minimum delay between speakers
                to avoid the same starting time for multiple speakers.
        """
    
        self.manifests = LazyJsonlIterator(manifest_filepath)
        self.min_delay = min_delay
        self.num_speakers = num_speakers
        self.simulator_type = simulator_type
        self.query_duration = query_duration

        self.spk2manifests = groupby(lambda x: x["speaker_id"], self.manifests)
        self.speaker_ids = list(self.spk2manifests.keys())

        if simulator_type == 'lsmix':    
            self.simulator = self.LibriSpeechMixSimulator_tgt
        elif simulator_type == 'meeting':
            self.simulator = self.MeetingSimulator
        elif simulator_type == 'conversation':
            self.simulator = self.ConversationSimulator

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.simulator()

    def LibriSpeechMixSimulator_tgt(self):
        """
        This function simulates a LibriSpeechMix-style training sample.
        Ref:
            Paper: https://arxiv.org/abs/2003.12687
            Github: https://github.com/NaoyukiKanda/LibriSpeechMix
        """
        # Sample the speakers
        sampled_speaker_ids = random.sample(self.speaker_ids, self.num_speakers)
        # Sample the cuts for each speaker
        mono_cuts = []
        for speaker_id in sampled_speaker_ids:
            manifest = random.choice(self.spk2manifests[speaker_id])
            mono_cuts.append(json_to_cut(manifest))

        tracks = []
        offset = 0.0
        for mono_cut in mono_cuts:
            custom = {
                    'pnc': 'no',
                    'source_lang': 'en',
                    'target_lang': 'en',
                    'task': 'asr'
                }
            mono_cut.custom.update(custom)
            #select random start time and duration for each speaker according to min and max duration
            start_time, duration = get_bounded_segment(mono_cut.start, mono_cut.duration, min_duration = 0, max_duration = 20)
            mono_cut.start = start_time
            mono_cut.duration = duration
            #TODO extract mono cut text according to start and duration
            tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
            offset += random.uniform(self.min_delay, mono_cut.duration)
    
        mixed_cut = MixedCut(id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks)

        index = random.randrange(len(sampled_speaker_ids))
        query_speaker_id = sampled_speaker_ids[index]
        query_manifest_list = deepcopy(self.spk2manifests[query_speaker_id])
        query_manifest = random.choice(query_manifest_list)
        query_cut = json_to_cut(query_manifest)
        text = self.get_text(mixed_cut, query_speaker_id) if hasattr(mixed_cut, 'text') else ""
        sup = SupervisionSegment(id = mixed_cut.id, recording_id = mixed_cut.id, start = 0, duration=mixed_cut.duration, text = text)
        query_offset, query_duration = get_bounded_segment(query_cut.start, query_cut.duration, min_duration=self.query_duration[0], max_duration=self.query_duration[1])
        custom = {
                'pnc': 'no',
                'source_lang': 'en',
                'target_lang': 'en',
                'task': 'asr',
                'query_audio_filepath': query_cut.recording.sources[0].source,
                'query_speaker_id': query_speaker_id,
                'query_offset': query_offset,
                'query_duration': query_duration,
                'query_rttm_filepath': query_cut.rttm_filepath if hasattr(query_cut, 'rttm_filepath') else None,
                'custom': None 
                    }
        mixed_cut.tracks[0].cut.supervisions = [sup]
        mixed_cut.tracks[0].cut.custom.update(custom)
        
        
        return mixed_cut

    def MeetingSimulator(self):
        raise NotImplementedError("MeetingSimulator is not implemented yet.")   

    def ConversationSimulator(self):
        raise NotImplementedError("ConversationSimulator is not implemented yet.")
    
    # TODO: text is necessary for msasr and tsasr, but not for diar
    def get_text(self, cut: MixedCut, query_speaker_id) -> str:
        for i, track in enumerate(cut.tracks):
            if track.cut.speaker_id == query_speaker_id:
                return track.cut.text
        return ValueError ('Error in finding query speaker in target utterance')




