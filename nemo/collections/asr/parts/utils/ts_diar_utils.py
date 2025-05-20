# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import json
import math
import multiprocessing
import os
import shutil
from dataclasses import dataclass
from itertools import repeat
from math import ceil, floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics import detection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from nemo.collections.asr.models import EncDecClassificationModel, EncDecFrameClassificationModel
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line

from nemo.collections.asr.metrics.der import uem_timeline_from_file
from pyannote.metrics.diarization import DiarizationErrorRate
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_uniqname_from_filepath,
    get_uniq_id_with_dur,
    generate_diarization_output_lines,
    labels_to_pyannote_object,
    get_uem_object,
)

#some functions re-write for sortformer_w_query (ts_sortformer)

def rttm_to_labels_query(rttm_filename, query_offset, query_duration, query_speaker_id):
    """
    Prepare time stamps label list from query's rttm file
    """
    labels = []
    with open(rttm_filename, 'r') as f:
        for line in f.readlines():
            start, end, speaker = convert_rttm_line(line, round_digits=3)
            #speaker will always be "speech"
            if start < query_offset + query_duration and end > query_offset:
                start = str(round(float(max(0, start - query_offset)), 3))
                end = str(round(float(min(query_duration, end - query_offset)), 3))
                labels.append('{} {} {}'.format(start, end, query_speaker_id))
    return labels

def rttm_to_labels_w_query(rttm_filename, query_bias, offset, duration):
    """
    Prepare time stamps label list from rttm file
    Changed:
        1. Only rttm lines between offset and offset + duration are considered (boundary included)
        2. query_bias is added to the start and end of the rttm lines
    """
    labels = []
    with open(rttm_filename, 'r') as f:
        for line in f.readlines():
            start, end, speaker = convert_rttm_line(line, round_digits=3)
            if start < offset + duration and end > offset:
                start = str(round(float(max(0, start - offset)) + query_bias, 3))
                end = str(round(float(min(duration, end - offset)) + query_bias, 3))
                labels.append('{} {} {}'.format(start, end, speaker))
    return labels

def predlist_to_timestamps_w_query(
    batch_preds_list: List[torch.Tensor],
    audio_rttm_map_dict: Dict[str, Dict[str, Union[float, int]]],
    cfg_vad_params: OmegaConf,
    unit_10ms_frame_count: int,
    bypass_postprocessing: bool = False,
    precision: int = 2,
) -> List[List[float]]:
    """
    Converts floating point number tensor diarization results to timestamps using VAD style
    post-processing methods.

    Args:
        batch_preds_list (List[Tensor]):
            Tensor diarization results for each sample.
            Dimension: [(num_frames, num_speakers), ...]
        audio_rttm_map_dict (Dict[str, Dict[str, Union[float, int]]]):
            Dictionary mapping unique audio file names to their rttm file entries.
        cfg_vad_params (OmegaConf):
            Configuration (omega config) of VAD parameters.
        unit_10ms_frame_count (int):
            an integer indicating the number of 10ms frames in a unit.
            For example, if unit_10ms_frame_count is 8, then each frame is 0.08 seconds.
        bypass_postprocessing (bool, optional):
            If True, diarization post-processing will be bypassed.
        precision (int, optional):
            The number of decimal places to round the timestamps. Defaults to 2.

    Returns:
        total_speaker_timestamps (List[List[List[float]]]):
            A list of lists of timestamp tensors for each session (utterance)
            Levels:
                - Session-level (uniq_id) [session1_list, session2_list,...]
                    - Segment-level: [[start1, end1], [start2, end2],...]]
                        - List of start and end timestamp [start, end]
    """
    total_speaker_timestamps = []
    pp_message = "Binarization" if bypass_postprocessing else "Post-processing"
    for sample_idx, (uniq_id, audio_rttm_values) in tqdm(
        enumerate(audio_rttm_map_dict.items()), total=len(audio_rttm_map_dict), desc=pp_message
    ):
        offset = audio_rttm_values['offset']
        speaker_assign_mat = batch_preds_list[sample_idx].squeeze(dim=0)
        speaker_timestamps = [[] for _ in range(speaker_assign_mat.shape[-1])]
        for spk_id in range(speaker_assign_mat.shape[-1]):
            ts_mat = ts_vad_post_processing(
                speaker_assign_mat[:, spk_id],
                cfg_vad_params=cfg_vad_params,
                unit_10ms_frame_count=unit_10ms_frame_count,
                bypass_postprocessing=bypass_postprocessing,
            )
            # remove ts = ts_mat + offset 
            # ts_mat always start from 0, i.e. reference side should also start from 0, represents range from [offset, offset + duration]
            ts_seg_raw_list = ts_mat.tolist()
            ts_seg_list = [[round(stt, precision), round(end, precision)] for (stt, end) in ts_seg_raw_list]
            speaker_timestamps[spk_id].extend(ts_seg_list)
        total_speaker_timestamps.append(speaker_timestamps)
    return total_speaker_timestamps

def score_labels_query_speaker_only(
    AUDIO_RTTM_MAP,
    all_reference: list,
    all_hypothesis: list,
    all_uem: List[List[float]] = None,
    collar: float = 0.25,
    ignore_overlap: bool = True,
    verbose: bool = True,
) -> Optional[Tuple[DiarizationErrorRate, Dict]]:
    '''
    Calculate diarization metrics only for query / target speaker
    '''
    metric = None
    if len(all_reference) == len(all_hypothesis):
        
        logging.info("Calculate diarization metrics only for query / target speaker")

        metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=ignore_overlap)

        mapping_dict, correct_spk_count = {}, 0
        for idx, (uniq_id, audio_rttm_values) in enumerate(AUDIO_RTTM_MAP.items()):
            reference = all_reference[idx]
            hypothesis = all_hypothesis[idx]
            ref_key, ref_labels = reference
            _, hyp_labels = hypothesis

            # extract query speaker from reference
            ref_labels = ref_labels.subset([audio_rttm_values['query_speaker_id']])
            # extract speaker_0 from hypothesis
            hyp_labels = hyp_labels.subset(['speaker_0'])

            if len(ref_labels.crop(all_uem[idx]).labels()) == len(hyp_labels.labels()):
                correct_spk_count += 1
            uem_obj = None
            if all_uem is not None:
                metric(ref_labels, hyp_labels, uem=all_uem[idx], detailed=True)
            elif AUDIO_RTTM_MAP[ref_key].get('uem_filepath', None) is not None:
                uem_file = AUDIO_RTTM_MAP[ref_key].get('uem_filepath', None)
                uem_obj = uem_timeline_from_file(uem_file=uem_file, uniq_name=ref_key)
                metric(ref_labels, hyp_labels, uem=uem_obj, detailed=True)
            else:
                metric(ref_labels, hyp_labels, detailed=True)
            mapping_dict[ref_key] = metric.optimal_mapping(ref_labels, hyp_labels)

        spk_count_acc = correct_spk_count / len(all_reference)
        DER = abs(metric)
        if metric['total'] == 0:
            raise ValueError("Total evaluation time is 0. Abort.")
        CER = metric['confusion'] / metric['total']
        FA = metric['false alarm'] / metric['total']
        MISS = metric['missed detection'] / metric['total']

        itemized_errors = (DER, CER, FA, MISS)

        if verbose:
            logging.info(f"\n{metric.report()}")
        logging.info(
            f"Cumulative Results for collar {collar} sec and ignore_overlap {ignore_overlap}: \n"
            f"| FA: {FA:.4f} | MISS: {MISS:.4f} | CER: {CER:.4f} | DER: {DER:.4f} | "
            f"Spk. Count Acc. {spk_count_acc:.4f}\n"
        )

        return metric, mapping_dict, itemized_errors
    elif verbose:
        logging.warning(
            "Check if each ground truth RTTMs were present in the provided manifest file. "
            "Skipping calculation of Diariazation Error Rate"
        )
    return None

def audio_rttm_map_w_query_info(manifest, attach_dur=False):
    """
    This function creates AUDIO_RTTM_MAP which is used by all diarization components to extract embeddings,
    cluster and unify time stamps

    Args:
        manifest (str): Path to the manifest file
        attach_dur (bool, optional): If True, attach duration information to the unique name. Defaults to False.

    Returns:
        AUDIO_RTTM_MAP (dict) : Dictionary with unique names as keys and corresponding metadata as values.
    """

    AUDIO_RTTM_MAP = {}
    with open(manifest, 'r') as inp_file:
        lines = inp_file.readlines()
        logging.info("Number of files to diarize: {}".format(len(lines)))
        for line in lines:
            line = line.strip()
            dic = json.loads(line)

            meta = {
                'audio_filepath': dic['audio_filepath'],
                'rttm_filepath': dic.get('rttm_filepath', None),
                'offset': dic.get('offset', None),
                'duration': dic.get('duration', None),
                'text': dic.get('text', None),
                'num_speakers': dic.get('num_speakers', None),
                'uem_filepath': dic.get('uem_filepath', None),
                'ctm_filepath': dic.get('ctm_filepath', None),
                'query_audio_filepath': dic.get('query_audio_filepath',None),
                'query_offset': dic.get('query_offset',0),
                'query_duration': dic.get('query_duration',0),
                'query_speaker_id': dic.get('query_speaker_id',None),
                'query_rttm_filepath': dic.get('query_rttm_filepath', None)
            }
            if attach_dur:
                uniqname = get_uniq_id_with_dur(meta)
            else:
                if "uniq_id" in dic.keys():
                    uniqname = dic['uniq_id']
                else:
                    uniqname = get_uniqname_from_filepath(filepath=meta['audio_filepath'])
            uniqname += str(meta['offset']) + str(meta['duration'])
            if 'query_speaker_id' in meta.keys():
                uniqname += '_'+meta['query_speaker_id']+'_'+str(meta['query_offset'])+'_'+str(meta['query_duration'])
            if uniqname not in AUDIO_RTTM_MAP:
                meta['uniq_id'] = uniqname
                AUDIO_RTTM_MAP[uniqname] = meta
            else:

                raise KeyError(
                    f"file {meta['audio_filepath']} is already part of AUDIO_RTTM_MAP, it might be duplicated, "
                    "Note: file basename must be unique"
                )

    return AUDIO_RTTM_MAP

def get_hidden_length_from_sample_length(
    num_samples: int, 
    num_sample_per_mel_frame: int = 160, 
    num_mel_frame_per_asr_frame: int = 8
) -> int:
    """ 
    Calculate the hidden length from the given number of samples.
    This function is needed for speaker diarization with ASR model trainings.

    This function computes the number of frames required for a given number of audio samples,
    considering the number of samples per mel frame and the number of mel frames per ASR frame.

    Parameters:
        num_samples (int): The total number of audio samples.
        num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
        num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

    Returns:
        hidden_length (int): The calculated hidden length in terms of the number of frames.
    """
    mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
    hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
    return int(hidden_length)



def timestamps_to_pyannote_object_w_query(
    speaker_timestamps: List[Tuple[float, float]],
    uniq_id: str,
    audio_rttm_values: Dict[str, str],
    all_hypothesis: List[Tuple[str, Timeline]],
    all_reference: List[Tuple[str, Timeline]],
    all_uems: List[Tuple[str, Timeline]],
    out_rttm_dir: str | None,
    consider_query_in_eval: bool = True,
    use_groundtruth_query_rttm: bool = True,
):
    """
    Convert speaker timestamps to pyannote.core.Timeline object.

    Args:
        speaker_timestamps (List[Tuple[float, float]]):
            Timestamps of each speaker: start time and end time of each speaker.
        uniq_id (str):
            Unique ID of each speaker.
        audio_rttm_values (Dict[str, str]):
            Dictionary of manifest values.
        all_hypothesis (List[Tuple[str, pyannote.core.Timeline]]):
            List of hypothesis in pyannote.core.Timeline object.
        all_reference (List[Tuple[str, pyannote.core.Timeline]]):
            List of reference in pyannote.core.Timeline object.
        all_uems (List[Tuple[str, pyannote.core.Timeline]]):
            List of uems in pyannote.core.Timeline object.
        out_rttm_dir (str | None):
            Directory to save RTTMs

    Returns:
        all_hypothesis (List[Tuple[str, pyannote.core.Timeline]]):
            List of hypothesis in pyannote.core.Timeline object with an added Timeline object.
        all_reference (List[Tuple[str, pyannote.core.Timeline]]):
            List of reference in pyannote.core.Timeline object with an added Timeline object.
        all_uems (List[Tuple[str, pyannote.core.Timeline]]):
            List of uems in pyannote.core.Timeline object with an added Timeline object.
    """
    offset, dur = float(audio_rttm_values.get('offset', None)), float(audio_rttm_values.get('duration', None))
    hyp_labels = generate_diarization_output_lines(
        speaker_timestamps=speaker_timestamps, model_spk_num=len(speaker_timestamps)
    )
    hypothesis = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
    if out_rttm_dir is not None and os.path.exists(out_rttm_dir):
        with open(f'{out_rttm_dir}/{uniq_id}.rttm', 'w') as f:
            hypothesis.write_rttm(f)
    all_hypothesis.append([uniq_id, hypothesis])
    rttm_file = audio_rttm_values.get('rttm_filepath', None)
    if rttm_file is not None and os.path.exists(rttm_file):
        #reference side add query information
        if consider_query_in_eval:
            #query related
            separater_duration = 1
            query_offset = audio_rttm_values.get('query_offset',0)
            query_duration = audio_rttm_values.get('query_duration',0)
            query_speaker_id = audio_rttm_values.get('query_speaker_id',None)
            query_rttm_filepath = audio_rttm_values.get('query_rttm_filepath',None)

            # uem_lines = [[offset, dur + offset + separater_duration + query_duration]]
            uem_lines = [[0, dur + separater_duration + query_duration]]
            query_bias = separater_duration + query_duration
            org_ref_labels = rttm_to_labels_w_query(rttm_file, query_bias, offset, dur)
            ref_labels = org_ref_labels
            if query_duration == 0:
                # if multi-speaker sample
                pass
            else:
                if use_groundtruth_query_rttm:
                    if not query_rttm_filepath:
                        raise ValueError('No query_rttm_filepath, set use_groundtruth_query_rttm to be False')
                    query_ref_labels = rttm_to_labels_query(query_rttm_filepath, query_offset, query_duration, query_speaker_id)
                    #extend ref_labels after query_ref_labels
                    query_ref_labels.extend(ref_labels)
                    ref_labels = query_ref_labels
                else:
                    # start, end, speaker
                    ref_labels.insert(0, '{} {} {}'.format(0, query_duration, query_speaker_id))
        else:
            uem_lines = [[0, dur]]
            org_ref_labels = rttm_to_labels_w_query(rttm_file, 0, offset, dur)
            ref_labels = org_ref_labels
        reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
        uem_obj = get_uem_object(uem_lines, uniq_id=uniq_id)
        all_uems.append(uem_obj)
        all_reference.append([uniq_id, reference])
    return all_hypothesis, all_reference, all_uems

def convert_pred_mat_to_segments(
    audio_rttm_map_dict: Dict[str, Dict[str, str]],
    postprocessing_cfg,
    batch_preds_list: List[torch.Tensor],
    unit_10ms_frame_count: int = 8,
    bypass_postprocessing: bool = False,
    out_rttm_dir: str | None = None,
    consider_query_in_eval: bool = True,
    use_groundtruth_query_rttm: bool = True,
):
    """
    Convert prediction matrix to time-stamp segments.

    Args:
        audio_rttm_map_dict (dict): dictionary of audio file path, offset, duration and RTTM filepath.
        batch_preds_list (List[torch.Tensor]): list of prediction matrices containing sigmoid values for each speaker.
            Dimension: [(1, num_frames, num_speakers), ..., (1, num_frames, num_speakers)]
        unit_10ms_frame_count (int, optional): number of 10ms segments in a frame. Defaults to 8.
        bypass_postprocessing (bool, optional): if True, postprocessing will be bypassed. Defaults to False.

    Returns:
       all_hypothesis (list): list of pyannote objects for each audio file.
       all_reference (list): list of pyannote objects for each audio file.
       all_uems (list): list of pyannote objects for each audio file.
    """
    batch_pred_ts_segs, all_hypothesis, all_reference, all_uems = [], [], [], []
    cfg_vad_params = OmegaConf.structured(postprocessing_cfg)

    #prediction side remove query prediction
    if consider_query_in_eval:
        pass
    else:
        #remove prediction from query part
        for sample_idx, (uniq_id, audio_rttm_values) in tqdm(
            enumerate(audio_rttm_map_dict.items()), total=len(audio_rttm_map_dict), desc='Removing query preds'
        ):
            query_duration = audio_rttm_values['query_duration']
            query_hidden_len = get_hidden_length_from_sample_length(int((1+query_duration) * 16000), 160, 8)
            batch_preds_list[sample_idx] = batch_preds_list[sample_idx][:,query_hidden_len:,:]
    total_speaker_timestamps = predlist_to_timestamps_w_query(
        batch_preds_list=batch_preds_list,
        audio_rttm_map_dict=audio_rttm_map_dict,
        cfg_vad_params=cfg_vad_params,
        unit_10ms_frame_count=unit_10ms_frame_count,
        bypass_postprocessing=bypass_postprocessing,
    )
    for sample_idx, (uniq_id, audio_rttm_values) in enumerate(audio_rttm_map_dict.items()):
        speaker_timestamps = total_speaker_timestamps[sample_idx]
        if audio_rttm_values.get("uniq_id", None) is not None:
            uniq_id = audio_rttm_values["uniq_id"]
        else:
            import ipdb; ipdb.set_trace()
            assert False, "uniq_id is not found"
            uniq_id = get_uniqname_from_filepath(audio_rttm_values["audio_filepath"])
            uniq_id += str(audio_rttm_values['offset']) + str(audio_rttm_values['duration'])
            if 'query_speaker_id' in audio_rttm_map_dict.keys():
                uniq_id += '_'+audio_rttm_values['query_speaker_id']+'_'+str(audio_rttm_map_dict['query_offset'])+'_'+str(audio_rttm_values['query_duration'])
        all_hypothesis, all_reference, all_uems = timestamps_to_pyannote_object_w_query(
            speaker_timestamps,
            uniq_id,
            audio_rttm_values,
            all_hypothesis,
            all_reference,
            all_uems,
            out_rttm_dir,
            consider_query_in_eval,
            use_groundtruth_query_rttm,
        )
    return all_hypothesis, all_reference, all_uems
    
