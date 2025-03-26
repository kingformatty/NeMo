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
from pyannote.core import Annotation, Segment
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
    
