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


"""
This script provides an inference and evaluation script for end-to-end speaker diarization models.
The performance of the diarization model is measured using the Diarization Error Rate (DER).
If you want to evaluate its performance, the manifest JSON file should contain the corresponding RTTM
(Rich Transcription Time Marked) file.
Please refer to the NeMo Library Documentation for more details on data preparation for diarization inference:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit
/asr/speaker_diarization/datasets.html#data-preparation-for-inference

Usage for diarization inference:

The end-to-end speaker diarization model can be specified by "model_path".
Data for diarization is fed through the "dataset_manifest".
By default, post-processing is bypassed, and only binarization is performed.
If you want to reproduce DER scores reported on NeMo model cards, you need to apply post-processing steps.
Use batch_size = 1 to have the longest inference window and the highest possible accuracy.

python $BASEPATH/neural_diarizer/e2e_diarize_speech.py \
    model_path=/path/to/diar_sortformer_4spk_v1.nemo \
    batch_size=1 \
    dataset_manifest=/path/to/diarization_manifest.json

"""
import logging
import os
import json
import tempfile
from dataclasses import dataclass, is_dataclass
from typing import Dict, List, Optional, Union
from pyannote.core import Annotation, Segment, Timeline

import lightning.pytorch as pl
import optuna
import torch
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning import seed_everything
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import math

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.models.sortformer_diar_models_w_query import SortformerEncLabelWQueryModel
from nemo.collections.asr.models.sortformer_titanet_diar_models_w_query import SortformerTitanetEncLabelWQueryModel

from nemo.collections.asr.parts.utils.vad_utils import (
    PostProcessingParams,
    load_postprocessing_from_yaml,
)
from nemo.collections.asr.parts.utils.ts_diar_utils import (
    score_labels_query_speaker_only,
    audio_rttm_map_w_query_info,
    convert_pred_mat_to_segments,
    get_hidden_length_from_sample_length,
)

from nemo.core.config import hydra_runner
from nemo.collections.asr.parts.utils.streaming_tgt_spk_utils import FrameBatchDiarizer_tgt_spk
seed_everything(42)
torch.backends.cudnn.deterministic = True


@dataclass
class DiarizationConfig:
    """Diarization configuration parameters for inference."""

    model_path: Optional[str] = None  # Path to a .nemo file
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    no_der: bool = False
    out_rttm_dir: Optional[str] = None

    # General configs
    session_len_sec: float = -1  # End-to-end diarization session length in seconds
    batch_size: int = 1
    num_workers: int = 0
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True  # If True, postprocessing will be bypassed

    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25  # Collar in seconds for DER calculation
    ignore_overlap: bool = False  # If True, DER will be calculated only for non-overlapping segments

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # Optuna Config
    launch_pp_optim: bool = False  # If True, launch optimization process for postprocessing parameters
    optuna_study_name: str = "optim_postprocessing"
    optuna_temp_dir: str = "/tmp/optuna"
    optuna_storage: str = f"sqlite:///{optuna_study_name}.db"
    optuna_log_file: str = f"{optuna_study_name}.log"
    optuna_n_trials: int = 100000
    consider_query_in_eval: bool = True # if False, metrics are computed only over target, i.e. exclude query prediction
    use_groundtruth_query_rttm: bool = False # if False, query-session rttm will be set to all one on first spk (regardless of vad)
    add_query_noise: bool = False
    noise_path: str = ''
    query_noise_mix_prob: float = 0.3
    query_snr: Tuple[float, float] = (2.5, 12.5)
    eval_query_speaker_only: bool = False
    chunk_len_in_secs: float = 1.0
    total_buffer_in_secs: float = 4.0
    use_sortformer_titanet: bool = False



def optuna_suggest_params(postprocessing_cfg: PostProcessingParams, trial: optuna.Trial) -> PostProcessingParams:
    """
    Suggests hyperparameters for postprocessing using Optuna.
    See the following link for `trial` instance in Optuna framework.
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial

    Args:
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        trial (optuna.Trial): The Optuna trial object used to suggest hyperparameters.

    Returns:
        PostProcessingParams: The updated postprocessing configuration with suggested hyperparameters.
    """
    postprocessing_cfg.onset = trial.suggest_float("onset", 0.4, 0.8, step=0.01)
    postprocessing_cfg.offset = trial.suggest_float("offset", 0.4, 0.9, step=0.01)
    postprocessing_cfg.pad_onset = trial.suggest_float("pad_onset", 0.1, 0.5, step=0.01)
    postprocessing_cfg.pad_offset = trial.suggest_float("pad_offset", 0.0, 0.2, step=0.01)
    postprocessing_cfg.min_duration_on = trial.suggest_float("min_duration_on", 0.0, 0.75, step=0.01)
    postprocessing_cfg.min_duration_off = trial.suggest_float("min_duration_off", 0.0, 0.75, step=0.01)
    return postprocessing_cfg


def get_tensor_path(cfg: DiarizationConfig) -> str:
    """
    Constructs the file path for saving or loading prediction tensors based on the configuration.

    Args:
        cfg (DiarizationConfig): The configuration object containing model and dataset details.

    Returns:
        str: The constructed file path for the prediction tensor.
    """
    tensor_filename = os.path.basename(cfg.dataset_manifest).replace("manifest.", "").replace(".json", "")
    model_base_path = os.path.dirname(cfg.model_path)
    model_id = os.path.basename(cfg.model_path).replace(".ckpt", "").replace(".nemo", "")
    bpath = f"{model_base_path}/pred_tensors"
    if not os.path.exists(bpath):
        os.makedirs(bpath)
    tensor_path = f"{bpath}/__{model_id}__{tensor_filename}.pt"
    return tensor_path


def diarization_objective(
    trial: optuna.Trial,
    postprocessing_cfg: PostProcessingParams,
    temp_out_dir: str,
    infer_audio_rttm_dict: Dict[str, Dict[str, str]],
    diar_model_preds_total_list: List[torch.Tensor],
    collar: float = 0.25,
    ignore_overlap: bool = False,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization in speaker diarization.

    This function evaluates the diarization performance using a set of postprocessing parameters
    suggested by Optuna. It converts prediction matrices to time-stamp segments, scores the
    diarization results, and returns the Diarization Error Rate (DER) as the optimization metric.

    Args:
        trial (optuna.Trial): The Optuna trial object used to suggest hyperparameters.
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        temp_out_dir (str): Temporary directory for storing intermediate outputs.
        infer_audio_rttm_dict (Dict[str, Dict[str, str]]): Dictionary containing audio file paths,
            offsets, durations, and RTTM file paths.
        diar_model_preds_total_list (List[torch.Tensor]): List of prediction matrices containing
            sigmoid values for each speaker.
            Dimension: [(1, num_frames, num_speakers), ..., (1, num_frames, num_speakers)]
        collar (float, optional): Collar in seconds for DER calculation. Defaults to 0.25.
        ignore_overlap (bool, optional): If True, DER will be calculated only for non-overlapping segments.
            Defaults to False.

    Returns:
        float: The Diarization Error Rate (DER) for the given set of postprocessing parameters.
    """
    with tempfile.TemporaryDirectory(dir=temp_out_dir, prefix="Diar_PostProcessing_") as local_temp_out_dir:
        if trial is not None:
            postprocessing_cfg = optuna_suggest_params(postprocessing_cfg, trial)
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(
            audio_rttm_map_dict=infer_audio_rttm_dict,
            postprocessing_cfg=postprocessing_cfg,
            batch_preds_list=diar_model_preds_total_list,
            unit_10ms_frame_count=8,
            bypass_postprocessing=False,
        )
        metric, mapping_dict, itemized_errors = score_labels(
            AUDIO_RTTM_MAP=infer_audio_rttm_dict,
            all_reference=all_refs,
            all_hypothesis=all_hyps,
            all_uem=all_uems,
            collar=collar,
            ignore_overlap=ignore_overlap,
        )
        der = abs(metric)
    return der



def run_optuna_hyperparam_search(
    cfg: DiarizationConfig,  # type: DiarizationConfig
    postprocessing_cfg: PostProcessingParams,
    infer_audio_rttm_dict: Dict[str, Dict[str, str]],
    preds_list: List[torch.Tensor],
    temp_out_dir: str,
):
    """
    Run Optuna hyperparameter optimization for speaker diarization.

    Args:
        cfg (DiarizationConfig): The configuration object containing model and dataset details.
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        infer_audio_rttm_dict (dict): dictionary of audio file path, offset, duration and RTTM filepath.
        preds_list (List[torch.Tensor]): list of prediction matrices containing sigmoid values for each speaker.
            Dimension: [(1, num_frames, num_speakers), ..., (1, num_frames, num_speakers)]
        temp_out_dir (str): temporary directory for storing intermediate outputs.
    """
    worker_function = lambda trial: diarization_objective(
        trial=trial,
        postprocessing_cfg=postprocessing_cfg,
        temp_out_dir=temp_out_dir,
        infer_audio_rttm_dict=infer_audio_rttm_dict,
        diar_model_preds_total_list=preds_list,
        collar=cfg.collar,
    )
    study = optuna.create_study(
        direction="minimize", study_name=cfg.optuna_study_name, storage=cfg.optuna_storage, load_if_exists=True
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    if cfg.optuna_log_file is not None:
        logger.addHandler(logging.FileHandler(cfg.optuna_log_file, mode="a"))
    logger.addHandler(logging.StreamHandler())
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    study.optimize(worker_function, n_trials=cfg.optuna_n_trials)


@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:
    """Main function for end-to-end speaker diarization inference."""
    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None:
        raise ValueError("cfg.model_path cannot be None. Please specify the path to the model.")

    #consider_query_in_eval
    #use groundtruth query rttm

    logging.info(f'Consider query in eval: {cfg.consider_query_in_eval}')

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    if cfg.use_sortformer_titanet:
        if cfg.model_path.endswith(".ckpt"):
            diar_model = SortformerTitanetEncLabelWQueryModel.load_from_checkpoint(
                checkpoint_path=cfg.model_path, map_location=map_location, strict=False
            )
        elif cfg.model_path.endswith(".nemo"):
            diar_model = SortformerTitanetEncLabelWQueryModel.restore_from(restore_path=cfg.model_path, map_location=map_location)
        else:
            raise ValueError("cfg.model_path must end with.ckpt or.nemo!")
    else:
        if cfg.model_path.endswith(".ckpt"):
            diar_model = SortformerEncLabelWQueryModel.load_from_checkpoint(
                checkpoint_path=cfg.model_path, map_location=map_location, strict=False
            )
        elif cfg.model_path.endswith(".nemo"):
            diar_model = SortformerEncLabelWQueryModel.restore_from(restore_path=cfg.model_path, map_location=map_location)
        else:
            raise ValueError("cfg.model_path must end with.ckpt or.nemo!")       

    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)

    diar_model = diar_model.eval()
    diar_model._cfg.test_ds.manifest_filepath = cfg.dataset_manifest
    infer_audio_rttm_dict = audio_rttm_map_w_query_info(cfg.dataset_manifest)
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    diar_model._cfg.test_ds.pin_memory = False

    # Model setup for inference
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model._cfg.test_ds.use_lhotse = True
    if cfg.add_query_noise:
        with open_dict(diar_model._cfg.test_ds):
            diar_model._cfg.test_ds['query_noise_path'] = cfg.noise_path
            diar_model._cfg.test_ds['query_noise_mix_prob'] = cfg.query_noise_mix_prob
            diar_model._cfg.test_ds['query_snr'] = cfg.query_snr
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)
    if 'diar_model_streaming_mode' in diar_model._cfg:
        diar_model._cfg.diar_model_streaming_mode = cfg.get('diar_model_streaming_mode', False)
    else:
        diar_model.streaming_mode = False
    postprocessing_cfg = load_postprocessing_from_yaml(cfg.postprocessing_yaml)
    tensor_path = get_tensor_path(cfg)

    # if os.path.exists(tensor_path):
    #     logging.info(
    #         f"A saved prediction tensor has been found. Loading the saved prediction tensors from {tensor_path}..."
    #     )
    #     diar_model_preds_total_list = torch.load(tensor_path)
    # else:
    logging.info(f"No saved prediction tensors found. Running inference on the dataset...")
    #offline inference
    # diar_model.test_batch()
    
    #online inference
    
    #configuration setup
    feature_stride = diar_model._cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * 8 # model_stride
    chunk_len = float(cfg.chunk_len_in_secs)
    total_buffer = cfg.total_buffer_in_secs
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)

    logging.info(f"Chunk length in secs: {chunk_len}, Total buffer in secs: {total_buffer}, Mid delay: {mid_delay}, Tokens per chunk: {tokens_per_chunk}")
    
    #declare framebatcher
    frame_diarizer = FrameBatchDiarizer_tgt_spk(
        diar_model = diar_model,
        frame_len = chunk_len,
        total_buffer = total_buffer,
        batch_size = cfg.batch_size,
        diar_model_streaming_mode = cfg.diar_model_streaming_mode,
    )
    diar_model_preds_total_list = []

    with open(cfg.dataset_manifest, 'r', encoding='utf-8') as f:
        for l in tqdm(f, desc = 'Sample:'):
            frame_diarizer.reset()
            row = json.loads(l.strip())
            audio_file = row['audio_filepath']
            offset = row['offset']
            duration = row['duration']
            query_audio_file = row['query_audio_filepath']
            query_offset = row['query_offset']
            query_duration = row['query_duration']
            separater_freq = diar_model._cfg.test_ds.get('separater_freq', 500)
            separater_duration = diar_model._cfg.test_ds.get('separater_duration', 1)
            separater_unvoice_ratio = diar_model._cfg.test_ds.get('separater_unvoice_ratio', 0.3)
            
            frame_diarizer.read_audio_file(
                audio_file,
                offset,
                duration,
                query_audio_file,
                query_offset,
                query_duration, 
                separater_freq,
                separater_duration,
                separater_unvoice_ratio,
                mid_delay,
                model_stride_in_secs,
                tokens_per_chunk,
            )

            frame_diarizer.infer_logits()

            #remove leading silcence from all_diar_preds and prepend with query_pred
            #override tailing diar preds config
            leading_silence_len = get_hidden_length_from_sample_length((frame_diarizer.frame_bufferer.feature_buffer_len - frame_diarizer.frame_bufferer.feature_frame_len), 160, 8)
            #aggreate mid-buffer diar preds
            # leading_silence_len = get_hidden_length_from_sample_length((frame_diarizer.frame_bufferer.feature_buffer_len - frame_diarizer.frame_bufferer.feature_frame_len) / 2, 160, 8)
            
            diar_preds = frame_diarizer.all_diar_preds[:,leading_silence_len:]
            diar_preds = torch.cat([frame_diarizer.query_pred, diar_preds], dim=1)
            diar_model_preds_total_list.append(diar_preds)

            # parent_dir = '/home/jinhanw/workdir/workdir_nemo_diarization/sortformer_infer/saved/temp'
            # os.makedirs(parent_dir, exist_ok=True)
            # import pickle; import numpy as np;
            # with open(os.path.join(parent_dir, 'all_diar_preds.pickle'), 'wb') as f:
            #     pickle.dump(diar_preds, f)
            # import ipdb; ipdb.set_trace()
            # break
   

    # diar_model_preds_total_list = diar_model.preds_total_list
    # torch.save(diar_model.preds_total_list, tensor_path)
    if cfg.launch_pp_optim:
        # Launch a hyperparameter optimization process if launch_pp_optim is True
        run_optuna_hyperparam_search(
            cfg=cfg,
            postprocessing_cfg=postprocessing_cfg,
            infer_audio_rttm_dict=infer_audio_rttm_dict,
            preds_list=diar_model_preds_total_list,
            temp_out_dir=cfg.optuna_temp_dir,
        )

    # Evaluation
    if not cfg.no_der:
        if cfg.out_rttm_dir is not None and not os.path.exists(cfg.out_rttm_dir):
            os.mkdir(cfg.out_rttm_dir)
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(
            infer_audio_rttm_dict,
            postprocessing_cfg=postprocessing_cfg,
            batch_preds_list=diar_model_preds_total_list,
            unit_10ms_frame_count=8,
            bypass_postprocessing=cfg.bypass_postprocessing,
            out_rttm_dir=cfg.out_rttm_dir,
            consider_query_in_eval=cfg.consider_query_in_eval,
            use_groundtruth_query_rttm=cfg.use_groundtruth_query_rttm,
        )
        logging.info(f"Evaluating the model on the {len(diar_model_preds_total_list)} audio segments...")
        metrics, mapping_dict, itemized_errors = score_labels(
            AUDIO_RTTM_MAP=infer_audio_rttm_dict,
            all_reference=all_refs,
            all_hypothesis=all_hyps,
            all_uem=all_uems,
            collar=cfg.collar,
            ignore_overlap=cfg.ignore_overlap,
        )
        logging.info(f"PostProcessingParams: {postprocessing_cfg}")
        #save metrics report to csv at the same directory as the rttm_folder
        report_path = os.path.join('/'.join(cfg.out_rttm_dir.split('/')[:-1]), 'metrics_report.csv')
        report = metrics.report(display = False)
        report.to_csv(report_path)
    ## additional metrics
    
    # 1. diarization error rate for query/target speaker, i.e.first row of  the groundtruth and prediction (temporaly blindly trust the model predicts the query speaker as speaker 0)

    if cfg.consider_query_in_eval and cfg.eval_query_speaker_only:
        metrics, mapping_dict, itemized_errors = score_labels_query_speaker_only(
            AUDIO_RTTM_MAP=infer_audio_rttm_dict,
            all_reference=all_refs,
            all_hypothesis=all_hyps,
            all_uem=all_uems,
            collar=cfg.collar,
            ignore_overlap=cfg.ignore_overlap,
        )
        #save metrics report to csv at the same directory as the rttm_folder
        report_path = os.path.join('/'.join(cfg.out_rttm_dir.split('/')[:-1]), 'metrics_report_tsder.csv')
        report = metrics.report(display = False)
        report.to_csv(report_path)
if __name__ == '__main__':
    main()
