---
name: nemotron-3.5-asr-streaming-0.6b
description: >-
  Fine-tune the nemotron-3.5-asr-streaming-0.6b Cache-Aware FastConformer-RNNT
  streaming, prompt-based MULTILINGUAL ASR model (~0.6B params). Use this to add
  or improve languages via per-clip target_lang prompt conditioning, train on
  tarred NeMo/Lhotse shards with a FIXED STEP BUDGET (Lhotse is an IterableDataset
  with no len(), so epochs do not apply), evaluate honestly at the low-latency
  streaming operating point you will actually deploy (att_context_size=[56,0],
  0 ms look-ahead), use multilingual replay to avoid catastrophic forgetting,
  scale in-language data where a language is weak, and export/deploy via NVIDIA
  Riva gRPC streaming or NeMo inference. Trigger phrases: "fine-tune nemotron
  streaming", "improve streaming ASR for Italian/Romanian/Greek/Bulgarian",
  "add a language to the streaming model", "low-latency multilingual ASR",
  "cache-aware FastConformer fine-tune", "streaming RNNT fine-tune". This skill
  is NOT for: basic transcription with default off-the-shelf models, non-streaming
  offline-only adaptation (covered by the offline parakeet recipe), or speaker
  diarization.
---

# Fine-Tune nemotron-3.5-asr-streaming-0.6b (Multilingual Streaming ASR)

This recipe fine-tunes the nemotron-3.5-asr-streaming-0.6b model — a Cache-Aware FastConformer-RNNT streaming, prompt-based multilingual ASR model — to add or sharpen specific languages. Language is conditioned per-clip through a `target_lang` prompt tag, data is streamed from tarred NeMo/Lhotse shards, and training runs on a fixed step budget rather than epochs. You evaluate at the exact low-latency streaming setting you intend to deploy, then export the same-architecture checkpoint into Riva or NeMo.

## The recipe at a glance

1. **Tarred data** — point the trainer at tarred NeMo/Lhotse shards for the target languages; no per-file unpacking, streamed efficiently by NeMo/Lhotse. Every cut carries a `target_lang` tag.
2. **Fine-tune from base** — full fine-tune from the base checkpoint (`init_from_nemo_model`) with the same cache-aware FastConformer-RNNT recipe, conditioned on each clip's language tag.
3. **Evaluate at deploy-time latency** — measure WER on a held-out set the model never saw, in the lowest-latency streaming mode (`att_context_size=[56,0]`, 0 ms look-ahead) — the most demanding condition.
4. **Add data where weak** — mix in more in-language data for the languages that lag, then retrain. More in-language data keeps helping; measure rather than assume.
5. **Export & deploy** — same architecture as the base, so it drops straight into NVIDIA Riva for gRPC streaming or NeMo for inference; pick the latency/accuracy operating point at inference via `att_context_size`.

## Architecture

```
  tarred Lhotse shards                speech_to_text_finetune.py
  (each cut tagged                    init_from_nemo_model
   target_lang: <it-IT>, ...)         cache-aware streaming RNNT
        |                             FastConformer encoder
        |   use_lhotse=true                  |
        |   is_tarred=true                   |
        +------------------>  [ fine-tune ]  +------>  fine-tuned .nemo
                                                            |
                                          +-----------------+-----------------+
                                          |                                   |
                                   Riva .riva                            NeMo inference
                                   gRPC streaming                        (manifest + target_lang)
                                   (att_context_size                     pick att_context_size
                                    chosen per request)                  at decode time
```

## What makes this model different

- **Cache-aware streaming** — you choose the latency/accuracy operating point at inference time via `att_context_size = [left_context, look_ahead]`. The model card lists the supported settings (commonly several look-ahead options, e.g. `[56,13]`, `[56,6]`, `[56,3]`, `[56,1]`, `[56,0]`); this recipe benchmarks at `[56,0]` = **0 ms look-ahead (lowest latency)**. The exact numbers are model-specific, so read the model card — the principle is fixed: larger right-context (look-ahead) = more latency but more accuracy; zero look-ahead = lowest latency, hardest condition.
- **Prompt-based language conditioning** — language is set per utterance via a `target_lang` prompt tag, not a separate model per language. Getting the tag right (and using a value the model recognizes) is essential.
- **RNNT decoder** — transducer decoding for streaming-friendly, monotonic, low-latency output.
- **~0.6B parameters** — small enough to fine-tune on a single GPU for a quick pass, scaling cleanly to multi-GPU.
- **Lhotse/tarred iterable data pipeline** — data is an `IterableDataset` (no `len()`), which is why training is scheduled on a fixed step budget instead of epochs.

## Prerequisites

### Software

The fine-tune runs on standard NeMo ASR tooling. Install via pip or conda, or pull the prebuilt NGC container.

```bash
# pip (into a fresh env, Python 3.10+)
pip install -U "nemo_toolkit[asr]"

# or conda + pip
conda create -n nemo-asr python=3.10 -y
conda activate nemo-asr
pip install -U "nemo_toolkit[asr]"
```

Docker (recommended for reproducibility — CUDA, PyTorch, NeMo pinned together):

```bash
docker run --gpus all -it --rm --shm-size=16g \
  nvcr.io/nvidia/nemo:24.07 bash
```

Lhotse ships bundled with recent NeMo releases — no separate install needed; it is what streams the tarred shards as an `IterableDataset`.

Currently, you also need a checkout of an forked NeMo repo for the nemotron-3.5-asr-streaming model support and training/eval scripts (e.g. `speech_to_text_finetune.py`):

```bash
git clone https://github.com/NVIDIA/NeMo.git
# scripts live under NeMo/examples/asr/
```

Verify the install:

```bash
python -c "import nemo, torch; import nemo.collections.asr as asr; print('NeMo', nemo.__version__, '| CUDA', torch.cuda.is_available())"
```

### Hardware

This is a ~0.6B Cache-Aware FastConformer-RNNT model. The dominant memory cost is the RNNT loss joint tensor, which is `O(B*T*U*V)` (batch x acoustic frames x text tokens x vocab) — so VRAM scales with **sequence length**, not just batch size. Long utterances or large vocab force the batch down. Use bucketing and cap `max_duration` to keep the joint tensor in check.

| GPU | VRAM | Typical batch | Note |
|-----|------|---------------|------|
| H100 / A100 80GB | 80 GB | 32-64 | Headroom for long clips; best multi-GPU node |
| L40S | 48 GB | 16-32 | Strong single-GPU; watch the joint on long utterances |
| A10G / L4 / 4090 | 24 GB | 8-16 | Fine for a quick pass; lower `max_duration`, use bucketing |
| T4 | 16 GB | 4-8 | Works but tight; short clips + grad accumulation |

A single 24GB+ GPU is enough for a quick fine-tune pass over the ~290 h mix. For the fuller ~4,300 h run (Step 4), use multi-GPU DDP (e.g. 4-8x A100/H100) — set `trainer.devices`, keep `trainer.strategy=ddp`, and note `use_distributed_sampler=false` because Lhotse handles sharding itself.

> An epoch on ~290 h is **minutes**, not hours, on a modern GPU — and Lhotse's `IterableDataset` has no `len()`. Schedule the run by a **fixed step budget**, not epochs (see Step 2).

## Step 1 — Prepare Data (tarred, language-tagged)

`nemotron-3.5-asr-streaming-0.6b` is a prompt-based, multilingual Cache-Aware FastConformer-RNNT. Two data decisions dominate everything else: **match the model's text style**, and **tag every clip with a language the model recognizes**. Get these wrong and the model trains against the wrong target distribution or routes the prompt incorrectly.

### Match the base model's text style — punctuated, properly-cased

The base model emits **punctuated, properly-cased** multilingual text. Your transcripts must look like its output. Do **not** lowercase or strip punctuation the way you would for an English non-PnC model — that teaches the model to unlearn what it already does well and inflates WER at eval time.

| Model style | Example target `text` | What to do |
|-------------|----------------------|------------|
| PnC (this model) | `Ciao, come stai?` | Keep case + punctuation as-is |
| non-PnC (English RNNT) | `ciao come stai` | (does NOT apply here) |

### Every clip carries a recognized `target_lang` tag

Language is conditioned **per utterance** via a `target_lang` prompt tag — there is no global language flag. The tag drives prompt-based language conditioning, so it must be a value the model knows. A wrong or unknown tag does not error; it **silently routes the prompt incorrectly** and the language degrades.

Use the model's BCP-47-style tags, e.g.:

| Language | `target_lang` |
|----------|---------------|
| Italian | `it-IT` |
| Romanian | `ro-RO` |
| Greek | `el-GR` |
| Bulgarian | `bg-BG` |

Both training and inference read this `target_lang` field as the per-utterance prompt. The tag must be present on every line and must survive into the tarred Lhotse cuts.

### Manifest line format (prompt / multilingual)

NeMo manifest, one JSON object per line. Beyond the standard `audio_filepath` / `text` / `duration`, each line carries a single `target_lang` field — that one tag is what conditions the language:

```json
{"audio_filepath": "/data/it/cv_000123.wav", "text": "Ciao, come stai?", "duration": 4.2, "target_lang": "it-IT"}
```

Rules:
- `audio_filepath`: **absolute** path to **16 kHz mono** audio.
- `text`: punctuated, properly-cased transcript (see above).
- `duration`: float seconds (compute it; do not guess — Lhotse uses it for bucketing).
- `target_lang`: the recognized language tag (e.g. `it-IT`, `es-US`). This is the only language field the model needs — both training and inference read it as the per-utterance prompt.

### The proven data mix (~290 h, four languages, balanced)

The validated recipe is a **balanced ~290-hour mix across four languages** — Italian, Romanian, Greek, Bulgarian — assembled from public multilingual corpora (read speech, Common Voice, web video). "Balanced" means roughly equal hours per language so no single language dominates the shared decoder.

Hold out **FLEURS test splits** (not in training) as the honest, in-the-wild per-language benchmark — never evaluate on data the model trained on.

### Tag + verify the manifest (fail fast on bad tags)

Set the per-utterance `target_lang`, match text style, and **assert every line has a recognized tag** before you spend GPU hours:

```python
import json

KNOWN_TAGS = {"it-IT", "ro-RO", "el-GR", "bg-BG"}  # extend to tags the model knows

def tag_and_verify(in_path, out_path, lang_tag):
    assert lang_tag in KNOWN_TAGS, f"unknown tag {lang_tag} — model will misroute the prompt"
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            e = json.loads(line)
            e["target_lang"] = lang_tag
            # DO NOT lowercase / strip punctuation: this is a PnC model
            e["text"] = e["text"].strip()
            assert e["text"], "empty transcript"
            assert e["target_lang"] in KNOWN_TAGS
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
            n += 1
    print(f"{out_path}: {n} lines, all tagged {lang_tag}")

tag_and_verify("it_raw.json", "it_tagged.json", "it-IT")
tag_and_verify("bg_raw.json", "bg_tagged.json", "bg-BG")
# ... ro-RO, el-GR
```

`ensure_ascii=False` keeps non-Latin scripts (Greek, Cyrillic) readable in the manifest.

### Build tarred NeMo/Lhotse shards

Convert each per-language manifest into **tarred shards**. Tarred data is streamed efficiently by NeMo/Lhotse with **no per-file unpacking** — at scale this is a **3–10x IO speedup** and is the only sane way to feed a streaming, iterable dataloader.

```bash
python NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
  --manifest_path=it_tagged.json \
  --target_dir=tarred/it \
  --num_shards=64 \
  --max_duration=20.0 \
  --min_duration=0.1 \
  --shuffle \
  --workers=16
```

- `--num_shards`: pick so shards are roughly 0.5–2 GB and `num_shards >= num_GPUs * num_workers` for clean sharding.
- `--max_duration` / `--min_duration`: drop outliers that waste padding / are too short to be useful.
- **Preserve `target_lang`:** the converter carries manifest fields into the tarred metadata, so `target_lang` survives into the Lhotse cuts. After conversion, spot-check a shard's `tarred_audio_manifest.json` to confirm `target_lang` is present on each cut — if it is dropped, per-utterance language conditioning silently breaks.

Repeat per language; in Step 2 you point the trainer at all four tarred sets (with `is_tarred=true`, `use_lhotse=true`) and weight them for a balanced mix.

### Data volume guidance

Grounded in the proven runs. See **Step 4** for the +1000 h/language scale-up.

| Scenario | Hours | Expected outcome |
|----------|-------|------------------|
| Pilot (per weak language) | ~50–75 h/lang | Enough to make a near-unusable language genuinely useful |
| Proven 4-lang gain | ~250–300 h total, balanced | The validated mix: large relative WER drops across all four (see Step 3) |
| In-domain scale-up | +~1000 h/lang (parliamentary, MOSEL/VoxPopuli) | Further gains on the weakest languages — but uneven across languages/domains, so **measure, don't assume** (Step 4) |

## Step 2 — Fine-Tune (full FT, fixed step budget)

This is a **full** fine-tune (all encoder + decoder + joint weights) from the base checkpoint, reusing the exact Cache-Aware FastConformer-RNNT recipe the base was trained with. You only change the data (your tarred Lhotse shards), the conditioning (per-utterance `target_lang`), and the schedule (a fixed step budget). Do not change the architecture, the att_context_size training setup, or the tokenizer.

Load the base weights with `init_from_nemo_model` pointing at the local `.nemo` file. If instead you are pulling a published checkpoint by name, drop the `.nemo` path and use `+init_from_pretrained_model=nvidia/nemotron-3.5-asr-streaming-0.6b`.

```bash
NEMO=/opt/NeMo
BASE=/path/to/nemotron-3.5-asr-streaming-0.6b.nemo
export PYTHONPATH=$NEMO:$PYTHONPATH # we will be using the latest nemo framework with nemotron-3.5-asr-streaming support
MAX_STEPS=20000          # computed below — drives BOTH the trainer and the LR scheduler
WARMUP=$(( MAX_STEPS / 20 ))   # ~5% of MAX_STEPS

python ${NEMO}/examples/asr/speech_to_text_finetune.py \
  --config-path=../conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming \
  init_from_nemo_model=${BASE} \
  \
  model.train_ds.use_lhotse=true \
  model.train_ds.is_tarred=true \
  model.train_ds.tarred_audio_filepaths=/data/tarred/train/audio__OP_0..511_CL_.tar \
  model.train_ds.manifest_filepath=/data/tarred/train/tarred_audio_manifest.json \
  model.train_ds.lang_field="target_lang" \
  ++model.train_ds.prompt_mode=true \
  model.train_ds.batch_size=16 \
  model.train_ds.shuffle=true \
  ++model.train_ds.use_distributed_sampler=false \
  \
  model.validation_ds.use_lhotse=true \
  model.validation_ds.is_tarred=false \
  model.validation_ds.manifest_filepath=/data/fleurs_val/manifest.json \
  model.validation_ds.lang_field="target_lang" \
  model.validation_ds.batch_size=16 \
  \
  model.optim.name=adamw \
  model.optim.lr=1e-4 \
  model.optim.weight_decay=1e-3 \
  model.optim.sched.name=CosineAnnealing \
  model.optim.sched.warmup_steps=${WARMUP} \
  model.optim.sched.min_lr=1e-6 \
  model.optim.sched.max_steps=${MAX_STEPS} \
  \
  trainer.devices=1 \
  trainer.max_steps=${MAX_STEPS} \
  trainer.max_epochs=-1 \
  trainer.val_check_interval=2000 \
  trainer.precision=bf16-mixed \
  trainer.gradient_clip_val=1.0 \
  trainer.accumulate_grad_batches=1 \
  ++trainer.use_distributed_sampler=false \
  \
  exp_manager.exp_dir=/exp/nemotron_asr_ft \
  exp_manager.name=nemotron35_streaming_4lang_ft \
  exp_manager.create_checkpoint_callback=true \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=3
```

**Streaming context — leave it alone unless you have a reason.** The base ships with a multi-context training setup (`model.encoder.att_context_size` is a *list* of contexts, with `att_context_style=chunked_limited`) so one checkpoint serves multiple latencies at inference. Train exactly as the base was trained — keep the default list — so the fine-tuned model preserves the same latency/accuracy operating points (including the lowest-latency `[56,0]` setting you'll evaluate and deploy). Only override `model.encoder.att_context_size`/`att_context_style` if you are deliberately specializing to a single latency, which costs you the others.

**Prompt conditioning.** `lang_field="target_lang"` tells Lhotse which manifest field carries the per-clip language tag; the prompt-based front end turns that tag into the language-conditioning prompt. The tag value must be one the model recognizes (see Step 1) and must be present on every utterance — a missing or unknown tag silently degrades conditioning rather than erroring.

### Why a step budget, not epochs (Lhotse IterableDataset)

Tarred Lhotse data is exposed as a PyTorch `IterableDataset` with **no `len()`**. Epoch-based scheduling cannot work: `CosineAnnealing` needs the total number of steps to shape its decay, and with no dataset length it cannot infer one. The failure mode is quiet and expensive — the scheduler decays the LR toward `min_lr` far too early, so the **last large fraction of training runs at ~0 LR** and learns almost nothing. You think you trained for N epochs; you effectively trained for a fraction of one.

Fix it by computing a step budget yourself and pinning it on **both** the trainer and the scheduler:

```python
import math
total_train_utts = 250_000      # number of utterances in your tarred manifest(s)
per_gpu_batch    = 16
num_gpus         = 1
accum            = 1
target_epochs    = 5            # how many passes over the data you actually want

global_batch  = per_gpu_batch * num_gpus * accum
steps_per_epoch = math.ceil(total_train_utts / global_batch)
MAX_STEPS = steps_per_epoch * target_epochs
WARMUP    = max(1, MAX_STEPS // 20)   # ~5%

assert MAX_STEPS > 0
print(f"steps_per_epoch={steps_per_epoch}  MAX_STEPS={MAX_STEPS}  WARMUP={WARMUP}")
```

Then enforce these invariants:
- Set the same value on `trainer.max_steps` and `model.optim.sched.max_steps` (assert `sched.max_steps >= trainer.max_steps` — if the scheduler budget is smaller, LR hits the floor before training ends).
- Set `trainer.max_epochs=-1` so steps, not epochs, terminate the run.
- `trainer.val_check_interval` must be an **integer number of steps** (e.g. `2000`), never a float fraction — a fraction is interpreted relative to epoch length, which is undefined here.
- `++model.train_ds.use_distributed_sampler=false` (and `++trainer.use_distributed_sampler=false`): Lhotse handles its own sharding/shuffling across ranks; letting Lightning wrap it in a `DistributedSampler` double-shards or crashes.

### Single GPU vs multi-GPU

Start single-GPU for a quick pass — on the ~290 h mix an "epoch" is **minutes, not hours**, so you get a first WER read fast:

```bash
trainer.devices=1
```

For the fuller run, scale out with DDP and grow the global batch. **Recompute `MAX_STEPS`** for the new global batch (more GPUs ⇒ larger global batch ⇒ fewer steps per epoch), and set both `trainer.max_steps` and `sched.max_steps` to the new value:

```bash
trainer.devices=8 \
trainer.strategy=ddp \
++trainer.use_distributed_sampler=false \
model.train_ds.batch_size=16          # per-GPU; global = 16 * 8 = 128
# global_batch went 16 -> 128, so steps_per_epoch (and MAX_STEPS) drop ~8x — recompute!
```

Keep `use_distributed_sampler=false` under DDP for the reason above. If you need a larger effective batch than fits in memory, raise `trainer.accumulate_grad_batches` and fold it into `global_batch` when computing `MAX_STEPS`.

### Hyperparameter reference

| Knob | Value | Notes |
|---|---|---|
| Optimizer | AdamW | base recipe default |
| Peak LR | `1e-4` | full FT from a converged base |
| Scheduler | `CosineAnnealing` | needs explicit `max_steps` (see above) |
| `warmup_steps` | ~5% of `MAX_STEPS` | `MAX_STEPS // 20` |
| `min_lr` | `1e-6` | LR floor at end of cosine decay |
| `max_steps` basis | `ceil(total_utts / global_batch) * target_epochs` | set on **both** trainer and sched |
| Per-GPU batch | `16` | tune to memory; affects `global_batch` |
| `accumulate_grad_batches` | `1` | raise for larger effective batch |
| Precision | `bf16-mixed` | |
| `gradient_clip_val` | `1.0` | RNNT loss can spike early |
| `weight_decay` | `1e-3` | base recipe default |
| `val_check_interval` | integer steps (e.g. `2000`) | never a float fraction |

> Multilingual note: a full FT on only the four target languages will **catastrophically forget** the base model's other languages. If you must retain them, mix a replay slice of the original languages into the tarred pool rather than fine-tuning on the four languages alone.

## Keeping Other Languages Alive (Multilingual Replay)

A prompt-based multilingual model holds *all* its languages in one shared decoder/joint. Fine-tune it on only your target languages and that shared head re-specializes around them — the languages you *didn't* train on quietly collapse. This is catastrophic forgetting, and it happens **even with a frozen encoder**: the encoder may stay general, but the RNNT decoder/joint stop seeing the other languages' tokens and drift. The analogous Hebrew result from the parakeet recipe is the canonical warning: the target language improved sharply while unrelated languages degraded badly, and the cure was not "train harder" — it was reintroducing the other languages. Blending roughly 25–35% replay restored them *while keeping* the target-language gain.

**The fix: blend in multilingual replay.** Reserve ~25–35% of the training mix for a slice of the model's *other* languages — languages you are NOT primarily targeting — each tagged with its own true `target_lang`. The decoder/joint never stops emitting those token distributions, so they don't atrophy. Keep your target languages dominant (the remaining ~65–75%) so they still move.

The replay slice is just more tarred/Lhotse cuts with the correct per-utterance tag. FLEURS or Common Voice *train* splits for the preserved languages work well — small, clean, and broad. Each cut must carry its true language tag (e.g. a Spanish clip tagged `es`, not your target tags), or replay does nothing useful.

Wire it up as a weighted multi-manifest mix and let `manifest_filepath_weights` set the replay ratio:

```python
# build_replay_mix.py — combine target-language data with a replay slice
# Each entry is a tarred Lhotse manifest; every cut already carries its true target_lang.
target_manifests = [
    "/data/it/tarred_manifest.json",   # Italian   (target)
    "/data/ro/tarred_manifest.json",   # Romanian  (target)
    "/data/el/tarred_manifest.json",   # Greek     (target)
    "/data/bg/tarred_manifest.json",   # Bulgarian (target)
]
# Languages you are NOT targeting but must preserve — each tagged with ITS OWN target_lang.
replay_manifests = [
    "/data/replay/es/tarred_manifest.json",  # Spanish, tagged es
    "/data/replay/de/tarred_manifest.json",  # German,  tagged de
    "/data/replay/fr/tarred_manifest.json",  # French,  tagged fr
]

manifests = target_manifests + replay_manifests
# Target dominant (~70%), replay (~30%) split evenly within each group; weights are normalized by Lhotse.
weights = [0.7 / len(target_manifests)] * len(target_manifests) \
        + [0.3 / len(replay_manifests)] * len(replay_manifests)

print("++model.train_ds.manifest_filepath=[" + ",".join(manifests) + "]")
print("++model.train_ds.manifest_filepath_weights=[" + ",".join(f"{w:.4f}" for w in weights) + "]")
```

Pass the resulting list and weights as Hydra overrides on the Step 2 command:

```bash
++model.train_ds.manifest_filepath="[/data/it/...,/data/ro/...,/data/el/...,/data/bg/...,/data/replay/es/...,/data/replay/de/...,/data/replay/fr/...]" \
+model.train_ds.manifest_filepath_weights="[0.175,0.175,0.175,0.175,0.10,0.10,0.10]"
```

**Scope for this task.** All four of *your* targets (Italian, Romanian, Greek, Bulgarian) are trained together, so they reinforce one another and need no replay relative to each other — the table in Step 3 is what that buys you. Replay matters for the languages *outside* these four that you still need to serve from one checkpoint. If you only ever ship these four, you can skip replay entirely; if the base model's full language coverage must survive, replay is mandatory.


## Step 3 — Evaluate (honestly, at deploy-time streaming latency)

Evaluate on the **held-out FLEURS test** (a split the model never saw in training) and do it in the **exact streaming condition you will deploy at**: 0 ms look-ahead, `att_context_size=[56,0]`. This is the most demanding setting — no future-audio "peeking." Evaluating offline/full-context flatters the model versus how it actually runs in a low-latency gRPC stream, so those numbers would lie about production behavior.

### Run streaming inference at the deploy-time latency

Inference uses NeMo's cache-aware streaming script — it loads the `.nemo`, runs the model in **true streaming mode** at the `att_context_size` you pass, and writes per-utterance hypotheses (and a per-utterance WER) to `output_path`. Set `att_context_size="[56,0]"` for the 0 ms look-ahead condition you deploy. `att_context_size` is `[left, right]` in encoder frames; `right=0` means **0 ms look-ahead** — the model commits without waiting for future audio.

```bash
# conda activate nemo_main
NEMO_ROOT=/path/to/NeMo            # the forked NeMo checkout with nemotron-3.5-asr-streaming support
LHOTSE_DIR=/path/to/lhotse
export PYTHONPATH=$NEMO_ROOT:$LHOTSE_DIR:$PYTHONPATH

MODEL_PATH=/path/to/nemotron-3.5-asr-streaming-0.6b-ft.nemo
MANIFEST_PATH=/path/to/fleurs_it_test.json   # lines carry target_lang
OUTPUT_FOLDER=/path/to/results
mkdir -p ${OUTPUT_FOLDER}

python ${NEMO_ROOT}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=${MODEL_PATH} \
    dataset_manifest=${MANIFEST_PATH} \
    output_path=${OUTPUT_FOLDER} \
    target_lang=it-IT \
    att_context_size="[56,0]" \
    decoder_type=rnnt \
    pad_and_drop_preencoded=true \
    batch_size=256 \
    cuda=0 \
    strip_lang_tags=true
```

- `target_lang=it-IT` — transcribe with a **known** language. Use `target_lang=auto` to let the model **detect** the language per utterance.
- `att_context_size="[56,0]"` — 0 ms look-ahead, the lowest-latency / hardest condition. Use e.g. `"[56,3]"` for a little look-ahead (higher accuracy, higher latency).
- `strip_lang_tags=true` — drop the `<lang>` tag token from the emitted text so it does not pollute WER (with `false` a hypothesis looks like `… <es-US>`).
- The script writes `streaming_out_<model>_<manifest>.json` into `OUTPUT_FOLDER` — one JSON line per utterance with `text` (reference), `pred_text` (hypothesis), and `wer`.

### Known language vs auto language ID

Pass an explicit tag when you know the language (best accuracy):

```bash
    target_lang=es-ES \
    att_context_size="[56,3]" \
    strip_lang_tags=true
```

Or let the model detect it per utterance:

```bash
    target_lang=auto \
    att_context_size="[56,3]" \
    strip_lang_tags=true
```

### Report NORMALIZED WER — and state raw vs normalized

The script's built-in `wer` field is raw (case/punctuation-sensitive) and includes any leaked `<lang>` tag. For cross-team / leaderboard parity, recompute on the script's output with the Whisper `BasicTextNormalizer` (for non-English) + `jiwer`, stripping any leftover prompt tokens, and normalize **both** sides. Always benchmark the base and fine-tuned models under the **identical normalization** on the **identical test set** — otherwise the comparison is meaningless.

```python
import re, json, jiwer
from whisper_normalizer.basic import BasicTextNormalizer

norm = BasicTextNormalizer()
PROMPT_TOK = re.compile(r"<[a-z]{2}-[A-Z]{2}>")  # strip any leftover <it-IT>, <es-US>, ...

def clean(s):
    return norm(PROMPT_TOK.sub("", s))

# the per-utterance file written by speech_to_text_cache_aware_streaming_infer.py
preds = [json.loads(l) for l in open("results/streaming_out_..._fleurs_it_test.json")]
refs = [clean(p["text"]) for p in preds]
hyps = [clean(p["pred_text"]) for p in preds]
print(f"{100 * jiwer.wer(refs, hyps):.1f}")
```

> Report both numbers. The table below is **raw** WER; normalized WER will be lower, but the *relative* gap between base and fine-tuned holds as long as both are scored the same way.

### Results

Raw WER %, held-out FLEURS test, lowest-latency streaming (`att_context_size=[56,0]`, 0 ms look-ahead), the same eval applied to both models:

| Language  | Base model | Fine-tuned | Relative reduction |
| --------- | ---------- | ---------- | ------------------ |
| Italian   | 19.2       | 15.9       | -17%               |
| Romanian  | 75.2       | 35.3       | -53%               |
| Greek     | 72.4       | 47.8       | -34%               |
| Bulgarian | 84.8       | 32.2       | -62%               |

**Takeaway:** languages that were nearly unusable in the base model became genuinely useful after a short fine-tune. Bulgarian and Romanian error rates **more than halved** — at the same 0 ms-latency streaming setting you will actually deploy.

## Step 4 — Scale the Data Where It Helps

The 290 h mix proves the recipe works; the next question is how far more in-language data pushes it. To test the lever, mix in **~1,000 additional hours per language** of parliamentary speech (MOSEL / VoxPopuli), taking the training pool from **~290 h to ~4,300 h**. Even **partway through** that longer run the weakest languages keep improving — Bulgarian, for example, drops from 32.2 into the **high-20s WER** before the run even finishes.

The takeaway: **more in-language data keeps helping**, but the gains are **uneven across languages and domains**. Parliamentary speech is a domain shift relative to read/Common-Voice/FLEURS audio, so a thousand hours buys more for one language than another. Do not assume — **measure per language** on the held-out FLEURS eval after each data addition.

Let error analysis drive where the effort goes: spend new-data hours on whichever language the held-out FLEURS eval shows weakest, not uniformly. When you grow the pool:

- **Keep the four-way balance.** A 1,000 h dump into one language skews the prompt-conditioning distribution; rebalance with `manifest_filepath_weights`.
- **Re-tag every new source** with the correct `target_lang` (`<it-IT>`, `<ro-RO>`, `<el-GR>`, `<bg-BG>`) — a mistagged shard trains the wrong prompt branch.
- **Re-tar the larger pool** into Lhotse shards (`is_tarred=true`); do not mix tarred and untarred sources in one `train_ds`.
- **Recompute the step budget.** Lhotse is an `IterableDataset` with no `len()`, so `trainer.max_steps` does not auto-scale with the data. Recompute it for ~4,300 h and reset `trainer.val_check_interval` to an integer step count.

```bash
# train_ds now lists the original 290h shards + the parliamentary shards.
# manifest_filepath_weights re-balances so each language stays ~25% of the
# sampled stream despite the parliamentary hours being lopsided per language.
python speech_to_text_finetune.py \
  --config-path=<conf_dir> --config-name=<rnnt_streaming_cfg> \
  init_from_nemo_model=/ckpts/nemotron-3.5-asr-streaming-0.6b.nemo \
  model.train_ds.use_lhotse=true \
  model.train_ds.is_tarred=true \
  '++model.train_ds.manifest_filepath=[
     [/data/it/cv_290h/tarred_audio_manifest.json],
     [/data/ro/cv_290h/tarred_audio_manifest.json],
     [/data/el/cv_290h/tarred_audio_manifest.json],
     [/data/bg/cv_290h/tarred_audio_manifest.json],
     [/data/it/voxpopuli_1000h/tarred_audio_manifest.json],
     [/data/ro/voxpopuli_1000h/tarred_audio_manifest.json],
     [/data/el/voxpopuli_1000h/tarred_audio_manifest.json],
     [/data/bg/mosel_1000h/tarred_audio_manifest.json]
   ]' \
  '++model.train_ds.tarred_audio_filepaths=[
     [/data/it/cv_290h/audio__OP_0..N_CL_.tar],
     [/data/ro/cv_290h/audio__OP_0..N_CL_.tar],
     [/data/el/cv_290h/audio__OP_0..N_CL_.tar],
     [/data/bg/cv_290h/audio__OP_0..N_CL_.tar],
     [/data/it/voxpopuli_1000h/audio__OP_0..N_CL_.tar],
     [/data/ro/voxpopuli_1000h/audio__OP_0..N_CL_.tar],
     [/data/el/voxpopuli_1000h/audio__OP_0..N_CL_.tar],
     [/data/bg/mosel_1000h/audio__OP_0..N_CL_.tar]
   ]' \
  '++model.train_ds.manifest_filepath_weights=[
     0.20,0.20,0.20,0.20,   # original read/CV pools (keep as replay)
     0.05,0.05,0.05,0.05     # parliamentary hours; bias toward the weakest language
   ]' \
  model.train_ds.lang_field=target_lang \
  ++trainer.max_steps=<recomputed_for_4300h> \
  ++trainer.val_check_interval=<integer_steps> \
  ++trainer.use_distributed_sampler=false \
  trainer.precision=bf16-mixed
```

Keep the original 290 h shards in the mix as **replay** — a multilingual prompt model will forget the languages whose data you stop showing it. Tune the weights toward the weakest language, re-run the Step 3 eval under the same 0 ms-latency streaming setting and the same normalization scheme, and only then decide whether the next thousand hours is worth it.

## Step 5 — Export & Deploy

The fine-tuned model is the **same architecture** as the base — a cache-aware FastConformer-RNNT, prompt-based streaming model. There is no graph surgery, no new vocab, no exporter quirks: it drops straight into the path you already use for `nemotron-3.5-asr-streaming-0.6b`.

First, save the fine-tuned checkpoint:

```python
model.save_to("nemotron-3.5-asr-streaming-0.6b-ft.nemo")
```

### Option A — NVIDIA Riva (gRPC streaming)

```bash
# 1. Convert .nemo -> .riva
nemo2riva --out nemotron-asr-ft.riva \
          --format nemo \
          nemotron-3.5-asr-streaming-0.6b-ft.nemo

# 2. Build the streaming ASR pipeline
riva-build speech_recognition \
    /servicemaker-dev/nemotron-asr-ft.rmir \
    /servicemaker-dev/nemotron-asr-ft.riva \
    --name=nemotron-asr-streaming-ft \
    --decoder_type=rnnt \
    --streaming=true \
    --chunk_size=0.16 \
    --padding_size=1.92

# 3. Deploy to the model repository served by riva-server
riva-deploy /servicemaker-dev/nemotron-asr-ft.rmir /data/models
```

The streaming latency/accuracy operating point is selected via the cache-aware `att_context_size` (and the matching `chunk_size`/`padding_size` in `riva-build`) — exactly the same knobs as the base model. Per-clip language is still driven by the `target_lang` prompt tag at request time.

### Option B — NeMo inference (cache-aware streaming script)

For offline/batch inference with NeMo, drive the same cache-aware streaming script you used to evaluate (Step 3). It runs the model in true streaming mode at the `att_context_size` you choose and reads the per-utterance `target_lang` from the manifest:

```bash
python ${NEMO_ROOT}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=nemotron-3.5-asr-streaming-0.6b-ft.nemo \
    dataset_manifest=eval_it_manifest.json \
    output_path=results/ \
    target_lang=it-IT \
    att_context_size="[56,0]" \
    decoder_type=rnnt \
    pad_and_drop_preencoded=true \
    batch_size=256 cuda=0 \
    strip_lang_tags=true
```

Use `target_lang=auto` for per-utterance language detection. Hypotheses land in `results/streaming_out_*.json` (`text` / `pred_text` / `wer` per line).

### Picking the operating point

`att_context_size` is `[left_context, right_context]`. Right context is *future* audio: more of it lowers WER but adds algorithmic latency. Choose at inference time — one checkpoint serves every point on the curve.

| `att_context_size` | Look-ahead | Latency | Accuracy |
|--------------------|-----------|---------|----------|
| `[56, 0]`          | 0 ms      | Lowest  | Lowest (this is the eval/deploy setting reported above) |
| `[56, k]`, k>0     | > 0 ms    | Higher  | Higher   |
| full context       | offline   | Highest | Highest  |

The exact supported `[left, right]` values are model-specific — read the model card before assuming a setting is valid. Always benchmark at the *same* context you intend to deploy.

## Best Practices

1. **Match the base model's text style.** Train on punctuated, properly-cased (PnC) transcripts — that is what the model emits; mismatched casing/punctuation inflates WER and confuses the head.
2. **Tag every clip with a correct, recognized `target_lang`.** Prompt-based conditioning is only as good as the tag; use a value the model knows (e.g. `it-IT`), and verify the tag — not just the audio — is right.
3. **Use tarred Lhotse shards for IO.** `model.train_ds.use_lhotse=true`, `is_tarred=true`. No per-file unpacking; it streams efficiently at scale.
4. **Schedule by a FIXED STEP BUDGET, not epochs.** Lhotse is an `IterableDataset` with no `len()`, so epoch-based LR scheduling silently breaks. Set `trainer.max_steps` **and** the scheduler's `max_steps` to the same explicit value, make `val_check_interval` an integer step count, and set `++model.train_ds.use_distributed_sampler=false`.
5. **Recompute the step budget whenever data volume or GPU count changes.** Steps = (target token/hour exposure) ÷ (global batch). Adding the 4,300 h pool or moving from 1 to N GPUs changes effective throughput — re-derive `max_steps` rather than reusing the old number.
6. **Evaluate at the latency you will deploy.** Report WER at `att_context_size=[56,0]` (0 ms streaming) — the hardest, no-peeking condition — so the benchmark reflects production.
7. **Run inference via the cache-aware streaming script.** Use `speech_to_text_cache_aware_streaming_infer.py` with a `dataset_manifest` whose lines carry `target_lang`, and pass `target_lang=<tag>` (or `target_lang=auto`) plus `att_context_size`. Manifests need only the `target_lang` field — no separate `lang`/`prompt_mode`.
8. **Use `target_lang=auto` when the language is unknown.** The model detects language per utterance; pass an explicit tag (e.g. `es-US`) when you know it for best accuracy. Set `strip_lang_tags=true` so the `<lang>` token never reaches your WER.
9. **Report normalized WER under one scheme.** Apply the Whisper `BasicTextNormalizer` (for non-English) + `jiwer`, strip any leaked prompt tokens (e.g. `<it-IT>`), and benchmark **all** models (base and fine-tuned) under the *same* normalization — otherwise the comparison is meaningless.
10. **Replay to prevent catastrophic forgetting.** Multilingual models forget fast; keep a mix of all target languages in every step rather than fine-tuning one language at a time.
11. **More in-language data keeps helping — but measure, don't assume.** Going from ~290 h to ~4,300 h pushed the weakest languages further down (Bulgarian into the high-20s), yet gains are uneven across languages and domains. Validate each addition.
12. **Same architecture ⇒ same serving path.** The fine-tuned `.nemo` deploys through the identical Riva (`nemo2riva` → `riva-build` → `riva-deploy`) or NeMo inference flow as the base.
13. **Pick latency vs. accuracy at inference, not at training time.** One checkpoint covers the whole `att_context_size` curve — tune the operating point per deployment.

## References

- Prompt RNNT training script: `examples/asr/asr_transducer/speech_to_text_rnnt_bpe_prompt.py` (in the forked NeMo checkout with nemotron-3.5-asr-streaming support)
- Cache-aware streaming prompt config: `examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml`
- Cache-aware streaming **inference** script: `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
- Cache-aware streaming FastConformer docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#cache-aware-streaming-conformer
- Build tarred datasets: `scripts/speech_recognition/convert_to_tarred_audio_dataset.py` — https://github.com/NVIDIA/NeMo/blob/main/scripts/speech_recognition/convert_to_tarred_audio_dataset.py
- Lhotse + NeMo dataloading: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#lhotse-dataloading
- `nemo2riva` export tool: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/model-overview.html#nemo2riva
- Riva ASR streaming deployment (`riva-build`/`riva-deploy`): https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html
- Whisper `BasicTextNormalizer`: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py — and `jiwer`: https://github.com/jitsi/jiwer
- FLEURS dataset: https://huggingface.co/datasets/google/fleurs
- VoxPopuli (parliamentary speech): https://github.com/facebookresearch/voxpopuli — MOSEL: https://huggingface.co/datasets/FBK-MT/mosel
- Model card: see the NVIDIA org on Hugging Face at `huggingface.co/nvidia/<model-id>` for the canonical `att_context_size` values and recognized `target_lang` tags.
