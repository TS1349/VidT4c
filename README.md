# VidT4c — Video + EEG Multimodal Emotion Recognition

Dense face-video clips and EEG are treated as nodes of a GCN and fused at the
clip level. Each video clip becomes a node (with sinusoidal positional encoding
and a learnable Gaussian temporal-decay adjacency), EEG contributes a pooled
node plus per-region nodes, and a single graph is used for valence/arousal
prediction.

Datasets: Emognition (fold0), MDMER (fold1), EAV (fold0). Fold splits are under
`datasets/updated_fold_csv_files/`.

## Setup

```bash
conda env create -f environment.yml   # env name: tsf-low
conda activate tsf-low
```

Pretrained video/EEG backbones go under `pretrained/`, raw data under
`datasets/` (only the fold CSVs are tracked in git).

## Running

Our fusion (GCN, clip + temporal + PE) — AdaMAE + CBraMod on Emognition:

```bash
python runner.py --model vemt --fusion naive --eeg_signal --eeg_backbone cbramod \
  --vemt_video AdaMAE --dataset emognition \
  --csv_file datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv \
  --dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --fps_normalize \
  --dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn \
  --eeg_full_unfreeze --eeg_full_signal \
  --gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe \
  --learning_rate 1e-4 --gcn_learning_rate 1e-4 --weight_decay 0.05 \
  --epochs 100 --patience 25 --num_gpus 4 --batch_size 3 --pretrained \
  --experiment_name emognition_adamae_cbramod
```

Unimodal baselines: add `--set_video_only` or `--set_eeg_only` (drop `--gcn`).
MDMER runs additionally pass `--grad_clip 0.5`.

The unimodal baselines can be run directly from `slurm/` — one sbatch file per
run:

```bash
sbatch slurm/eav_video_videomae.slurm     # a single baseline
bash   slurm/submit_all_eav.sh            # submit all EAV baselines
```

EEG-only runs on 1 GPU (batch 12), video-only on 3 GPUs × batch 4 (effective
batch 12, DDP). Regenerate the sbatch files with `bash slurm/_gen_eav_slurm.sh`.

## Key arguments

Video / dense-clip:

| arg | type | default | meaning |
|---|---|---|---|
| `--vemt_video` | str | AdaMAE | video backbone (AdaMAE/VideoMAE/ViViT/TSF/Swin) |
| `--dense_video_clips` | flag | off | use K dense clips per sample (required for GCN) |
| `--frame_interval` | int | 2 | frame stride within a clip |
| `--clip_overlap_ratio` | float | 0.0 | overlap between consecutive clips |
| `--dense_chunk_size` | int | 6 | frames per clip |
| `--fps_normalize` | flag | off | normalize clip sampling to a common fps |
| `--dense_cache_features` | flag | off | cache frozen backbone features to disk |
| `--video_unfreeze_last_n_blocks` | int | 0 | fine-tune the last N video blocks |
| `--clip_pool` | str | mean | clip pooling (mean/attn) |

Our GCN fusion:

| arg | type | default | meaning |
|---|---|---|---|
| `--gcn` | flag | off | enable GCN fusion |
| `--gcn_video_per_clip` | flag | off | one graph node per video clip |
| `--gcn_temporal_adj` | flag | off | Gaussian temporal-decay adjacency (learnable σ) |
| `--gcn_clip_pe` | flag | off | sinusoidal positional encoding on clip nodes |
| `--gcn_region_eeg_source` | str | stft | EEG feature for region nodes (stft/cbramod) |
| `--gcn_region_eeg_only` | flag | off | connect region nodes to EEG only (cut video↔region) |
| `--relresfuse` | flag | off | reliability-guided residual fusion (unimodal aux heads + supervised per-sample gate) |
| `--relresfuse_res` | float | 0.0 | initial (learnable) residual scale |
| `--relresfuse_kl` | float | 0.3 | gate KL-supervision weight |
| `--relresfuse_uni` | float | 0.5 | unimodal CE weight |
| `--relresfuse_tau` | float | 0.5 | reliability-target temperature |
| `--adaptive_gate` | flag | off | sample-adaptive fusion gate |

EEG:

| arg | type | default | meaning |
|---|---|---|---|
| `--eeg_signal` | flag | off | use the EEG modality |
| `--eeg_backbone` | str | cbramod | EEG backbone (cbramod/reve/labram/sttransformer) |
| `--eeg_full_unfreeze` | flag | off | fine-tune the whole EEG backbone |
| `--eeg_full_signal` | flag | off | feed the full EEG signal (no cropping) |

Training / DDP:

| arg | type | default | meaning |
|---|---|---|---|
| `--fusion` | str | naive | fusion head |
| `--learning_rate` / `--gcn_learning_rate` | float | 1e-4 | backbone / GCN learning rate |
| `--weight_decay` | float | 0.05 | weight decay |
| `--epochs` / `--patience` | int | 50 / 30 | max epochs / early-stopping patience |
| `--num_gpus` | int | 1 | GPUs; DDP is spawned internally per process |
| `--batch_size` | int | 2 | per-GPU batch size |
| `--grad_clip` | float | 0.0 | gradient clipping (0.5 for MDMER) |
| `--set_video_only` / `--set_eeg_only` | flag | off | unimodal baselines |

## Layout

```
runner.py            entry point (argparse, DDP spawn, multi-run loop)
trainer.py           training / eval loop
models/vemt/vemt.py  VEMT model: dense-clip GCN, region nodes, fusion
models/EEGs/         EEG backbones (CBraMod, LaBraM, ST-Transformer, ...)
models/eav, milmer   benchmark reproductions
dataloader/          datasets + transforms
slurm/               sbatch generators and files
```
