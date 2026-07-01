# EAV fold0 unimodal baselines — SLURM

Runs the **video-only** and **eeg-only** baselines (our GCN+temp+PE fusion = "ours"
is *excluded*) on the **EAV dataset, fold0**, with the same protocol as our
Emognition/MDMER unimodal baselines (lr 1e-4, wd 0.05, 100 epochs, patience 25,
pretrained). One SLURM job per model, single GPU each.

## Models (9 jobs)

| kind  | model         | file                          | backbone config (video) / route (eeg)           |
|-------|---------------|-------------------------------|--------------------------------------------------|
| video | AdaMAE        | `eav_video_adamae.slurm`      | fi=4 overlap=0.5, chunk 6, clip_pool attn        |
| video | VideoMAE      | `eav_video_videomae.slurm`    | fi=4 overlap=0.5, chunk 6, clip_pool attn        |
| video | ViViT         | `eav_video_vivit.slurm`       | fi=2, chunk 6, clip_pool attn                    |
| video | TSF           | `eav_video_tsf.slurm`         | fi=2, chunk 6, clip_pool attn                    |
| video | Swin          | `eav_video_swin.slurm`        | fi=2, chunk 3, clip_pool attn (batch 2)          |
| eeg   | CBraMod       | `eav_eeg_cbramod.slurm`       | `--model vemt --eeg_backbone cbramod`            |
| eeg   | REVE          | `eav_eeg_reve.slurm`          | `--model vemt --eeg_backbone reve`               |
| eeg   | LaBraM        | `eav_eeg_labram.slurm`        | `--model labram`                                 |
| eeg   | ST-Transformer| `eav_eeg_sttransformer.slurm` | `--model sttransformer`                          |

Video jobs use `--batch_size 4` (Swin 2; dense clips are memory-heavy), eeg jobs
`--batch_size 12`. All single-GPU (`--num_gpus 1`).

## One-time setup on the new server

```bash
# 1) build the conda env from the exported spec (repo root)
module purge && module load miniforge/24.9.0     # or your cluster's conda module
conda env create -f environment.yml              # creates env "tsf-low"
# (pip-only deps incl. torch 2.1.1+cu118 are in the file's pip: section)
# fallback if conda solve is slow/unavailable:
#   conda create -n tsf-low python=3.10 && pip install -r requirements.txt

# 2) data + pretrained weights must be present (gitignored, copy separately):
#    datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv
#    datasets/EAV/...            (extracted EAV video+eeg)
#    pretrained/                 (AdaMAE checkpoint-199.pth, VideoMAE *_v0.pth, etc.;
#                                 ViViT/TSF/Swin self-load their own pretrained)
```

## Submit

```bash
bash slurm/submit_all_eav.sh                 # all 9
bash slurm/submit_all_eav.sh eav_video_swin  # a subset (names w/ or w/o .slurm)
```
Logs: `NJZ_logs/eav/JZ_{OUTPUT,ERROR}_eav_<model>.out`.
Results: `checkpoints_clip/eav_<kind>_<model>_fold0_1e-4_0.05/run_0/` (best.txt etc.).

## Notes / things to check on the target cluster

- **`#SBATCH` header is the provided cluster template, unchanged** (account `vfj@h100`,
  `-C h100`, qos, etc.) — only `--job-name` / `--output` / `--error` differ per job.
- **`--time=01:00:00` is the template default and is almost certainly too short for the
  video backbones** (100 epochs × dense-clip feature extraction). Raise `--time` for the
  video jobs (eeg jobs are fast). Early stopping (patience 25) usually ends well before
  100 epochs.
- The module lines (`gcc/11.3.1`, `miniforge/24.9.0`) are from the template — adjust to
  your cluster's available modules if they differ.
- `--dense_cache_features` caches extracted clip features under `.feature_cache/` on
  first epoch; the cold first epoch is the slow one.

## Regenerate the sbatch files

`bash slurm/_gen_eav_slurm.sh` rewrites all 9 from the model table in the generator.
