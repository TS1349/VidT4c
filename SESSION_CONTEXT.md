# Vid4c Session Context

## 구현 완료 사항

### fusion 모드 (구현됨, 성능 하락으로 폐기)

`feat_router`, `entropy_gate` — GCN baseline보다 성능 낮음, 사용 안 함
- feat_router: 5 seed 중 2-3개 gate collapse → 불안정
- entropy_gate: Emognition -2pp, MDMER 동일 수준

### disagree_loss (2026-04-26 구현, v2 수정 완료)

`--disagree_loss gce` / `sym_ce` / `none` 플래그 (`runner.py`)
`--gce_q` 기본값 0.7

**동작 방식 (2-phase schedule):**
- epoch < 10: `combined += 0.5 * aux_CE(eeg_head, video_head)` — head calibration만
- epoch >= 10: `combined += 0.1 * aux + w_ce * delta`
  - `delta = ratio_disagree * (GCE - CE)` — CE를 GCE로 교체하는 delta (loss 총량 유지)
  - GCE ≤ CE이므로 delta ≤ 0 → disagree sample의 CE 부분만 완화, 전체 loss scale 불변

**v1 실패 원인 (수정됨):**
- 기존: `combined += GCE` → disagree sample에 CE+GCE 2배 loss → overfitting
- 수정: `combined += w_ce * delta` → CE를 GCE로 교체하는 correction만 적용

**Early stopping:** `--patience 20`, DDP broadcast, 해당 run만 종료 후 다음 run 진행

**변경 파일:** `models/vemt/vemt.py`, `trainer.py`, `runner.py`
- eeg_head/video_head: router/entropy_gate와 공유 (중복 생성 없음)

---

## 실험 결과 (기준값)

| Model | Emognition ACC | MDMER ACC |
|---|---|---|
| AdaMAE (video only) | 27.5% | 36.67% |
| CBraMod (EEG only) | 33.4% | 36.50% |
| GCN baseline | **33.86%** | **39%** |
| feat_router | 33.07% ± 0.029 | ~39% |
| entropy_gate | 31.70% ± 0.015 | ~39% |
| GCN + disagree_loss gce v1 | 실패 (overfitting) | 실패 (overfitting) |
| GCN + disagree_loss gce v2 | 실험 대기 | 실험 대기 |

---

## 실험 명령어 (v2 — 현재 실행할 것)

### MDMER (GPU 4,5,6,7)
```bash
cd /SSD4/jh/INRIA && CUDA_VISIBLE_DEVICES=4,5,6,7 python runner.py \
  --vemt_video AdaMAE --fusion naive --eeg_signal --gcn \
  --model vemt --dataset mdmer \
  --csv_file ./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv \
  --learning_rate 1e-4 --weight_decay 0.05 \
  --epochs 100 --num_gpus 4 --batch_size 4 --pretrained \
  --port 20025 --checkpoint_dir ./checkpoints_5 \
  --experiment_name gcn_disagree_gce_mdmer_v2 \
  --disagree_loss gce
```

### Emognition (GPU 0,1,2,3)
```bash
cd /SSD4/jh/INRIA && CUDA_VISIBLE_DEVICES=0,1,2,3 python runner.py \
  --vemt_video AdaMAE --fusion naive --eeg_signal --gcn \
  --model vemt --dataset emognition \
  --csv_file ./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv \
  --learning_rate 1e-4 --weight_decay 0.05 \
  --epochs 100 --num_gpus 4 --batch_size 4 --pretrained \
  --port 20026 --checkpoint_dir ./checkpoints_5 \
  --experiment_name gcn_disagree_gce_emognition_v2 \
  --disagree_loss gce
```

---

## 알려진 문제 / 검토 필요

1. **5-class disagree 비율 문제**: 5-class에서 agree 확률 ~20% → disagree mask가 대부분 sample에 적용. v2에서는 delta 방식이므로 비율이 높아도 loss scale이 유지됨
2. **confidence-gated disagree**: 논의했으나 보류 (결과 보고 결정)
3. **all_wrong ceiling**: label noise → 구조로 개선 한계

---

## 다음 단계 후보

1. **disagree_loss v2 결과 수집** (실험 예정)
2. confidence-gated disagree (결과에 따라 결정)
3. Cross-modal contrastive (agree sample만 InfoNCE)
4. Prototype-based inference 보정
