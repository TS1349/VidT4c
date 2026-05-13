# VEMT — Video-EEG Multi-clip Training

원래 `full length coverage` 로 돌리던 거랑 비교해서 dataloader / training policy / GCN 새로 정리한 브랜치. 무엇이 바뀌었고 왜 그렇게 했는지 간단히 정리해두었습니다.

## 데이터 측면

### 기존 (`full length coverage`) 문제

- **Video**: 32 frame을 비디오 전체 시간축에 uniform decimation. stride가 동적으로 늘어나서 AdaMAE / ViViT 같은 short-clip pretrained model의 native 분포에서 벗어남. 표정의 순간적인 변화를 포착하기 어려움
- **EEG**: 비디오 첫 frame ~ 마지막 frame에 해당하는 EEG 연속 슬라이스를 CBraMod의 `target_len=2000` 으로 강제 resample. Emognition 120s @ 256Hz 면 ~15× 다운샘플, Hz 정보 뭉개짐.

→ 양쪽 backbone이 사전학습 분포에서 어긋난 채 동작. 그래서 GCN 같은 fusion 얹어도 single modality 대비 의미 있는 향상이 안 나옴.

### 새 sampling 옵션 (dataloader)

| flag | 의미 |
|---|---|
| `--num_clips K` | K개 클립을 backbone-native stride로 추출 |
| `--frame_interval fi` | 클립 한 개 stride (총 길이 = `32 × fi / fps` 초) |
| `--clip_pool {mean,max,attn}` | K개 클립 feature/logit 합치는 방식 |
| `--fps_normalize` | 30/60fps 섞인 데이터셋(Emognition)에서 `fi × fps/30` 로 wall-clock 통일 |

이제 video는 short clip K개를 native stride로 받고, EEG도 비슷하게 처리 가능.

## 고려는 했지만 benchmark에선 안 쓰는 옵션

### `--train_random_crop`
Train만 K=1 random clip, val/test는 K-clip 그대로. Video에는 augmentation처럼 작동해서 single-modality video는 약간 잘 나오긴 함.

근데 fusion에 쓰면:
- Train graph가 K=1로 degenerate, val은 K-clip → **inter-clip / cross-modal edge가 train에서 학습 안 됨**
- backbone auto-freeze 조건도 따로 처리해야 됨

→ benchmark setup에선 안 씀. Video ablation 따로 볼 때만 옵션.

### `--eeg_full_signal`
EEG를 비디오 전체 span의 1개 윈도우로. Standalone EEG-only에선 짧은 random window의 label noise를 피하니까 잘 나옴 (ACC 0.335 vs per-clip 0.283).

근데:
- 120s → 2000 samples 다운샘플은 CBraMod 사전학습 분포(10s @ native fs)에서 OOD
- Fusion에서는 EEG가 video clip과 시간 align 안 됨 → graph가 cross-modal temporal correspondence를 학습할 수 없음
- 결국 GCN이 fusion gain 못 만들고 `num_clips=1` 때 문제 재현
- Benchmark 자체는 성능을 높이는 것보다, 원래 backbone이 기대하는 setup과 동일하게 가져가는것이 맞다고 판단

→ benchmark setup에서 쓰지 않는것으로 정리. EEG-only ablation 비교용으로만 켜봤음.

## 현재 Emognition dataset setup

| 선택 | 이유 |
|---|---|
| K=6, fi=10 (Emognition) | 클립 ~10.7s → CBraMod 사전학습 분포 친화 + K=6 × 10.7s = 64s 로 median 120s (Emognition) 의 절반 커버 |
| `--fps_normalize` | 30/60fps wall-clock 일치 |
| `--train_random_crop` 빼기 | Train/val graph 위상 동일 유지 |
| `--eeg_full_signal` 빼기 | EEG도 K-clip coupled → backbone-native 분포 + graph가 cross-modal 시간 align 학습 가능 |
| `--gcn_video_per_clip --gcn_eeg_per_clip` | K개 video / K개 EEG를 각각 graph 노드로 노출. modality-aware adjacency 로 `video[i] ↔ EEG[i]` 시간 paired |

### MDMER, EAV dataset은 long term이 아니기 때문에, 큰 문제가 되지 않음
Ex) K=4, fi= 4 or 10


## Backbone freeze

K-clip은 forward 를 K번 해야 해서 메모리 부담 큼. 기본적으로 backbone auto-freeze 발동함 (`--train_random_crop` or `--no_kclip_freeze` 안 줬을 때).

### Backbone full-freeze 안쓰는 이유

- VideoMAE / AdaMAE 는 Kinetics 같은 action recognition data 로 pretrained 되어 있어서 학습된 representation 도 motion / action pattern 위주. Emotion recognition 에서 중요한 facial expression / micro-expression cue 와는 결이 다름. Backbone 통째로 freeze 하고 head 만 학습시켜보면 실제로 학습이 거의 안 됨. Backbone 이 emotion 신호를 안 들고 있으니 head 가 뽑을 게 없음.
- CBraMod 도 비슷한 맥락. Masked EEG reconstruction 으로 학습된 generic EEG representation 이라 emotion task 에 곧장 쓰기엔 representation 적응이 필요함.
- 그래서 양쪽 다 적어도 마지막 N block 정도는 풀어서 task 에 맞추는 게 필요.

Arg flags:
- `--video_unfreeze_last_n_blocks N` : video 마지막 N block + head trainable
- `--eeg_unfreeze_last_n_blocks N` : CBraMod `encoder.layers[-N:]` + classifier
- `--eeg_full_unfreeze` : CBraMod 전체 (encoder + patch_embedding + classifier). CBraMod이 22.5M 으로 작아서 K-clip 환경에서도 풀 fine-tune 가능
- `--no_kclip_freeze` : auto-freeze 자체 끄기

## Running code

### Video baseline (set_video_only)
```bash
python runner.py --vemt_video AdaMAE --fusion naive --set_video_only \
  --model vemt --dataset emognition --csv_file <fold0.csv> \
  --learning_rate 1e-4 --weight_decay 0.05 \
  --epochs 100 --num_gpus 4 --batch_size 3 --pretrained \
  --num_clips 6 --frame_interval 16 --clip_pool attn --fps_normalize \
  --video_unfreeze_last_n_blocks 1
```
(ACC: 0.274, UAR: 0.226, F1-w 0.171)
(--clip_pool max -> ACC: 0.269, UAR: 0.226, F1-w: 0.177)

### EEG baseline (set_eeg_only, CBraMod 전체 fine-tune)
```bash
python runner.py --fusion naive --set_eeg_only \
  --model vemt --dataset emognition --csv_file <fold0.csv> \
  --learning_rate 1e-4 --weight_decay 5e-2 \
  --epochs 100 --num_gpus 1 --batch_size 12 --pretrained \
  --num_clips 6 --frame_interval 10 --clip_pool mean --fps_normalize \
  --eeg_full_unfreeze
```
(ACC: 0.283, UAR: 0.235, F1-w: 0.177)

### GCN Fusion (main setup ★)
```bash
python runner.py --vemt_video AdaMAE --fusion naive --eeg_signal --gcn \
  --model vemt --dataset emognition --csv_file <fold0.csv> \
  --learning_rate 3e-5 --weight_decay 0.1 --gcn_learning_rate 5e-5 \
  --epochs 100 --num_gpus 4 --batch_size 12 --pretrained \
  --num_clips 6 --frame_interval 10 --clip_pool attn --fps_normalize \
  --video_unfreeze_last_n_blocks 1 --eeg_full_unfreeze
```

(Run_0 -> ACC: 0.313, UAR: 0.233, F1-w: 0.242, Running (Run_1,2,3,4))

- VEMT 내의 ViViT, AdaMAE, Cbramod code 중심으로 체크 (본래 benchmark code인 각 model.py 들은 아직 변경전)
- 예전에 쓰던 slurm인 old_slurm은 현재 구현으로 돌아가지 않습니다. (srun으로 위 code run)
- 현재 main.py는 총 5번의 run 이후 mean, std이 자동 계산 됩니다.
- Backbone ckpt download (https://drive.google.com/drive/folders/1f44ETZWN6MN_ARuE2TeGbaf8fcMTScLf?usp=sharing) -> ./pretrained

## New method

### GCN Fusion 기본 setup (pooled, 4-node graph)

위 main setup 은 K개 video clip / K개 EEG clip 을 각각 K-pool (mean / attn) 해서 single video global / single EEG global 노드 한 개씩 만들고, brain region 노드들과 함께 GCN graph 구성하는 방식. 노드 4 ~ 6개짜리 작은 그래프.

- 장점: 안정적이고 baseline 잘 잡힘
- 단점: K개 클립의 temporal 정보가 K-pool 에서 다 사라짐. video 의 시간 별 emotion 변화 / EEG 시간 dynamics 가 graph 에 직접 들어가지 못함

### Per-clip nodes + modality-aware adjacency (현재 실험 중)

`--gcn_video_per_clip --gcn_eeg_per_clip` 옵션. K개 video / K개 EEG 가 각각 별도 노드로 graph 에 들어감 (총 2 * K-clip + region 노드).

**핵심 idea**: 두 modality 의 i번째 clip 은 같은 시간 윈도우를 보고 있으니, graph adjacency 에 이 **시간 prior 를 직접 encode** 해보자는 방향.

현재 구현된 modality-aware adjacency:
- `video[i] ↔ video[j]` : fully connected (video 내부 temporal interaction)
- `EEG[i] ↔ EEG[j]` : fully connected (EEG 내부 temporal interaction)
- `video[i] ↔ EEG[i]` : paired (같은 시간 clip prior, **strict time-align**)
- `video[i] ↔ EEG[j], i≠j` : 직접 edge X (2-hop 으론 도달 가능)
- 모든 clip ↔ region, region ↔ region

→ 같은 시간 구간의 video–EEG 만 직접 edge 로 연결하는 strict time-align 구성.

### 다음 방향: weak time-align (고려중)

Strict time-align 은 prior 가 너무 강할 수 있음. Emotion 에서 video facial expression 과 EEG neural response 는 정확히 같은 시점에 일어나는 게 아닐 수 있음 (neural response delay, 표정과 내부 감정 상태 mismatch 등).

**weak time-align** 으로 풀어보는 옵션 고려 중:
- `video[i] ↔ EEG[j]` 모든 cross edge 살리되, `|i - j|` 거리에 따라 weight 감쇠 (Gaussian kernel 등)
- 또는 paired edge 에만 학습 가능한 강한 prior 두고, 나머지 cross edge 는 약한 baseline 으로

아직 구현 전. Strict 버전 결과 보고 어느 정도 prior 가 적당한지 판단할 예정. 두 방향 다 확정된 건 아니고, 체크중인 단계.
