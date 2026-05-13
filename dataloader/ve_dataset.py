import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import pandas as pd
import math
from .utils import torch_random_int
import numpy as np
from collections import Counter
import ast

def _motion_scores_from_video(video_TCHW: torch.Tensor) -> np.ndarray:
    """
    video_TCHW: [T, C, H, W], float tensor assumed in [0,1] or [0,255]
    return: np.ndarray of shape [T], per-frame motion score (non-negative)
    """
    v = video_TCHW.float()
    diffs = (v[1:] - v[:-1]).abs().mean(dim=(1,2,3))  # [T-1]

    scores = torch.cat([torch.zeros(1, device=v.device), diffs], dim=0)  # [T]
    scores = scores.clamp_min(0).sqrt()
    scores = scores.cpu().numpy()

    if scores.sum() <= 1e-8:
        scores = np.ones_like(scores, dtype=np.float32)
    return scores

def _cdf_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float64)
    scores /= scores.sum()
    cdf = np.cumsum(scores)
    cdf = np.clip(cdf, 0.0, 1.0)
    return cdf

def _pick_indices_by_cdf(cdf: np.ndarray, num_out_frames: int, deterministic: bool) -> np.ndarray:
    """
    cdf: shape [T], monotonically increasing in [0,1]
    num_out_frames: e.g., 32
    deterministic: True for val/test, False for train
    return: np.ndarray of length num_out_frames (0-based frame indices, sorted)
    """
    T = len(cdf)
    targets = []
    for i in range(num_out_frames):
        lo, hi = i / num_out_frames, (i + 1) / num_out_frames
        if deterministic:
            t = (lo + hi) * 0.5
        else:
            t = np.random.uniform(lo, hi)
        targets.append(t)

    # Nearest cdf position as frame idx
    cdf_np = cdf
    idxs = []
    for t in targets:
        j = int(np.abs(cdf_np - t).argmin())
        idxs.append(j)

    # Remove duplication
    idxs = np.array(idxs, dtype=np.int64)
    idxs = np.clip(idxs, 0, T-1)

    used = set()
    for k in range(len(idxs)):
        if idxs[k] not in used:
            used.add(int(idxs[k]))
            continue
        # If collapse, find side idxs
        left = idxs[k] - 1
        right = idxs[k] + 1
        moved = False
        while left >= 0 or right < T:
            if left >= 0 and left not in used:
                idxs[k] = left; used.add(int(left)); moved = True; break
            if right < T and right not in used:
                idxs[k] = right; used.add(int(right)); moved = True; break
            left -= 1; right += 1
        if not moved:
            used.add(int(idxs[k]))
    idxs.sort()
    return idxs


class VERandomDataset(Dataset):
    def __init__(
        self,
        motion_sampler,
        csv_file,
        eeg_sampling_rate,
        eeg_channel_count,
        output_shape,
        time_window,
        split="train",
        video_output_format="TCHW",
        video_transform=None,
        eeg_transform=None,
        eeg_transform_local=None,
            num_out_frames=32,
        num_out_eeg=64,
        compute_weights=True,
        cb_beta=0.999,          # class-balanced weight(beta)
        weight_clip=(0.25, 4.0),
        num_clips=1,            # K-clip multi-clip inference (1 = original behaviour)
        frame_interval=2,       # backbone-native stride for K-clip mode (frames per output step)
        eeg_full_signal=False,  # if True with num_clips>1: video=K clips, EEG=single full-view window
        train_random_crop=False, # train: one random clip at native stride; val/test: K-clip
        fps_normalize=False,    # if True: scale frame_interval by fps/30 so 30/60fps videos cover the same wall-clock per clip
    ):
        self.motion_sampler = motion_sampler
        self.num_clips = max(1, int(num_clips))
        self.frame_interval = max(1, int(frame_interval))
        self.eeg_full_signal = bool(eeg_full_signal)
        self.train_random_crop = bool(train_random_crop)
        self.fps_normalize = bool(fps_normalize)
        self.csv_file = str(csv_file)
        self.split = split
        self.video_output_format = video_output_format

        self.time_window = time_window
        self.num_out_frames = num_out_frames
        self.time_sub_window = time_window / (num_out_frames - 1)

        self.eeg_sampling_rate = eeg_sampling_rate
        self.num_out_eeg = num_out_eeg

        self.video_transform = video_transform
        self.eeg_transform = eeg_transform
        self.eeg_transform_local = eeg_transform_local

        self.output_shape = output_shape
        self.eeg_channel_count = eeg_channel_count

        df = pd.read_csv(self.csv_file)
        self.df = df[(df["data_split"] == self.split) & (df["bool_both_file"] == True)].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No samples after filtering in split='{self.split}'")

        sample_row = self.df.iloc[0]
        anno_type = sample_row.anno_type

        if anno_type == "category":
            self.anno_mode = "category"
            self._get_label = self._get_ctgr_labels
        else:
            self.anno_mode = "va"  # valence/arousal
            self._get_label = self._get_cont_labels

        self.valence_weights = None
        self.arousal_weights = None
        self.class_stats = None

        if (self.anno_mode == "va") and (self.split.lower() in ["train"]) and compute_weights:
            v_labels, a_labels = self._collect_va_5cls_labels(self.df)
            v_counts = self._counts(v_labels, num_classes=5)
            a_counts = self._counts(a_labels, num_classes=5)

            val_w = self._class_balanced_weights(v_counts, beta=cb_beta)  # [5]
            aro_w = self._class_balanced_weights(a_counts, beta=cb_beta)  # [5]

            val_w = val_w / (val_w.mean() + 1e-12)
            aro_w = aro_w / (aro_w.mean() + 1e-12)
            if weight_clip is not None:
                lo, hi = weight_clip
                val_w = np.clip(val_w, lo, hi)
                aro_w = np.clip(aro_w, lo, hi)

            self.valence_weights = torch.tensor(val_w, dtype=torch.float32)
            self.arousal_weights = torch.tensor(aro_w, dtype=torch.float32)

            self.class_stats = {
                "valence_counts": {i: int(v_counts[i]) for i in range(5)},
                "arousal_counts": {i: int(a_counts[i]) for i in range(5)},
                "valence_ratios": {i: float(v_counts[i] / max(1, v_counts.sum())) for i in range(5)},
                "arousal_ratios": {i: float(a_counts[i] / max(1, a_counts.sum())) for i in range(5)},
            }

    def __len__(self):
        return len(self.df)

    def _get_label(self, *args):
        raise NotImplementedError("This function shouldn't have been accessed")

    def _get_cont_labels(self, row):
        self_annotation = row.self_annotation[1:-1].split(r",")
        self_annotation = [int(entry) - 1 for entry in self_annotation]

        # bin mapping: 0-1→0, 2-3→1, 4→2, 5-6→3, 7-8→4
        mapping = {0:0, 1:0, 2:1, 3:1, 4:2, 5:3, 6:3, 7:4, 8:4}
        va_labels = [mapping[self_annotation[0]], mapping[self_annotation[1]]]  # V, A
        return torch.tensor(va_labels, dtype=torch.int64)

    def _get_ctgr_labels(self, row):
        return torch.tensor(row.label_id, dtype=torch.int64)

    def _get_full_eeg(self, row):
        eeg = pd.read_csv(row.EEG)  # shape: [Samples, Channels]
        eeg = torch.tensor(eeg.to_numpy(), dtype=torch.float32)
        return eeg

    @staticmethod
    def _decimate_idxs(start_idx, end_idx, final_number):
        delta = end_idx - start_idx
        return [round(start_idx + i * delta / (final_number - 1)) for i in range(final_number)]

    def _get_sampler_frame_idxs(self, total_frames, fps, video_tensor=None):
        assert video_tensor is not None and video_tensor.shape[0] == total_frames
        scores = _motion_scores_from_video(video_tensor)  # [T]
        cdf = _cdf_from_scores(scores)                   # [T] in [0,1]
        deterministic = (self.split.lower() in ["test", "val", "validation"])
        idxs = _pick_indices_by_cdf(cdf, self.num_out_frames, deterministic)
        return idxs.tolist()

    def _get_random_frame_idxs(self, total_frames, fps):
        start = max(0, 1)
        end = max(start + 1, total_frames - 1)
        idxs = VERandomDataset._decimate_idxs(
            start_idx=start,
            end_idx=end,
            final_number=self.num_out_frames
        )
        return idxs

    def _get_k_clip_frame_idxs(self, total_frames, K, frame_interval=2):
        """K clips at backbone's native temporal stride.

        Each clip = `num_out_frames` frames at `frame_interval` native stride
        (≈ num_out_frames * frame_interval / fps seconds per clip; e.g. 32*2/30 ≈ 2.13s).
        Clip centers are uniformly distributed across the full video, with
        ±half-segment random jitter for the train split (deterministic at val/test).
        Clips intentionally do NOT cover the whole video — they are representative
        windows sampled at the backbone's pretrained recipe.
        """
        span = self.num_out_frames * frame_interval        # native frames per clip
        half = span // 2

        min_c = half
        max_c = total_frames - 1 - half

        # Video too short for native stride: fall back to single-stretched clip × K
        if max_c <= min_c:
            idxs = self._decimate_idxs(0, total_frames - 1, self.num_out_frames)
            return [idxs] * K

        # Uniform centers
        if K == 1:
            centers = [(min_c + max_c) // 2]
        else:
            step = (max_c - min_c) / (K - 1)
            centers = [int(round(min_c + i * step)) for i in range(K)]

        # Random jitter for training (deterministic at val/test)
        deterministic = (self.split.lower() in ("test", "val", "validation"))
        if not deterministic and K >= 1:
            seg_half = max(1, int((max_c - min_c) / max(1, K) / 2))
            jittered = []
            for c in centers:
                lo = max(min_c, c - seg_half)
                hi = min(max_c, c + seg_half)
                if hi <= lo:
                    jittered.append(c)
                else:
                    jittered.append(int(np.random.randint(lo, hi + 1)))
            centers = jittered

        clips = []
        for c in centers:
            start = c - half
            idxs = [start + i * frame_interval for i in range(self.num_out_frames)]
            idxs = [max(0, min(total_frames - 1, i)) for i in idxs]
            clips.append(idxs)
        return clips

    def _get_random_clip_idxs(self, total_frames, frame_interval):
        """One random-position clip at the backbone's native stride.

        Span = num_out_frames * frame_interval native frames. Random start in
        [0, total_frames - span]. If the video is shorter than the span,
        falls back to a stretched clip over the entire video — that way the
        sample is still usable for very short videos.

        Used at training time when --train_random_crop is set: each epoch
        sees a different random window of the same video, which doubles as
        natural temporal augmentation.
        """
        span = self.num_out_frames * frame_interval
        if total_frames <= span:
            return self._decimate_idxs(0, total_frames - 1, self.num_out_frames)
        start = int(np.random.randint(0, total_frames - span + 1))
        return [start + i * frame_interval for i in range(self.num_out_frames)]

    def _effective_fi(self, fps):
        """frame_interval normalized to 30fps reference if fps_normalize is on.
        Keeps wall-clock clip duration consistent across 30/60fps videos
        (e.g. fi=12 stays 12.8s on 30fps and becomes 24-stride/12.8s on 60fps).
        """
        if not self.fps_normalize:
            return self.frame_interval
        return max(1, int(round(self.frame_interval * float(fps) / 30.0)))

    def _get_corresponding_eeg_idxs(self, frame_idxs, fps):
        start_eeg_idx = math.floor(frame_idxs[0] * self.eeg_sampling_rate / fps)
        end_eeg_idx = math.floor(frame_idxs[-1] * self.eeg_sampling_rate / fps)
        eeg_idxs = list(range(start_eeg_idx, end_eeg_idx + 1))
        return eeg_idxs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row.facial_video

        # if any(f"EAV/subject{sid}" in video_path for sid in [4, 5, 18, 20, 38]):
        #     return self.__getitem__((idx + 1) % len(self.df))
        try:
            video_orig, _, metadata = read_video(
                filename=video_path,
                pts_unit="sec",
                output_format=self.video_output_format,
            )
        except (RuntimeError, FileNotFoundError) as e:
            # print(f"Skipping file {video_path} because: {e}")
            return self.__getitem__((idx + 1) % len(self.df))

        fps = metadata["video_fps"]

        eeg = self._get_full_eeg(row)

        eeg_time_to_frames = math.floor(math.floor((eeg.shape[0] / self.eeg_sampling_rate)) * fps)
        total_frames = min(video_orig.shape[0], eeg_time_to_frames)

        if total_frames < self.num_out_frames:
            return self.__getitem__((idx + 1) % len(self.df))

        # Train-time random temporal crop: single random clip at native stride,
        # regardless of num_clips. Val/test keep the configured K-clip behaviour
        _use_train_random_crop = (
            self.train_random_crop
            and self.split.lower() == "train"
        )

        if _use_train_random_crop:
            video_idxs = self._get_random_clip_idxs(total_frames, self._effective_fi(fps))
            video = video_orig[video_idxs, ...]
            if self.video_transform is not None:
                video = self.video_transform(video)
            # EEG decoupled from the tiny random video clip: CBraMod prefers
            # ~10s-scale windows close to its pretraining distribution. With
            # eeg_full_signal=True (default), use the whole-video span so
            # train and val see the same EEG distribution (val K-clip path
            # uses the same full-view rule below). Set eeg_full_signal=False
            # to fall back to legacy clip-aligned EEG.
            if self.eeg_full_signal:
                eeg_video_idxs = self._get_random_frame_idxs(total_frames, fps)
            else:
                eeg_video_idxs = video_idxs
            eeg_idxs = self._get_corresponding_eeg_idxs(eeg_video_idxs, fps)
            eeg = eeg[eeg_idxs, ...]
            if "Emognition" in self.csv_file:
                eeg = torch.nan_to_num(eeg)
            if self.eeg_transform is not None:
                eeg_local = self.eeg_transform_local(eeg).contiguous()
                eeg = self.eeg_transform(eeg).contiguous()

        elif self.num_clips > 1:
            # K uniform clips for video. EEG: K aligned slices (default) or single full-view (eeg_full_signal).
            all_idxs = self._get_k_clip_frame_idxs(
                total_frames, self.num_clips, frame_interval=self._effective_fi(fps)
            )
            videos = []
            for video_idxs in all_idxs:
                v = video_orig[video_idxs, ...]
                if self.video_transform is not None:
                    v = self.video_transform(v)
                videos.append(v)
            video = torch.stack(videos)        # [K, T, C, H, W]

            if self.eeg_full_signal:
                # EEG: single full-view window (same logic as K=1 path)
                eeg_video_idxs = self._get_random_frame_idxs(total_frames, fps)
                eeg_idxs = self._get_corresponding_eeg_idxs(eeg_video_idxs, fps)
                eeg = eeg[eeg_idxs, ...]
                if "Emognition" in self.csv_file:
                    eeg = torch.nan_to_num(eeg)
                if self.eeg_transform is not None:
                    eeg_local = self.eeg_transform_local(eeg).contiguous()
                    eeg = self.eeg_transform(eeg).contiguous()
                # eeg, eeg_local stay 3D/4D (no K dim)
            else:
                eegs, eeg_locals = [], []
                for video_idxs in all_idxs:
                    e_idxs = self._get_corresponding_eeg_idxs(video_idxs, fps)
                    e = eeg[e_idxs, ...]
                    if "Emognition" in self.csv_file:
                        e = torch.nan_to_num(e)
                    if self.eeg_transform is not None:
                        el = self.eeg_transform_local(e).contiguous()
                        e = self.eeg_transform(e).contiguous()
                    eegs.append(e)
                    eeg_locals.append(el)
                eeg = torch.stack(eegs)            # [K, ...]
                eeg_local = torch.stack(eeg_locals)
        else:
            if self.motion_sampler:
                video_slice = video_orig[:total_frames, ...]
                video_idxs = self._get_sampler_frame_idxs(
                    total_frames=total_frames,
                    fps=fps,
                    video_tensor=video_slice,
                )
            else:
                video_idxs = self._get_random_frame_idxs(total_frames, fps)

            video = video_orig[video_idxs, ...]

            if self.video_transform is not None:
                # Single call on [T, C, H, W] keeps the flip/jitter decision
                video = self.video_transform(video)

            eeg_idxs = self._get_corresponding_eeg_idxs(video_idxs, fps)
            eeg = eeg[eeg_idxs, ...]

            if "Emognition" in self.csv_file:
                eeg = torch.nan_to_num(eeg)

            if self.eeg_transform is not None:
                eeg_local = self.eeg_transform_local(eeg).contiguous()
                eeg = self.eeg_transform(eeg).contiguous()

        output = self._get_label(row)

        return {
            "video": video,      # [T,C,H,W] (K=1) or [K,T,C,H,W] (K>1)
            "eeg": eeg,          # [S,Ch] (K=1) or [K,S,Ch] (K>1)
            "eeg_local": eeg_local,
            "output": output
        }

    @staticmethod
    def _map_to_5cls(x: int):
        """
        0~8 scale -> 5-bin
        0-1→0, 2-3→1, 4→2, 5-6→3, 7-8→4
        """
        if x <= 1: return 0
        if 2 <= x <= 3: return 1
        if x == 4: return 2
        if 5 <= x <= 6: return 3
        if 7 <= x <= 8: return 4
        # safety
        return max(0, min(4, x))

    def _collect_va_5cls_labels(self, df_split):
        v_list, a_list = [], []
        for _, row in df_split.iterrows():
            try:
                arr = ast.literal_eval(row.self_annotation)
                v_raw = int(arr[0]) - 1
                a_raw = int(arr[1]) - 1
                v_list.append(self._map_to_5cls(v_raw))
                a_list.append(self._map_to_5cls(a_raw))
            except Exception:
                continue
        return np.array(v_list, dtype=np.int64), np.array(a_list, dtype=np.int64)

    @staticmethod
    def _counts(labels: np.ndarray, num_classes=5):
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        return counts

    @staticmethod
    def _class_balanced_weights(counts: np.ndarray, beta=0.999):
        """
        Cui et al., CVPR'19 Class-Balanced Loss
        w_c = (1 - beta) / (1 - beta^{n_c})
        """
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / (eff_num + 1e-12)
        return w

class EAVDataset(VERandomDataset):
    def __init__(
            self,
            motion_sampler,
            csv_file,
            time_window = 15.0,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            eeg_transform_local=None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):

        super(EAVDataset,self).__init__(
                motion_sampler=motion_sampler,
                csv_file = csv_file,
                eeg_sampling_rate = 500,
                time_window = time_window,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                eeg_transform_local=eeg_transform_local,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (5,1),
                eeg_channel_count = 30,
        )

class MDMERDataset(VERandomDataset):
    def __init__(
            self,
            motion_sampler,
            csv_file,
            time_window = 15.0,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            eeg_transform_local=None,
            num_out_frames = 32,
            num_out_eeg = 64,
            num_clips = 1,
            frame_interval = 2,
            eeg_full_signal = False,
            train_random_crop = False,
            fps_normalize = False,
            ):

        super(MDMERDataset,self).__init__(
                motion_sampler=motion_sampler,
                csv_file = csv_file,
                eeg_sampling_rate = 300,
                time_window = time_window,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                eeg_transform_local=eeg_transform_local,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (5,2),
                eeg_channel_count = 18,
                num_clips = num_clips,
                frame_interval = frame_interval,
                eeg_full_signal = eeg_full_signal,
                train_random_crop = train_random_crop,
                fps_normalize = fps_normalize,
        )

class EmognitionDataset(VERandomDataset):
    def __init__(
            self,
            motion_sampler,
            csv_file,
            time_window = 30.0,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            eeg_transform_local = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            num_clips = 1,
            frame_interval = 2,
            eeg_full_signal = False,
            train_random_crop = False,
            fps_normalize = False,
            ):

        super(EmognitionDataset,self).__init__(
                motion_sampler=motion_sampler,
                csv_file = csv_file,
                eeg_sampling_rate = 256,
                time_window = time_window,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                eeg_transform_local=eeg_transform_local,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (5,2),
                eeg_channel_count = 4,
                num_clips = num_clips,
                frame_interval = frame_interval,
                eeg_full_signal = eeg_full_signal,
                train_random_crop = train_random_crop,
                fps_normalize = fps_normalize,
        )
