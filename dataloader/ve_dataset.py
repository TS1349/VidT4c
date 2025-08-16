import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import pandas as pd
import math
from .utils import torch_random_int
import numpy as np

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
    deterministic: True for val/test (구간 중앙), False for train (구간 내 랜덤)
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
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ignore_list = None,
            ):
        self.motion_sampler = motion_sampler
        self.csv_file = str(csv_file)
        self.split = split
        self.video_output_format = video_output_format

        self.num_out_frames = num_out_frames


        self.eeg_sampling_rate = eeg_sampling_rate
        self.num_out_eeg = num_out_eeg


        self.video_transform = video_transform
        self.eeg_transform = eeg_transform

        self.output_shape = output_shape
        self.eeg_channel_count = eeg_channel_count

        df = pd.read_csv(self.csv_file)
        self.df = df[(df["data_split"] == self.split) & (df["bool_both_file"] == True)]

        if ignore_list is not None:
            regex_string = f"subject{\
                str(ignore_list).replace(" ", "").replace(",", "|")}"
            self.df = self.df[~df.facial_video.str.contains(regex_string, regex = True)]

    def __len__(self):
        return len(self.df)

    def _get_label(self, row):
        # returns 3 int32 class ids starting from 0
        self_annotation = row.self_annotation[1:-1].split(r",")
        self_annotation = [ int(entry) - 1 for entry in self_annotation ]

        # bin mapping: 0-1→0, 2-3→1, 4→2, 5-6→3, 7-8→4
        mapping = {
            0: 0, 1: 0,
            2: 1, 3: 1,
            4: 2,
            5: 3, 6: 3,
            7: 4, 8: 4
        }
        va_labels = [mapping[val] for val in self_annotation[:2]]  # Only valence, arousal
        return torch.tensor(va_labels, dtype=torch.int64)
    
    def _get_full_eeg(self, row):
        eeg = pd.read_csv(row.EEG) # Sample x channel
        eeg = torch.tensor(eeg.to_numpy(), dtype = torch.float32)
        #primarly for Emognition, but makes sense to run it for all of them
        eeg = torch.nan_to_num(eeg)
        return eeg

    @staticmethod
    def _decimate_idxs(start_idx, end_idx, final_number):
        delta = end_idx - start_idx
        return [ round(start_idx + i * delta / (final_number -1)) for i in range(final_number) ]
    

    def _get_uniform_frames(self, fps, video, eeg):
        eeg_sample_count = eeg.shape[0]

        video_time = video.shape[0] / fps 
        eeg_time = eeg_sample_count / self.eeg_sampling_rate

        if (eeg_time > video_time):
            eeg_sample_count = round(video_time * self.eeg_sampling_rate)

        #take multiple of video frames
        eeg_sample_step = eeg_sample_count // self.num_out_frames
        eeg_upper_idx = self.num_out_frames * eeg_sample_step + 1

        video_idxs =[
            round((i + 0.5)*eeg_sample_step / self.eeg_sampling_rate*fps) for i in range(self.num_out_frames)]

        return (video[video_idxs,...], eeg[:eeg_upper_idx, ...])
    
    def _get_sampler_frame_idxs(self, total_frames, fps, video_tensor):

        assert video_tensor.shape[0] == total_frames

        scores = _motion_scores_from_video(video_tensor)           # [T]
        cdf = _cdf_from_scores(scores)                             # [T] in [0,1]

        # Data split
        deterministic = (self.split.lower() in ["test"])

        # CDF based sector sampling
        idxs = _pick_indices_by_cdf(cdf, self.num_out_frames, deterministic)  # [num_out_frames]
        return idxs.tolist()
    
    def _get_random_frame_idxs(self, total_frames, fps):
        idxs = VERandomDataset._decimate_idxs(
            start_idx = 5,
            end_idx = total_frames - 5,
            final_number=self.num_out_frames)        

        return idxs


    def _get_corresponding_eeg_idxs(self, frame_idxs, fps):
        
        start_eeg_idx = math.floor(frame_idxs[0] * self.eeg_sampling_rate / fps)
        end_eeg_idx = math.floor(frame_idxs[-1] * self.eeg_sampling_rate / fps)

        eeg_idxs = list(range(start_eeg_idx, end_eeg_idx + 1))

        return eeg_idxs


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row.facial_video

        # I think it should throw an exception if something goes wrong here also I've moved the filter to __init__
        video_orig, _, metadata = read_video(
            filename=video_path,
            pts_unit="sec",
            output_format=self.video_output_format,
        )

        # If you have error with new EAV dataset, you can use below code
        # if any(f"EAV/subject{sid}" in video_path for sid in [4, 5, 18, 20, 38]):
        #     print("error missing file existing")
        #     return self.__getitem__((idx + 1) % len(self.df))   

        #try:
        #    video_orig, _, metadata = read_video(
        #        filename=video_path,
        #        pts_unit="sec",
        #        output_format=self.video_output_format,
        #    )
        #except (RuntimeError, FileNotFoundError) as e:
        #    print(f"Skipping file {video_path} because: {e}")
        #    return self.__getitem__((idx + 1) % len(self.df))
    
        
        fps = metadata["video_fps"]
        eeg = self._get_full_eeg(row)


        if self.motion_sampler:
            eeg_time_to_frames = math.floor(math.floor((eeg.shape[0] / self.eeg_sampling_rate)) * fps) # Seconds x fps for EEG frame
            total_frames = min(video_orig.shape[0], eeg_time_to_frames) # Based on frame
            video_orig = video_orig[:total_frames, ...]

            video_idxs = self._get_sampler_frame_idxs(
                total_frames=total_frames,
                fps=fps,
                video_tensor=video_orig,
            )
            video = video_orig[video_idxs, ...]
            eeg_idxs = self._get_corresponding_eeg_idxs(video_idxs, fps)
            eeg = eeg[eeg_idxs,...]
        else:
            video, eeg = self._get_uniform_frames(fps, video_orig, eeg)

        if self.video_transform is not None:
            video = torch.stack([self.video_transform(frame) for frame in video])
        
        if self.eeg_transform is not None:
            eeg = self.eeg_transform(eeg)

        # final outputs: (start with 0)
        output = self._get_label(row)

        return {"video" : video,
                "eeg" : eeg,
                "output" : output,
               }


class EAVDataset(VERandomDataset):
    def __init__(
            self,
            motion_sampler,
            csv_file,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):

        super(EAVDataset,self).__init__(
                motion_sampler=motion_sampler,
                csv_file = csv_file,
                eeg_sampling_rate = 500,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
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
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):

        super(MDMERDataset,self).__init__(
                motion_sampler=motion_sampler,
                csv_file = csv_file,
                eeg_sampling_rate = 300,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (5,2),
                eeg_channel_count = 18,
        )

class EmognitionDataset(VERandomDataset):
    def __init__(
            self,
            motion_sampler,
            csv_file,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):

        super(EmognitionDataset,self).__init__(
                motion_sampler=motion_sampler,
                csv_file = csv_file,
                eeg_sampling_rate = 256,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (5,2),
                eeg_channel_count = 4,
        )
