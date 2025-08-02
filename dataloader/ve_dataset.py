import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import pandas as pd
import math
from .utils import torch_random_int

class VERandomDataset(Dataset):
    def __init__(
            self,
            csv_file,
            eeg_sampling_rate,
            eeg_channel_count,
            output_shape,
            time_window,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):
        
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

        self.output_shape = output_shape
        self.eeg_channel_count = eeg_channel_count

        df = pd.read_csv(self.csv_file)
        self.df = df[(df["data_split"] == self.split) & (df["bool_both_file"] == True)]
        sample_row = self.df.iloc[0]
        anno_type = sample_row.anno_type

        if anno_type == "category":
            self._get_label = self._get_ctgr_labels
        else:
            self._get_label = self._get_cont_labels


    def __len__(self):
        return len(self.df)

    def _get_label(self, *args):
        raise NotImplementedError("This fuction shouldn't have been accessed")

    def _get_cont_labels(self, row):
        # returns 3 int32 class ids starting from 0
        self_annotation = row.self_annotation[1:-1].split(r",")
        self_annotation = [ int(entry) - 1 for entry in self_annotation ]
        return torch.tensor(self_annotation, dtype = torch.int64)

    def _get_ctgr_labels(self, row):
        return torch.tensor(row.label_id, dtype = torch.int64)
    
    def _get_full_eeg(self, row):
        eeg = pd.read_csv(row.EEG) # Sample x channel
        eeg = torch.tensor(eeg.to_numpy(), dtype = torch.float32)
        return eeg
    
    @staticmethod
    def _decimate_idxs(start_idx, end_idx, final_number):
        delta = end_idx - start_idx
        return [ round(start_idx + i * delta / (final_number -1)) for i in range(final_number) ]
    
    def _get_random_frame_idxs(self, total_frames, fps):
        num_frames_in_window = math.floor(fps * self.time_window)

        lower_bound = math.ceil((num_frames_in_window / (self.num_out_frames - 1)))
        upper_bound = total_frames - num_frames_in_window - lower_bound - 1 # since num_frames_in_window is floored

        random_start_idx = torch_random_int(
            low = lower_bound,
            high = upper_bound, # ! low + 32 x 6.4
            )
        
        idxs = VERandomDataset._decimate_idxs(
            start_idx = random_start_idx,
            end_idx = random_start_idx + num_frames_in_window,
            final_number=self.num_out_frames)

        return idxs
    
    def _get_corresponding_eeg_idxs(self, frame_idxs, fps):
        num_samples_sub_window = round(self.time_sub_window * self.eeg_sampling_rate)

        eeg_idxs = [ math.floor((idx  * self.eeg_sampling_rate / fps)) - num_samples_sub_window // 2
                    for idx in frame_idxs ]
        eeg_idxs.append(eeg_idxs[-1] + num_samples_sub_window)

        decimated_eeg_idxs_list = [ VERandomDataset._decimate_idxs(
            start_idx = eeg_idxs[i],
            end_idx = eeg_idxs[i+1],
            final_number=self.num_out_eeg
            ) for i in range(len(eeg_idxs)-1) ]

        return decimated_eeg_idxs_list

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row.facial_video

        # Skip missing file compared with csv (after face crop)
        # The 'EAV/subject4, 5, 18, 20, 38' folder dosen't exist now
        if any(f"EAV/subject{sid}" in video_path for sid in [4, 5, 18, 20, 38]):
            return self.__getitem__((idx + 1) % len(self.df))

        try:
            video, _, metadata = read_video(
                filename=video_path,
                pts_unit="sec",
                output_format=self.video_output_format,
            )
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Skipping file {video_path} because: {e}")
            return self.__getitem__((idx + 1) % len(self.df))
        
        fps = metadata["video_fps"]

        eeg = self._get_full_eeg(row)
        eeg_time_to_frames = math.floor(math.floor((eeg.shape[0] / self.eeg_sampling_rate)) * fps) # Seconds x fps for EEG frame

        total_frames = min(video.shape[0], eeg_time_to_frames) # Based on frame

        video_idxs = self._get_random_frame_idxs(total_frames, fps)
        video = video[video_idxs, ...]


        if self.video_transform is not None:
            # video = self.video_transform(video)
            video = torch.stack([self.video_transform(frame) for frame in video]) # Torchvision Compose function is for 'B x C x H x W'
        
        # eeg part:
        eeg_idxs = self._get_corresponding_eeg_idxs(video_idxs, fps)
        eeg = eeg[eeg_idxs,...]

        if "Emognition" in self.csv_file: # To handle nan value of the eeg on the Emognition dataset.
            eeg = torch.nan_to_num(eeg)
        
        if self.eeg_transform is not None:
            eeg = self.eeg_transform(eeg)

        # final outputs: (start with 0)
        output = self._get_label(row)
        #print(video_idxs)
        #print(eeg_idxs)

        #mean_time_video = [ idx / fps for idx in video_idxs]
        #mean_time_eeg = [ (idxs[0] + idxs[-1]) / 2 / self.eeg_sampling_rate for idxs in eeg_idxs ]
        #print(mean_time_video)
        #print(mean_time_eeg)
        
        #time_diff = [ abs(mean_time_eeg[i] - mean_time_video[i]) for i in range(self.num_out_frames)]
        #print(f"{max(time_diff) / self.time_sub_window * 100}%")

        return {"video" : video,
                "eeg" : eeg,
                "output" : output,
               }


class EAVDataset(VERandomDataset):
    def __init__(
            self,
            csv_file,
            time_window = 15.0,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):

        super(EAVDataset,self).__init__(
                csv_file = csv_file,
                eeg_sampling_rate = 500,
                time_window = time_window,
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
            csv_file,
            time_window = 15.0,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):

        super(MDMERDataset,self).__init__(
                csv_file = csv_file,
                eeg_sampling_rate = 300,
                time_window = time_window,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (9,3),
                eeg_channel_count = 18,
        )

class EmognitionDataset(VERandomDataset):
    def __init__(
            self,
            csv_file,
            time_window = 30.0,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ):

        super(EmognitionDataset,self).__init__(
                csv_file = csv_file,
                eeg_sampling_rate = 256,
                time_window = time_window,
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (9,3),
                eeg_channel_count = 4,
        )
