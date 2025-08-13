import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import pandas as pd
import math
from .utils import torch_random_int

class VEUniformDataset(Dataset):
    def __init__(
            self,
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
    

    def _get_unform_frame(self, fps, video, eeg):
        eeg_sample_count = eeg.shape[0]

        video_time = video.shape[0] / fps 
        eeg_time = eeg_sample_count / self.eeg_sampling_rate

        if (eeg_time > video_time):
            eeg_sample_count = round(video_time * self.eeg_sampling_rate)

        #take multiple of video frames
        eeg_sample_step = eeg_sample_count // self.num_out_frames
        eeg_upper_idx = self.num_out_frames * eeg_sample_step + 1

        video_idxs =[
            round((i + 0.5)*eeg_sample_step / self.eeg_sample_rate*fps) for i in range(self.num_out_frames)]

        return (video[video_idxs,...], eeg[:eeg_upper_idx, ...])

    
    def _get_corresponding_eeg_idxs(self, frame_idxs, fps):
        
        start_eeg_idx = math.floor(frame_idxs[0] * self.eeg_sampling_rate / fps)
        end_eeg_idx = math.floor(frame_idxs[-1] * self.eeg_sampling_rate / fps)

        eeg_idxs = list(range(start_eeg_idx, end_eeg_idx + 1))

        return eeg_idxs


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row.facial_video

        video, _, metadata = read_video(
            filename=video_path,
            pts_unit="sec",
            output_format=self.video_output_format,
            )
        
        fps = metadata["video_fps"]
        eeg = self._get_full_eeg(row)

        video,eeg = self._get_uniform_frames(fps, video, eeg)

        if self.video_transform is not None:
            # video = self.video_transform(video)
            video = torch.stack([self.video_transform(frame) for frame in video])
        
        if self.eeg_transform is not None:
            eeg = self.eeg_transform(eeg)

        # final outputs: (start with 0)
        output = self._get_label(row)

        return {"video" : video,
                "eeg" : eeg,
                "output" : output,
               }


class EAVDataset(VEUniformDataset):
    def __init__(
            self,
            csv_file,
            split="train",
            video_output_format = "TCHW",
            video_transform = None,
            eeg_transform = None,
            num_out_frames = 32,
            num_out_eeg = 64,
            ignore_list = [4, 5, 18, 20, 38],
            ):

        super(EAVDataset,self).__init__(
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
                ignore_list = ignore_list,
        )

class MDMERDataset(VEUniformDataset):
    def __init__(
            self,
            csv_file,
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
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (5,2),
                eeg_channel_count = 18,
        )

class EmognitionDataset(VEUniformDataset):
    def __init__(
            self,
            csv_file,
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
                split=split,
                video_output_format = video_output_format,
                video_transform = video_transform,
                eeg_transform = eeg_transform,
                num_out_frames = num_out_frames,
                num_out_eeg = num_out_eeg,
                output_shape = (5,2),
                eeg_channel_count = 4,
        )
