import torch
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(
            self,
            num_out_frames = 32,
            image_size = 224,
            num_eeg_channels = 8,
            num_out_eeg = 64,
            ):
           self.number_of_frames =  num_out_frames
           self.image_size = image_size
           self.number_of_eeg_channels = num_eeg_channels
           self.number_of_eeg_samples = num_out_eeg 


    def __len__(self):
        return 128

    def __getitem__(self, idx):

        video = torch.randn(self.number_of_frames, 3, self.image_size, self.image_size)
        eeg = torch.randn(self.number_of_eeg_samples, self.number_of_eeg_channels)
        output = torch.tensor(0, dtype = torch.float32)

        return {"video" : video,
                "eeg" : eeg,
                "output" : output,
               }
