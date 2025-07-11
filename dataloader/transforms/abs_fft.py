from torch.fft import fft
import torch

class AbsFFT(object):
    def __init__(self, dim=-2):
        self.dim = dim
    
    def __call__(self, sample):
        fft_trans = fft(sample, dim=self.dim)
        return fft_trans.abs()


class sftf(object):
    def __init__(self, n_fft=256, hop_length=128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

    def __call__(self, sample):
        spec = torch.stft(
            sample,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        # # Check either one
        # spec = torch.stft(
        #     sample,
        #     n_fft=128,
        #     hop_length=64,
        #     window=self.window,
        #     return_complex=True
        # )

        return spec.abs()