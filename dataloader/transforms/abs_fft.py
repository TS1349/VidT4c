from torch.fft import fft

class AbsFFT(object):
    def __init__(self, dim=-2):
        self.dim = dim
    
    def __call__(self, sample):
        sample = fft(sample, dim=self.dim)
        return sample.abs()