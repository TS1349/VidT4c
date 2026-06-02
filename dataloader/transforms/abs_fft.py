from torch.fft import fft
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchaudio.functional as AF
import numpy as np
import scipy.signal

class MedformerEEGPreprocess:
    """
    Medformer-style EEG preprocessing
    Input : (T, C) or (C, T) torch.float
    Steps : bandpass(0.5-45) -> resample(->256Hz) -> z-score -> 1s segmentation
    Output:
      - out='tc'  : (seg_len, C)
      - out='ntc' : (N_seg, seg_len, C)
      - out='ct'  : (C, seg_len)
    """

    def __init__(
        self,
        sfreq_in: int,
        sfreq_out: int = 256,
        band=(0.5, 45.0),
        seg_len: int = 256,
        seg_overlap: float = 0.5,
        out: str = "tc",            # 'tc' | 'ntc' | 'ct'
        zero_phase: bool = False,
        eps: float = 1e-6,
    ):
        self.sfreq_in  = int(sfreq_in)
        self.sfreq_out = int(sfreq_out)
        self.low_hz, self.high_hz = float(band[0]), float(band[1])
        self.seg_len = int(seg_len)
        self.hop = max(1, int(round(seg_len * (1.0 - float(seg_overlap)))))
        self.out = out
        self.zero_phase = bool(zero_phase)
        self.eps = float(eps)

    # ---- utils ----
    @staticmethod
    def _to_ct(x: torch.Tensor):
        # (T, C) -> (C, T), (C, T) -> (C, T)
        if x.ndim != 2:
            raise AssertionError(f"Expected 2D tensor, got {tuple(x.shape)}")
        T, C = x.shape[0], x.shape[1]
        if T >= C: 
            return x.transpose(0, 1).contiguous(), "tc"
        else:
            return x.contiguous(), "ct"

    @staticmethod
    def _back_to(x_ct: torch.Tensor, orig: str, out: str):
        if out == "ct":
            return x_ct
        if out == "tc":
            return x_ct.transpose(0, 1).contiguous()
        return x_ct

    # core ops
    def _biquad_fb(self, x: torch.Tensor, fn, *args, **kwargs):
        y = fn(x, *args, **kwargs)
        if not self.zero_phase:
            return y
        y = torch.flip(y, dims=[-1])
        y = fn(y, *args, **kwargs)
        y = torch.flip(y, dims=[-1])
        return y

    def _bandpass(self, x_ct: torch.Tensor):
        if self.low_hz  is not None and self.low_hz  > 0:
            x_ct = self._biquad_fb(x_ct, AF.highpass_biquad, self.sfreq_in, cutoff_freq=self.low_hz)
        if self.high_hz is not None and self.high_hz > 0:
            x_ct = self._biquad_fb(x_ct, AF.lowpass_biquad,  self.sfreq_in, cutoff_freq=self.high_hz)
        return x_ct

    def _resample(self, x_ct: torch.Tensor):
        if self.sfreq_in == self.sfreq_out:
            return x_ct
        return AF.resample(x_ct, self.sfreq_in, self.sfreq_out)

    def _zscore(self, x_ct: torch.Tensor):
        mean = x_ct.mean(dim=-1, keepdim=True)
        std  = x_ct.std(dim=-1, keepdim=True)
        return (x_ct - mean) / (std + self.eps)

    def _segment_ct(self, x_ct: torch.Tensor):
        # (C, T) -> (N, C, seg_len)
        C, T = x_ct.shape
        if T < self.seg_len:
            pad = self.seg_len - T
            left, right = pad // 2, pad - pad // 2
            x_ct = F.pad(x_ct, (left, right))
            T = x_ct.shape[-1]
        tail = (-(T - self.seg_len) % self.hop) % self.hop
        if tail:
            x_ct = F.pad(x_ct, (0, tail))
            T = x_ct.shape[-1]
        # unfold last-dim
        win = x_ct.unfold(dimension=-1, size=self.seg_len, step=self.hop)  # (C, N, seg_len)
        win = win.permute(1, 0, 2).contiguous()                             # (N, C, seg_len)
        return win

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.to(torch.float32)

        x_ct, orig = self._to_ct(x) # (C, T)
        x_ct = self._bandpass(x_ct)
        x_ct = self._resample(x_ct) # (C, T_out)
        x_ct = self._zscore(x_ct)

        segs = self._segment_ct(x_ct) # (N, C, seg_len)

        if self.out == "ntc":
            # (N, seg_len, C)
            return segs.permute(0, 2, 1).contiguous().to(device)

        x_ct = segs.mean(dim=0)
        return self._back_to(x_ct, orig, self.out).to(device)


class AbsFFT(object):
    def __init__(self, dim=-2):
        self.dim = dim
    
    def __call__(self, sample):
        fft_trans = fft(sample, dim=self.dim)
        return fft_trans.abs()


class STFT(object):
    # def __init__(self, n_fft=127, hop_length=16, visualize=False):
    def __init__(self, n_fft=127, visualize=False):
        self.n_fft = n_fft
        self.visualize = visualize
        self.window = torch.hann_window(n_fft)

    def __call__(self, sample):
        """
        sample shape: (num_frames, num_time, num_channels)
        output shape: (1, num_channels*2, freq_bins, time_steps)
        """
        # Flatten over time
        x = sample.transpose(1, 0)
        num_channels, _ = x.shape

        # STFT
        stfted = torch.stft(
            x,
            n_fft=self.n_fft,
            # hop_length=self.hop_length, # Based on the torch documentation, hop_length = n_fft / 4 (overlap ratio: 75%)
            window=self.window,
            return_complex=True
        )  # (num_channels, freq_bins, time_steps)

        # Do not use phase at EEG singal
        magnitude = torch.log1p(torch.abs(stfted))

        # phase = torch.angle(stfted)
        # mag_phase = torch.stack([magnitude, phase], dim=1)  # (C, 2, F, T)
        # mag_phase = mag_phase.view(num_channels * 2, magnitude.shape[2], magnitude.shape[1])  # (2*C, T, F)
        mag_phase = magnitude.transpose(2, 1)

        if self.visualize:
            for ch in range(num_channels):
                mag_phase_up = F.interpolate(mag_phase.unsqueeze(0), size=(mag_phase.shape[1]*2, mag_phase.shape[2]*2), mode='bilinear', align_corners=False)[0]
                self._plot_channel(x, mag_phase[ch * 2], mag_phase_up[ch * 2], ch)


        return mag_phase

    def _plot_channel(self, signal_all, mag_before, mag_after, ch):
        signal = signal_all[ch].detach().cpu().numpy()
        spectrogram_before = mag_before.cpu().numpy()
        spectrogram_after = mag_after.cpu().numpy()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Raw signal
        axes[0].plot(signal)
        axes[0].set_title(f"Raw EEG Signal (Channel {ch})")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Amplitude")

        # Spectrogram before interpolation
        im1 = axes[1].imshow(spectrogram_before, aspect='auto', origin='lower', cmap='magma')
        axes[1].set_title(f"Spectrogram (Before Interpolation, Channel {ch})")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Frequency bin")
        fig.colorbar(im1, ax=axes[1], orientation='vertical')

        # Spectrogram after interpolation
        im2 = axes[2].imshow(spectrogram_after, aspect='auto', origin='lower', cmap='magma')
        axes[2].set_title(f"Spectrogram (After Interpolation x2, Channel {ch}) [Larger]")
        axes[2].set_xlabel("Time step")
        axes[2].set_ylabel("Frequency bin")
        fig.colorbar(im2, ax=axes[2], orientation='vertical')

        fig.tight_layout()

        path = f"/home/jakim/research/emotion/VidT4c-main/STFT_plot/FFT_interp_ch{ch}.png"
        fig.savefig(path)
        print(f"[✓] Saved: {path}")
        plt.close(fig)


class STFTFixedSize(STFT):
    def __init__(self, n_fft=127, target_freq=128, target_time=256, visualize=False):
        super().__init__(n_fft=n_fft, visualize=visualize)
        self.target_freq = target_freq
        self.target_time = target_time

    def __call__(self, sample):
        mag_phase = super().__call__(sample)  # (C, T, F)
        mag_phase_up = F.interpolate(
            mag_phase.unsqueeze(0), 
            size=(self.target_time, self.target_freq), 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        return mag_phase_up

class CBraModTransform:
    def __init__(
        self,
        target_len=2000,     # Overall time length
        patch_size=200,      # CBraMod in_dim
    ):
        self.target_len = target_len
        self.patch_size = patch_size
        assert target_len % patch_size == 0, "target_len must be divisible by patch_size"
        self.patch_num = target_len // patch_size

    def __call__(self, eeg):
        """
        eeg: torch.Tensor, shape [S, Ch]
        return: torch.Tensor, shape [Ch, patch_num, patch_size]
        """

        if isinstance(eeg, torch.Tensor):
            eeg_np = eeg.detach().cpu().numpy().copy()
        else:
            eeg_np = eeg

        # resample
        eeg_np = scipy.signal.resample(eeg_np, self.target_len, axis=0)
        eeg = torch.from_numpy(eeg_np.copy()).float()

        # (S, Ch) → (Ch, T)
        eeg = eeg.permute(1, 0)

        # Patch slicing
        eeg = eeg.view(eeg.shape[0], self.patch_num, self.patch_size)
        eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-6)

        return eeg


class CBraModStratifiedTransform:
    """
    Stratified patch sampling for CBraMod input.
    Splits the per-sample EEG slice into `patch_num` equal subwindows along time,
    then extracts a 1-second native-rate window from each subwindow (random in train,
    center in val). Each 1s patch is resampled to `patch_size`.

    Benefits vs. global resampling:
      - Each patch stays near the native sampling rate that CBraMod was trained on
        (no severe OOD downsampling for long clips).
      - Patches span the full clip duration → temporal alignment with the 32 video
        frames (also uniformly sampled across the clip).

    Falls back to global resampling when subwindow_len < 1 second worth of samples.
    Output shape: [Ch, patch_num, patch_size] (identical to CBraModTransform).
    """
    def __init__(
        self,
        sample_rate,
        target_len=2000,
        patch_size=200,
        train=True,
    ):
        self.sample_rate = int(sample_rate)
        self.target_len = target_len
        self.patch_size = patch_size
        assert target_len % patch_size == 0, "target_len must be divisible by patch_size"
        self.patch_num = target_len // patch_size  # e.g., 10
        self.native_patch_len = self.sample_rate    # 1 second of native samples
        self.train = train

    def _global_resample(self, eeg_np):
        eeg_np = scipy.signal.resample(eeg_np, self.target_len, axis=0)
        eeg = torch.from_numpy(eeg_np.copy()).float()
        eeg = eeg.permute(1, 0).contiguous()
        eeg = eeg.view(eeg.shape[0], self.patch_num, self.patch_size)
        return eeg

    def __call__(self, eeg):
        """
        eeg: torch.Tensor [S, Ch] — native EEG samples for this clip
        return: torch.Tensor [Ch, patch_num, patch_size]
        """
        if isinstance(eeg, torch.Tensor):
            eeg_np = eeg.detach().cpu().numpy().copy()
        else:
            eeg_np = np.asarray(eeg)

        S = eeg_np.shape[0]
        subwindow_len = S // self.patch_num

        # Fallback: clip too short to fit a 1s native patch per subwindow
        if subwindow_len < self.native_patch_len:
            eeg = self._global_resample(eeg_np)
            eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-6)
            return eeg

        avail = subwindow_len - self.native_patch_len  # >= 0
        patches = []
        for i in range(self.patch_num):
            sub_start = i * subwindow_len
            if self.train and avail > 0:
                offset = np.random.randint(0, avail + 1)
            else:
                offset = avail // 2
            p_start = sub_start + offset
            patch = eeg_np[p_start : p_start + self.native_patch_len]  # [native_patch_len, Ch]
            patch = scipy.signal.resample(patch, self.patch_size, axis=0)
            patches.append(patch)

        eeg = np.stack(patches, axis=0)  # [patch_num, patch_size, Ch]
        eeg = torch.from_numpy(eeg.copy()).float()
        eeg = eeg.permute(2, 0, 1).contiguous()  # [Ch, patch_num, patch_size]
        # mean = eeg.mean(dim=(1, 2), keepdim=True)   # [Ch, 1, 1]
        # std  = eeg.std(dim=(1, 2), keepdim=True)    # [Ch, 1, 1]
        # eeg = (eeg - mean) / (std + 1e-6) 
        eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-6)
        return eeg
