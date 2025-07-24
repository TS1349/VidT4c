from torch.fft import fft
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class AbsFFT(object):
    def __init__(self, dim=-2):
        self.dim = dim
    
    def __call__(self, sample):
        fft_trans = fft(sample, dim=self.dim)
        return fft_trans.abs()


class STFT(object):
    def __init__(self, n_fft=127, hop_length=16, visualize=False):
        self.n_fft = n_fft
        self.hop_length = hop_length # Based on the torch documentation, hop_length = n_fft / 4
        self.visualize = visualize
        self.window = torch.hann_window(n_fft)

    def __call__(self, sample):
        """
        sample shape: (num_frames, num_time, num_channels)
        output shape: (1, num_channels*2, freq_bins, time_steps)
        """
        _, _, num_channels = sample.shape

        # Flatten over time
        x = sample.permute(2, 0, 1).reshape(num_channels, -1)  # (num_channels, total_time)

        # STFT
        stfted = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )  # (num_channels, freq_bins, time_steps)

        magnitude = torch.log1p(torch.abs(stfted))
        phase = torch.angle(stfted)

        mag_phase = torch.stack([magnitude, phase], dim=1)  # (C, 2, F, T)
        mag_phase = mag_phase.view(num_channels * 2, magnitude.shape[2], magnitude.shape[1])  # (2*C, T, F)

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

        path = f"./STFT_plot/FFT_interp_ch{ch}.png"
        fig.savefig(path)
        print(f"[✓] Saved: {path}")
        plt.close(fig)
