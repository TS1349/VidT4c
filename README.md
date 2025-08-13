## tsf4

Please train below things on the server based on the updated code.
Using updated_csv_folders (-> mv /datasets/updated_csv_folders) with cropped videos.

# Train lists
- You should set learning rate as 1e-4 ~ 1e-5 and weight_decay as 5e-5 with existing training code. Please find the appropriate learning parameters including lrscheduler_step, lrscheduler_decay.
- The prefixed option value seems good training with 4 batch size.
- Recommend setup for all models -> Batch: 32, Epochs 50

  
- Only video models for 3 datasets (3x3 - fold0 csv) -> Set args.eeg_signal = False
- EEG + Video models for 3 datasets (3x3 - fold0 csv) -> Set args.eeg_signal = True

- Audio + Video models for 3 datasets (2x3 - fold0 csv) -> Set args.fft_mode = 'Spectrogram', args.pretraind=True

- Our own framework -> Set args.model=vemt, args.eeg_signal=True, args.gcn=False
  + gcn : Set args.eeg_signal=True, args.gcn=True
  + You can also train with one network by set_eeg_only and set_video_only options on vemt model


# Pretrained
- [ViViT](https://github.com/rishikksh20/ViViT-pytorch) model's pretrained weight -> https://drive.google.com/file/d/1-JVhSN3QHKUOLkXLWXWn5drdvKn0gPll/view?usp=sharing -> './pretrained/vivit.pth'
- [TimeSFormer](https://github.com/facebookresearch/TimeSformer) model's pretrained weight -> https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0 -> './pretrained/tsf.pth'
- [Hicmae](https://dl.acm.org/doi/10.1016/j.inffus.2024.102382) model's pretrained weight -> https://drive.google.com/file/d/1mR2r-_LWmtTYl4pS_L3YlhKNCCnv_ZaE/view -> './pretrained/hicmae.pth'
- [TVLT](https://proceedings.neurips.cc/paper_files/paper/2022/file/3ea3134345f2e6228a29f35b86bce24d-Paper-Conference.pdf) model's pretraind weight -> https://huggingface.co/TVLT/models/resolve/main/TVLT.ckpt) -> './pretrained/tvlt.ckpt'
- Our framework needs [VideoMaev2](https://github.com/OpenGVLab/VideoMAEv2/tree/master)'s pretrained weight -> https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth -> './pretrained/vemt.pth'

# Mamba Setup
Please refer [AudioMamba](https://github.com/kaistmm/Audio-Mamba-AuM) page to setup environment.

# TODO
- This should work for both compute facilites equall well
