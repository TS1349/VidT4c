## tsf4

Please train below things on the server based on the updated code.
Using updated_csv_folders (-> mv /datasets/updated_csv_folders) with cropped videos.

# Train lists
- Only video models for 3 datasets (3x3 - fold0 csv) -> Set args.eeg_signal = False
- EEG + Video models for 3 datasets (3x3 - fold0 csv) -> Set args.eeg_signal = True

- We should check the feasibility of EEG signal with FFT transformation as input!

# Pretrained
- [ViViT](https://github.com/rishikksh20/ViViT-pytorch) model's pretrained weight -> https://drive.google.com/file/d/1-JVhSN3QHKUOLkXLWXWn5drdvKn0gPll/view?usp=sharing -> './pretrained/vivit.pth'
- [TimeSFormer](https://github.com/facebookresearch/TimeSformer) model's pretrained weight -> https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0 -> './pretrained/tsf.pth'
- [Hicmae](https://dl.acm.org/doi/10.1016/j.inffus.2024.102382) model's pretrained weight -> https://drive.google.com/file/d/1mR2r-_LWmtTYl4pS_L3YlhKNCCnv_ZaE/view -> './pretrained/hicmae.pth'
- [TVLT](https://proceedings.neurips.cc/paper_files/paper/2022/file/3ea3134345f2e6228a29f35b86bce24d-Paper-Conference.pdf) model's pretraind weight -> https://huggingface.co/TVLT/models/resolve/main/TVLT.ckpt) -> './pretrained/tvlt.ckpt'

# TODO
- Our own framework code
- This should work for both compute facilites equall well
