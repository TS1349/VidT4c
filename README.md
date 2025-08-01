## tsf4

Please train below things on the server based on the updated code.
Using updated_csv_folders (-> mv /datasets/updated_csv_folders) with cropped videos.

# Train lists
- Only video models for 3 datasets (3x3 - fold0 csv) -> Set args.eeg_signal = False
- EEG + Video models for 3 datasets (3x3 - fold0 csv) -> Set args.eeg_signal = True

- We should check the feasibility of EEG signal with FFT transformation as input!

# Pretrained
- Download Hicmae model's pretrained weight through here (https://drive.google.com/file/d/1mR2r-_LWmtTYl4pS_L3YlhKNCCnv_ZaE/view) -> './pretrained/hicmae.pth'
- TVLT model's pretraind weight (https://huggingface.co/TVLT/models/resolve/main/TVLT.ckpt) -> './pretrained/tvlt.ckpt'

# TODO
- Benchmark Audio-Video model (Audio -> EEG FFT) -> Done
- Now you can train the two audio + video models (HicMAE, TVLT).
