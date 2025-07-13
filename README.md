## tsf4

Please train below things on the server based on the updated code.
Using updated_csv_folders (-> mv /datasets/updated_csv_folders/fold0) with cropped videos.

# Train lists
- Only video models for 3 datasets (3x3) -> Set args.eeg_signal = False
- EEG + Video models for 3 datasets (3x3) -> Set args.eeg_signal = True

- ! We should check the feasibility of EEG signal with FFT transformation as input.

# TODO
- Benchmark Audio-Video model (Audio -> EEG FFT) -> I'll add this code within next week.
