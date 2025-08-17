from dataloader import EAVDataset, EmognitionDataset, MDMERDataset


emognition = EmognitionDataset(
    motion_sampler = None,
    csv_file = "./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv",
    split = "test",
)

eav = EAVDataset(
    motion_sampler = None,
    csv_file = "./datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv",
    split = "test",
    )

mdmer = MDMERDataset(
    motion_sampler = None,
    csv_file = "./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv",
    split = "test",
)

def run_through(dataset):
    for i in range(len(dataset)):
        item = dataset[i]
        print(f"{i} :{item["video"].shape} : {item["eeg"].shape}")


run_through(mdmer)
run_through(emognition)
run_through(eav)

