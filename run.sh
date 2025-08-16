#!/bin/bash

fold_csv="./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv"

python -u ./runner.py \
        --epochs 50\
        --num_gpus 1\
        --batch_size 8\
        --learning_rate 0.01\
        --weight_decay 0.0004\
        --csv_file "$fold_csv"\
        --checkpoint_dir "./checkpoints"\
        --experiment_name "emognition_swin_0"\
        --dataset "emognition"\
        --model "vivit"\
        --server "nef"\
        --port 12355

