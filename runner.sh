python -u ./runner.py \
        --epochs 15\
        --batch_size 16\
        --learning_rate 0.005\
        --weight_decay 0.0004\
        --csv_file "$SCRATCH"\
        --checkpoint_dir "./checkpoints"\
        --experiment_name "first run"
