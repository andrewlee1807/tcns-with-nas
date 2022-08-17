#!/usr/bin/env bash
python auto_stride_searching.py --dataset_name=cnu --dataset_path=../dataset/ --history_len=168 --num_features=1 --max_trials=20 --device=0 --write_log_file=True

python auto_stride_searching.py --dataset_name=spain --dataset_path=../dataset/ --history_len=168 --num_features=1 --max_trials=20 --device=0 --write_log_file=True

python auto_stride_searching.py --dataset_name=household --dataset_path=../dataset/ --history_len=168 --num_features=1 --max_trials=20 --device=0 --write_log_file=True