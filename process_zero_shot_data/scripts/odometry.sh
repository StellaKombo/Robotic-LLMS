#!/bin/bash

# === CONFIG ===
gpu=0
seq_len=96
random_seed=2021
pred_len=0
enc_in=13  # 13 odometry features

# Paths
odometry_npy_path=~/Desktop/TokenizedTimeForecasting/odometry.npy
root_path_name=process_zero_shot_data/data/odometry
data_path_name=odometry
data_name=odometry

echo "🚗 Processing odometry data..."

# === Step 1: Convert odometry.npy to train/val/test .npy splits ===
python process_zero_shot_data/process_odometry_data.py \
    --npy_path $odometry_npy_path \
    --save_path $root_path_name \
    --seq_len $seq_len

# === Step 2: Generate revin + non-revin input windows ===
PYTHONPATH=. python process_zero_shot_data/save_notrevin_notrevinmasked_revinx_revinxmasked.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in $enc_in \
  --gpu $gpu \
  --save_path "process_zero_shot_data/data/imputation/odometry"

# === Step 3: Prepare anomaly detection formatted data ===
python process_zero_shot_data/prep_data_for_anomaly_detection.py \
    --base_path 'process_zero_shot_data/data/imputation/odometry/' \
    --save_path 'process_zero_shot_data/data/anomaly_detection/odometry/'

echo "✅ All odometry processing completed!"
