#!/bin/bash

# === CONFIG ===
gpu=0
seq_len=96
random_seed=2021
pred_len_list=(96 192 336 720)
enc_in=13
compression_factor=4

root_path_name="process_zero_shot_data/data/odometry"
data_path_name="odometry"
data_name="odometry"
base_path="forecasting/data"

vqvae_config_path="forecasting/scripts/odometry_vqvae_config.json"
vqvae_save_path="forecasting/saved_models/odometry/"
vqvae_batch_size=512

# Define the expected VQ-VAE model path
trained_vqvae_model_path="${vqvae_save_path}/CD64_CW256_CF4_BS${vqvae_batch_size}_ITR10000_seed${random_seed}/checkpoints/final_model.pth"

# === Step 0: Optionally train VQ-VAE tokenizer on odometry data ===
read -p "Do you want to train the VQ-VAE model? (y/n): " train_choice

if [[ "$train_choice" == "y" || "$train_choice" == "Y" ]]; then
    echo "🎯 Training VQ-VAE on odometry dataset..."
    python forecasting/train_vqvae.py \
      --config_path $vqvae_config_path \
      --model_init_num_gpus $gpu \
      --data_init_cpu_or_gpu cpu \
      --save_path $vqvae_save_path \
      --base_path $root_path_name \
      --batchsize $vqvae_batch_size \
      --seed $random_seed  
else
    echo "⚡ Skipping VQ-VAE training. Using pre-trained model at:"
    echo "$trained_vqvae_model_path"
fi

# Wait until VQ-VAE model is saved
if [ ! -f "$trained_vqvae_model_path" ]; then
  echo "❌ ERROR: VQ-VAE model was not found at expected path:"
  echo "   $trained_vqvae_model_path"
  echo "   Aborting extraction and forecasting..."
  exit 1
fi

echo "✅ VQ-VAE model is ready. Proceeding to extract forecastable data."

# === Step 1: Save ReVIN-normalized inputs and codes from VQ-VAE ===
for pred_len in "${pred_len_list[@]}"; do
  echo "📦 Extracting forecastable data for pred_len=$pred_len"

  python -u forecasting/extract_forecasting_data.py \
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
    --save_path "${base_path}/odometry/Tin${seq_len}_Tout${pred_len}/" \
    --trained_vqvae_model_path $trained_vqvae_model_path \
    --compression_factor $compression_factor \
    --classifiy_or_forecast "forecast"
done

# === Step 2: Train forecaster and record results ===
for seed in 2021; do
  for pred_len in "${pred_len_list[@]}"; do
    echo "🚀 Running forecaster for Tin=$seq_len, Tout=$pred_len"

    python forecasting/train_forecaster.py \
      --data-type "odometry" \
      --Tin $seq_len \
      --Tout $pred_len \
      --cuda-id $gpu \
      --seed $seed \
      --data_path "${base_path}/odometry/Tin${seq_len}_Tout${pred_len}/" \
      --codebook_size 256 \
      --checkpoint \
      --checkpoint_path "forecasting/saved_models/odometry/forecaster_checkpoints/odometry_Tin${seq_len}_Tout${pred_len}_seed${seed}" \
      --file_save_path "forecasting/results/odometry/"
  done
done

echo "✅ Forecasting pipeline complete for odometry."

# === Step 3: Plot forecasts and open three example plots ===
echo "🎨 Generating forecast plots..."
python plot_forecast.py \
  --pred_file forecasting/results/odometry/predictions_Tin96_Tout720_seed2021.npy \
  --gt_file forecasting/results/odometry/groundtruth_Tin96_Tout720_seed2021.npy \
  --save_dir plots/odometry_forecast \
  --num_examples 3

echo "✅ Plots generated."

echo "🔍 Opening example plots..."
# Determine OS type to select the correct open command
if [[ "$OSTYPE" == "darwin"* ]]; then
    open_cmd="open"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    open_cmd="xdg-open"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    open_cmd="start"
else
    open_cmd="xdg-open"
fi

# Brief pause to ensure plots are written to disk
sleep 2

# Loop over the first three PNG files in the plots directory and open them
plot_dir="plots/odometry_forecast"
counter=0
for file in "$plot_dir"/*.png; do
  if [ $counter -ge 3 ]; then
      break
  fi
  echo "Opening $file..."
  $open_cmd "$file"
  counter=$((counter+1))
done

echo "✅ All done."
