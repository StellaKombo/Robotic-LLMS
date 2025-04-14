seq_len=96
root_path_name=/home/subuntu/Autoencoders/LLM/TOTEM/zero_shot_data-20250330T235451Z-001/zero_shot_data/
data_path_name=odometry_data.csv
data_name=custom
random_seed=2021
pred_len=96
gpu=0
trained_vqvae_model_path="/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/saved_models/base_odom/CD64_CW256_CF4_BS4096_ITR15000/checkpoints/final_model.pth"


# Check if the trained model already exists:
if [ ! -f "$trained_vqvae_model_path" ]; then
  echo "Model not found starting model training..."

  # Train the model
  python -u /home/subuntu/Autoencoders/LLM/TOTEM/forecasting/save_revin_data.py \
    --random_seed $random_seed \
    --data $data_name \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --features "M" \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 0 \
    --enc_in 13 \
    --gpu $gpu\
    --save_path "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/data/base_odom"
  echo "Data has been processed by revin and saved in the /data/base_odom..."

  echo "Training of the VQ-VAE has began..."
    gpu=0
    python /home/subuntu/Autoencoders/LLM/TOTEM/forecasting/train_vqvae.py \
      --config_path /home/subuntu/Autoencoders/LLM/TOTEM/forecasting/scripts/base.json \
      --model_init_num_gpus $gpu \
      --data_init_cpu_or_gpu cpu \
      --save_path "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/saved_models/base_odom/"\
      --base_path "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/data"\
      --batchsize 4096
  
  # Extract the final model path from the output
  final_model_path=$(echo "$output" | grep "FINAL_MODEL_PATH:" | sed 's/FINAL_MODEL_PATH: //')
  echo "Final trained VQ-VAE model path to be used: $final_model_path"
else
  echo "Model already exists. Skipping the model training."

  # Set final_model_path directly if the model exists
  final_model_path="$trained_vqvae_model_path"
  echo "Final trained VQ-VAE model path to be used: $final_model_path"

fi

# The extraction process
for pred_len in 96 192 336 500; do
  echo "Extracting the forecasting data for prediction length $pred_len"

  python -u /home/subuntu/Autoencoders/LLM/TOTEM/forecasting/extract_forecasting_data.py \
    --random_seed $random_seed \
    --data $data_name \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --features "M" \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 0 \
    --enc_in 13 \
    --gpu $gpu\
    --save_path "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/data/base_odom/Tin"$seq_len"_Tout"$pred_len"/" \
    --trained_vqvae_model_path "$final_model_path" \
    --compression_factor 4 \
    --classifiy_or_forecast "forecast"
done

# Training the forecaster
gpu=0
Tin=96
datatype=base_odom
echo "Training the forecaster with $Tin.."

for seed in 2021 1 13; do
  for Tout in 96 192 336 500; do
    python /home/subuntu/Autoencoders/LLM/TOTEM/forecasting/train_forecaster.py \
      --data-type $datatype \
      --Tin $Tin \
      --Tout $Tout \
      --cuda-id $gpu \
      --seed $seed \
      --data_path "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/data/"$datatype"/Tin"$Tin"_Tout"$Tout"" \
      --codebook_size 256 \
      --checkpoint \
      --checkpoint_path "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/saved_models/"$datatype"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
      --file_save_path "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/results/"$datatype"/"
  done
done