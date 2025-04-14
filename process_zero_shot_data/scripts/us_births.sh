python '/home/subuntu/Autoencoders/LLM/TOTEM/process_zero_shot_data/process_saugeen_sun_births.py' \
    --base_path '/home/subuntu/Autoencoders/LLM/TOTEM/zero_shot_data-20250330T235451Z-001/zero_shot_data/us_births_dataset.tsf' \
    --save_path '../data'

gpu=0
seq_len=96
root_path_name=../data/us_births/Tin96_Tout96
data_path_name=saugeen  
data_name=saugeen  
random_seed=2021
pred_len=0


python -u ../save_notrevin_notrevinmasked_revinx_revinxmasked.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 1 \
  --gpu $gpu \
  --save_path "../data/imputation/us_births"

# draws from imputation processed data
# python ../prep_data_for_anomaly_detection.py \
#     --base_path '../data/imputation/us_births/' \
#     --save_path '../data/anomaly_detection/us_births/'

python ../forecasting_saugeen_sun_births.py \
    --base_path '/home/subuntu/Autoencoders/LLM/TOTEM/zero_shot_data-20250330T235451Z-001/zero_shot_data/us_births_dataset.tsf' \
    --save_path '../data/forecasting'
