import os
import numpy as np
import matplotlib.pyplot as plt


data_folder = "/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/results/base_odom/"
input = 96
output = 500
seed = 1
input_folder = f"/home/subuntu/Autoencoders/LLM/TOTEM/forecasting/data/base_odom/Tin{input}_Tout{output}/"

# Load the test ground truth data
odom_data = np.load(data_folder + f"Nominal_test_Tin{input}_Tout{output}_seed{seed}.npy")
input_data = np.load(input_folder + f"test_x_original.npy")


# Load the prediction data 
prediction_data = np.load(data_folder + f"Prediction_test_Tin{input}_Tout{output}_seed{seed}.npy")

# Check the shapes
print("test nominal odometry data shape:", odom_data.shape) # should be (num_samples, prediction, number_variables)
print("test nominal input shape:", input_data.shape)
print("test prediction odometry data shape:", prediction_data.shape) 

sample_index = output  # change this index to plot a different sample
feature_index = 12  # Variable to compare in this case - odom_lin_x

# Plot the original input (past sequence) and the forecast (future sequence)
input_sequence = odom_data[sample_index, :, feature_index]    # first feature if multi-dimensional
forecast_sequence = prediction_data[sample_index, :, feature_index]   # corresponding forecast (ground truth)

time_input = np.arange(len(input_sequence))

time_forecast = np.arange(len(forecast_sequence))

plt.figure(figsize=(12, 6))
plt.plot(time_input, input_sequence, label="Nominal ", color="blue")
plt.plot(time_forecast, forecast_sequence, label="Forecast Target", color="orange", marker="o")
plt.xlabel("Time Step")
plt.ylabel("Linear Odometry measurement")
plt.title(f"Sample {sample_index} |Feature {feature_index}: Forecast Visualization (Buffer window size {input}| Prediction window {output})| Seed {seed})")
plt.legend()
plt.grid(True)
plt.show()