import os
import pdb
import argparse
import csv
import numpy as np
import pandas as pd
from distutils.util import strtobool
from datetime import datetime

def convert_csv_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN"
):
    col_names = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_section = False

    # Open the CSV file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        
        # Read the header line (first line) to extract column names
        header = next(csv_reader)
        col_names = header  # Use the header row as column names
        
        # Prepare the dictionary to hold data
        for col in col_names:
            all_data[col] = []

        # Iterate through the rest of the CSV data
        for line in csv_reader:
            if len(line) != len(col_names):
                raise Exception(f"Mismatch between data and column names in {full_file_path_and_name}.")
            
            # Process each column's value and handle missing data
            for i, val in enumerate(line):
                if val == "?":
                    all_data[col_names[i]].append(replace_missing_vals_with)
                else:
                    try:
                        # Convert numeric columns to float
                        if col_names[i] == "Time":
                            all_data[col_names[i]].append(float(val))  # Keep timestamp as float
                        else:
                            all_data[col_names[i]].append(float(val))  # Process other numeric data
                    except ValueError:
                        # If conversion fails, treat as string (for non-numeric columns)
                        all_data[col_names[i]].append(str(val))

            line_count += 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing column names section.")

    # Convert all_data dictionary into a DataFrame
    loaded_data = pd.DataFrame(all_data)

    # Optionally, convert the 'Time' column to datetime (if needed for indexing or analysis)
    if "Time" in loaded_data.columns:
        loaded_data["Time"] = pd.to_datetime(loaded_data["Time"], unit='s')

    return (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    )


def dropna(x):
    return x[~np.isnan(x)]


def ltf_stride(data, input_size, output_size, stride):  # want output shape to be (examples, time, sensors)
    start = 0
    middle = input_size
    stop = input_size + output_size

    data_x = []
    data_y = []

    # Iterate to create sliding windows for input and output
    for i in range(0, data.shape[0] - input_size - output_size + 1, stride):
        data_x.append(data[i:i + input_size, :])  # input_size time steps, all sensors
        data_y.append(data[i + input_size:i + input_size + output_size, :])  # output_size time steps, all sensors

    # Convert the list of arrays into a 3D numpy array (num_samples, time_steps, num_sensors)
    data_x_arr = np.stack(data_x, axis=0)  # Stacking along axis 0 to get the samples
    data_y_arr = np.stack(data_y, axis=0)  # Stacking along axis 0 to get the samples

    # Ensure the final shape is (num_samples, time_steps, num_sensors)
    print(f"Shape of data_x_arr: {data_x_arr.shape}")
    print(f"Shape of data_y_arr: {data_y_arr.shape}")


    return data_x_arr, data_y_arr 


def main(args):
    path = args.base_path

    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_csv_to_dataframe(path)

    time_series_columns = [
        "Odom_lin_X", "Odom_lin_Y", "Odom_lin_Z",
        "Odom_Qw", "Odom_Qx", "Odom_Qy", "Odom_Qz",
        "Odom_lin_velX", "Odom_lin_velY", "Odom_lin_velZ",
        "Odom_ang_velX", "Odom_ang_velY", "Odom_ang_velZ"
    ]

    # Drop NaN values for each series and convert them to float32
    timeseries = [df[col].dropna().astype(np.float32) for col in time_series_columns]

    # Convert list of series into a numpy array
    timeseries_arr = np.array([ts.to_numpy() for ts in timeseries]).T

    print(f"timeseries shape: {timeseries_arr.shape}")  # Check the shape of the resulting array


    # if 'saugeen' in path:
    #     in_size = 96
    #     out_size = 96
    #     data_x_arr, data_y_arr = ltf_stride(timeseries_arr, in_size, out_size, 1)

    #     print(data_x_arr.shape, data_y_arr.shape)

    #     save_path = args.save_path + '/saugeen/Tin'+ str(in_size)+ '_Tout' + str(out_size) + '/'
    #     print(save_path)

    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)

    #     np.save(save_path + 'all_x_original.npy', data_x_arr)
    #     np.save(save_path + 'all_y_original.npy', data_y_arr)

    # elif 'us_births' in path:
    #     in_size = 96
    #     out_size = 96
    #     data_x_arr, data_y_arr = ltf_stride(timeseries_arr, in_size, out_size, 1)

    #     print(data_x_arr.shape, data_y_arr.shape)

    #     save_path = args.save_path + '/us_births/Tin' + str(in_size) + '_Tout' + str(out_size) + '/'
    #     print(save_path)

    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)

    #     np.save(save_path + 'all_x_original.npy', data_x_arr)
    #     np.save(save_path + 'all_y_original.npy', data_y_arr)

    # elif 'sunspot' in path:
    #     in_size = 96
    #     out_size = 96
    #     data_x_arr, data_y_arr = ltf_stride(timeseries_arr, in_size, out_size, 1)

    #     print(data_x_arr.shape, data_y_arr.shape)

    #     save_path = args.save_path + '/sunspot/Tin' + str(in_size) + '_Tout' + str(out_size) + '/'
    #     print(save_path)

    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)

    #     np.save(save_path + 'all_x_original.npy', data_x_arr)
    #     np.save(save_path + 'all_y_original.npy', data_y_arr)

    print(f"Path received: {path}")
    if 'odometry_data' in path:
        in_size = 96
        out_size = 96
        data_x_arr, data_y_arr = ltf_stride(timeseries_arr, in_size, out_size, 1)

        print(data_x_arr.shape, data_y_arr.shape)

        save_path = args.save_path + '/base_odom/Tin' + str(in_size) + '_Tout' + str(out_size) + '/'
        print(f"Saved to base odometry: {save_path}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + 'all_x_original.npy', data_x_arr)
        np.save(save_path + 'all_y_original.npy', data_y_arr)



if __name__== '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_path', type=str, default='', help='base path to nc files')
    parser.add_argument('--save_path', type=str, default='', help='place to save processed files')

    args = parser.parse_args()
    main(args)


