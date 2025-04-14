import os
import argparse
import numpy as np


def ltf_stride_multivariate(data, input_size, output_size, stride=1):
    # data: [channels, timesteps]
    start = 0
    middle = input_size
    stop = input_size + output_size

    data_x = []
    data_y = []

    while stop <= data.shape[1]:
        data_x.append(np.transpose(data[:, start:middle]))  # [time, sensors]
        data_y.append(np.transpose(data[:, middle:stop]))   # [time, sensors]

        start += stride
        middle += stride
        stop += stride

    # final shape: [num_samples, time, sensors]
    data_x_arr = np.stack(data_x)
    data_y_arr = np.stack(data_y)

    return data_x_arr, data_y_arr


def main(args):
    odometry = np.load(args.base_path)  # [N, 13]
    odometry = odometry.astype(np.float32)
    odometry = odometry.T  # Now [13, N] = [channels, time]

    input_size = 96
    for output_size in [96, 192, 336, 720]:
        data_x_arr, data_y_arr = ltf_stride_multivariate(odometry, input_size, output_size)

        print("x:", data_x_arr.shape, "y:", data_y_arr.shape)

        save_path = os.path.join(args.save_path, f'odometry/Tin{input_size}_Tout{output_size}')
        os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path, 'all_x_original.npy'), data_x_arr)
        np.save(os.path.join(save_path, 'all_y_original.npy'), data_y_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True, help='Path to odometry.npy')
    parser.add_argument('--save_path', type=str, required=True, help='Where to save processed data')
    args = parser.parse_args()

    main(args)
