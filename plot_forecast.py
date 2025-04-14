import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_predictions(pred_file, gt_file, save_dir, num_examples=5):
    os.makedirs(save_dir, exist_ok=True)
    predictions = np.load(pred_file)  # shape: (B, T, S)
    groundtruth = np.load(gt_file)

    # We'll only visualize the first sensor (S=1 assumed)
    assert predictions.shape == groundtruth.shape, "Shape mismatch"

    chunk_lengths = [96, 192, 336, 720]  # customize as needed

    for idx in range(min(num_examples, predictions.shape[0])):
        for chunk_idx, chunk_len in enumerate(chunk_lengths):
            if chunk_len > predictions.shape[1]:
                continue

            start = np.random.randint(0, predictions.shape[1] - chunk_len + 1)
            end = start + chunk_len

            pred = predictions[idx, start:end, 0]
            gt = groundtruth[idx, start:end, 0]

            plt.figure(figsize=(12, 4))
            plt.plot(gt, label='Ground Truth', color='black')
            plt.plot(pred, label='Forecasted', linestyle='--', color='red')

            plt.title(f"Odometry Forecast: Example {idx} | Chunk {chunk_idx+1} | Start={start} | Length={chunk_len}")
            plt.xlabel("Timestep")
            plt.ylabel("Sensor Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            filename = f"example{idx}_chunk{chunk_idx}_start{start}_len{chunk_len}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"🖼️ Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True, help="Path to predictions .npy file")
    parser.add_argument("--gt_file", required=True, help="Path to ground truth .npy file")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots (optional)")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to plot")
    args = parser.parse_args()

    plot_predictions(args.pred_file, args.gt_file, args.save_dir, args.num_examples)
