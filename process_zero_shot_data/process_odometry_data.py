import argparse
import numpy as np
import os
import pandas as pd

def load_odometry_data(npy_path):
    columns = [
        "timestamp",
        "pos_x", "pos_y", "pos_z",
        "ori_x", "ori_y", "ori_z", "ori_w",
        "lin_vel_x", "lin_vel_y", "lin_vel_z",
        "ang_vel_x", "ang_vel_y", "ang_vel_z",
    ]

    data = np.load(npy_path, allow_pickle=True)
    print(f"✅ Loaded data: {data.shape}")
    df = pd.DataFrame(data, columns=columns)
    df["date"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')
    df.drop(columns=["timestamp"], inplace=True)
    df = df[["date"] + [col for col in df.columns if col != "date"]]
    print("📅 First timestamps:\n", df["date"].head())
    print("📈 First row of data (after date):\n", df.iloc[0, 1:])
    return df

def sliding_window_segments(df, full_seq_len):
    values = df.drop(columns=['date']).values
    timestamps = df['date'].dt.to_pydatetime()
    total_len = len(values)

    print(f"🔁 Creating segments with window size = {full_seq_len}")
    print(f"📊 Total rows available: {total_len}")

    segments = []
    timestamp_segments = []

    for i in range(total_len - full_seq_len + 1):
        segment = values[i:i + full_seq_len]
        ts_segment = [ts.isoformat() for ts in timestamps[i:i + full_seq_len]]
        segments.append(segment)
        timestamp_segments.append(ts_segment)

    segments = np.array(segments)
    timestamp_segments = np.array(timestamp_segments, dtype=object)

    print(f"✅ Created {len(segments)} segments")
    print(f"📐 Each segment shape: {segments.shape[1:]}")
    print(f"🕒 Sample timestamp window:\n{timestamp_segments[0][:5]}")
    return segments, timestamp_segments

def split_and_save(data, timestamps, save_path):
    total = len(data)
    train_end = int(total * 0.7)
    val_end = int(total * 0.8)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    train_ts = timestamps[:train_end]
    val_ts = timestamps[train_end:val_end]
    test_ts = timestamps[val_end:]

    os.makedirs(save_path, exist_ok=True)

    paths = {
        "train_data": train,
        "val_data": val,
        "test_data": test,
        "train_timestamps": train_ts,
        "val_timestamps": val_ts,
        "test_timestamps": test_ts,
    }

    for name, arr in paths.items():
        file_path = os.path.join(save_path, f"{name}.npy")
        np.save(file_path, arr)
        print(f"💾 Saved {name}.npy to {file_path} with shape {arr.shape} and dtype {arr.dtype}")

    print("\n📦 Final save summary:")
    print(f"  Train: {train.shape}, Timestamps: {train_ts.shape}")
    print(f"  Val:   {val.shape}, Timestamps: {val_ts.shape}")
    print(f"  Test:  {test.shape}, Timestamps: {test_ts.shape}")

    # Sample reload check
    try:
        reloaded_ts = np.load(os.path.join(save_path, "train_timestamps.npy"), allow_pickle=True)
        print("🧪 Reload test sample timestamps:\n", reloaded_ts[0][:5])
    except Exception as e:
        print("❌ Reload failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, required=True, help="Path to input odometry .npy file")
    parser.add_argument('--save_path', type=str, required=True, help="Directory to save split files")
    parser.add_argument('--seq_len', type=int, default=96, help="Input sequence length")
    parser.add_argument('--pred_len', type=int, default=96, help="Prediction length")

    args = parser.parse_args()

    full_seq_len = args.seq_len + args.pred_len
    print(f"\n🚀 Processing with seq_len={args.seq_len}, pred_len={args.pred_len}, total window={full_seq_len}\n")

    df = load_odometry_data(args.npy_path)
    all_segments, all_timestamps = sliding_window_segments(df, full_seq_len)
    split_and_save(all_segments, all_timestamps, args.save_path)
