import pdb

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Neuro,
    Dataset_Saugeen_Web,
    Dataset_Odometry  # ✅ Add this import
)
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'neuro': Dataset_Neuro,
    'saugeen_web': Dataset_Saugeen_Web,
    'odometry': Dataset_Odometry,  
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(f"Dataset '{args.data}' not supported. Choose from: {list(data_dict.keys())}")

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'odometry':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    print(f"[{flag.upper()}] Dataset length: {len(data_set)}")

    return data_set, data_loader
