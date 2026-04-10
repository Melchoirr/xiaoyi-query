from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute


data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(f"unsupported dataset {args.data}")

    data_cls = data_dict[args.data]
    timeenc = 1 if args.embed == "timeF" else 0

    shuffle_flag = flag == "train"
    drop_last = flag == "train"

    dataset = data_cls(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return dataset, loader
