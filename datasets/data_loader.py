import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.utils.data as data
from datasets.AudioDataset import AudioDataset
from configs.configs import Config
import random
import numpy as np


def get_protocol_dir(dataset_name):
    base_path = Config.base_path

    protocol_paths = {
        "ASVspoof2021DF": (
            os.path.join(base_path, "ASVspoof2021DF/train_meta.txt"),
            os.path.join(base_path, "ASVspoof2021DF/dev_meta.txt"),
            os.path.join(base_path, "ASVspoof2021DF/eval_meta.txt"),
            None,
        ),
        "ASVspoof5": (
            os.path.join(base_path, "ASVspoof5/train_meta.txt"),
            os.path.join(base_path, "ASVspoof5/dev_meta_SIGNL.txt"),
            os.path.join(base_path, "ASVspoof5/eval_meta_SIGNL.txt"),
            None,
        ),
        "InTheWild": (
            None,
            None,
            os.path.join(base_path, "InTheWild/eval_meta.txt"),
            None,
        ),
        "CFAD": (
            os.path.join(base_path, "CFAD/train_meta.txt"),
            os.path.join(base_path, "CFAD/dev_meta.txt"),
            os.path.join(base_path, "CFAD/eval_meta.txt"),
            os.path.join(base_path, "CFAD/eval_unseen_meta.txt"),
        ),
    }

    return protocol_paths.get(dataset_name)


def get_dataloader(args):
    print("Reading datasets with fixed length...")
    num_workers = 4
    print("Number of Workers:", num_workers)

    (
        train_protocol_path,
        dev_protocol_path,
        eval_protocol_path,
        eval_unseen_protocol_path,
    ) = get_protocol_dir(args.dataset)

    train_loader = dev_loader = None

    if args.dataset.lower() != "inthewild":
        train_dataset = AudioDataset(train_protocol_path, "train", args)
        dev_dataset = AudioDataset(dev_protocol_path, "dev", args)

        train_loader = DataLoader(
            dataset=train_dataset,
            drop_last=True,
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
        )

        dev_loader = DataLoader(
            dataset=dev_dataset,
            drop_last=True,
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
        )

    eval_dataset = AudioDataset(eval_protocol_path, "eval", args)
    if args.dataset == "CFAD" and eval_unseen_protocol_path:
        unseen_eval_dataset = AudioDataset(eval_unseen_protocol_path, "eval", args)
        eval_dataset = ConcatDataset([eval_dataset, unseen_eval_dataset])

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    print(f"Number of training data: {len(train_dataset) if train_loader else 0}")
    print(f"Number of dev data: {len(dev_dataset) if dev_loader else 0}")
    print(f"Number of eval data: {len(eval_dataset) if eval_loader else 0}")

    return train_loader, dev_loader, eval_loader
