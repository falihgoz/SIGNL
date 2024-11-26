import argparse
import warnings
import sys
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from distutils.util import strtobool
from datasets.data_loader import get_dataloader
from utils.model_runner import (
    start_enc_training,
    start_cls_training,
    cls_eval_only,
)
from utils.utils import synchronize_inputs
import torch
import pytorch_lightning as pl
import transformers

def set_seed(seed):
    pl.seed_everything(seed, workers=True)
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def str2bool(v):
    return bool(strtobool(v))

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=PossibleUserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--training_type",
        default="classifier",
        choices=["encoder", "classifier", "full"],
        help="Type of training to perform: 'encoder', 'classifier', or 'full'.",
    )

    parser.add_argument(
        "--dataset",
        default="ASVspoof2021",
        help=(
            "Dataset to use for training or evaluation. Options for training: "
            "'asvspoof19', 'asvspoof21', 'CFAD'. Option for evaluation only: 'inthewild'."
        ),
    )

    parser.add_argument(
        "--model",
        default="W2VSIGNL",
        help="Model architecture to use.",
    )

    parser.add_argument(
        "--cls_eval",
        type=str2bool,
        default=False,
        help="Set to True for evaluation only using a pre-trained model.",
    )

    parser.add_argument(
        "--encoder_path",
        default=None,
        help="Path to a pre-trained encoder model, if available.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--label_ratio",
        type=float,
        default=1.0,
        help="Ratio of labeled data to be used for training.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="Learning rate for training.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate for regularization.",
    )

    parser.add_argument(
        "--max_audio_len",
        type=float,
        help="Maximum length of input audio in seconds.",
    )

    parser.add_argument(
        "--num_k",
        type=int,
        help="Number of 'k' used in the training configuration.",
    )

    parser.add_argument(
        "--num_patches_id",
        type=int,
        help="Number of patches used for identification.",
    )

    parser.add_argument(
        "--de",
        type=str2bool,
        help="Enable or disable SIGNL pair augmentation: drop edge.",
    )

    parser.add_argument(
        "--gn",
        type=str2bool,
        help="Enable or disable SIGNL pair augmentation: add Gaussian noise.",
    )

    parser.add_argument(
        "--fm",
        type=str2bool,
        help="Enable or disable SIGNL pair augmentation: feature masking.",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Specify the hardware accelerator: 'gpu' or 'cpu'.",
    )

    args = parser.parse_args()
    args = synchronize_inputs(args)

    print("seed number: ", args.seed)
    set_seed(args.seed)
    
    train_loader, dev_loader, eval_loader, itw_eval_loader = get_dataloader(args)

    if args.cls_eval:
        cls_eval_only(args, eval_loader, itw_eval_loader)
        sys.exit(0)

    encoder_training = args.training_type in ["encoder", "full"]
    cls_training = args.training_type in ["classifier", "full"]

    if encoder_training:
        args.encoder_path = start_enc_training(args, train_loader, dev_loader)

    if cls_training:
        start_cls_training(args, train_loader, dev_loader, eval_loader, itw_eval_loader)
