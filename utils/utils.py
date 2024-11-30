import yaml
import csv
import os


def load_config(config_file):
    with open(f"./configs/{config_file}", "r") as file:
        config = yaml.safe_load(file)
    return config


def synchronize_inputs(args):

    if args.dataset == "CFAD":
        config_file = "SIGNL_CFAD.yaml"
    else:
        config_file = "SIGNL_ASV.yaml"

    config = load_config(config_file)

    args.batch_size = (
        args.batch_size if args.batch_size is not None else config.get("batch_size")
    )
    args.dropout = args.dropout if args.dropout is not None else config.get("dropout")
    args.lr = args.lr if args.lr is not None else config.get("lr")
    args.max_audio_len = (
        args.max_audio_len
        if args.max_audio_len is not None
        else config.get("max_audio_len")
    )
    args.num_k = args.num_k if args.num_k is not None else config.get("num_k")
    args.num_patches_id = (
        args.num_patches_id
        if args.num_patches_id is not None
        else config.get("num_patches_id")
    )
    args.de = args.de if args.de is not None else config.get("de")
    args.gn = args.gn if args.gn is not None else config.get("gn")
    args.fm = args.fm if args.fm is not None else config.get("fm")
    args.visual_type = (
        args.visual_type if args.visual_type is not None else config.get("visual_type")
    )

    args.seed = args.seed if args.seed is not None else config.get("seed")

    return args


def eval_results_to_csv(
    model, dataset, label_ratio, epoch, eer, accuracy, encoder_path
):
    file_path = "results.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a" if file_exists else "w", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Models",
                    "Dataset",
                    "Label_ratio",
                    "Epochs",
                    "EER",
                    "Accuracy",
                    "Encoder",
                ]
            )
        writer.writerow(
            [model, dataset, label_ratio, epoch, eer, accuracy, encoder_path]
        )
