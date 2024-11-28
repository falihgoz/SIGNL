import os
import pandas as pd
import shutil
import zipfile

# Define base directory paths
base_dir = "./datasets/CFAD"
raw_dataset_dir = os.path.join(base_dir, "CFAD")
audio_dir = os.path.join(raw_dataset_dir, "clean_version")

# Define partitions
partitions = ["train_clean", "dev_clean", "test_seen_clean", "test_unseen_clean"]
labels = ["fake_clean", "real_clean"]


def print_partition_info(file_path, partition_name):
    df = pd.read_csv(file_path, sep="|")
    fake_count = len(df[df["label"] == "fake"])
    real_count = len(df[df["label"] == "real"])
    print(f"{partition_name} - Fake: {fake_count}, Real: {real_count}")


def generate_protocol_txt(partition, audio_dir):
    """
    Generates a protocol .txt file for a given partition.

    Args:
        partition (str): The partition name (e.g., training, validation, testing).
        audio_dir (str): Base directory containing the datasets.
    """
    # Paths
    partition_dir = os.path.join(audio_dir, partition)
    partition_rename = {
        "train_clean": "train",
        "dev_clean": "dev",
        "test_seen_clean": "eval",
        "test_unseen_clean": "eval_unseen",
    }

    dest_folder = os.path.join(base_dir, partition_rename[partition])
    os.makedirs(dest_folder, exist_ok=True)
    protocol_file = os.path.join(base_dir, f"{partition_rename[partition]}_meta.txt")

    # Collect metadata
    metadata = []
    for label in labels:
        label_dir = os.path.join(partition_dir, label)
        vocoder_folders = os.listdir(
            label_dir
        )  # List vocoder folders inside fake_clean
        for vocoder in vocoder_folders:
            vocoder_dir = os.path.join(label_dir, vocoder)
            for file_name in os.listdir(vocoder_dir):
                file_path = os.path.join(vocoder_dir, file_name)
                label_flag = "fake" if label == "fake_clean" else "real"
                ext_folder = os.path.join(dest_folder, file_name.split(".")[-1])
                os.makedirs(ext_folder, exist_ok=True)
                shutil.move(file_path, os.path.join(ext_folder, file_name))

                metadata.append(
                    {
                        "id": f"/CFAD/{partition_rename[partition]}/{file_name.split('.')[-1]}/{file_name}",
                        "label": label_flag,
                        "vocoder": vocoder,
                        "language": "Chinese",
                        "speaker": "unknown",
                    }
                )

    # Save to protocol file
    df = pd.DataFrame(metadata)
    df.to_csv(protocol_file, sep="|", index=False, header=True)
    print(f"Protocol file saved: {protocol_file}")
    print_partition_info(protocol_file, partition)


# Generate protocol files for each partition
for partition in partitions:
    generate_protocol_txt(partition, audio_dir)

if os.path.exists(raw_dataset_dir):
    shutil.rmtree(raw_dataset_dir)
    print(f"Deleted folder: {raw_dataset_dir}")
