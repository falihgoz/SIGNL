import os
import tarfile
import logging

base_dir = "./datasets/ASVspoof5"
train_tar_file = "flac_T.tar"
dev_tar_file = "flac_D.tar"

for folder in ["train", "dev", "eval"]:
    destination_folder = os.path.join(base_dir, folder)
    os.makedirs(destination_folder, exist_ok=True)

train_tar_path = os.path.join(base_dir, train_tar_file)
dev_tar_path = os.path.join(base_dir, dev_tar_file)

extract_dir = base_dir
train_audio_dir = os.path.join(extract_dir, "train")
dev_audio_dir = os.path.join(extract_dir, "dev")

os.makedirs(train_audio_dir, exist_ok=True)
os.makedirs(dev_audio_dir, exist_ok=True)

# Password for the ZIP files
# password = "...." #change the password


# Function to rename directories
def rename_directory(old_dir, new_dir):
    if os.path.exists(old_dir):
        os.rename(old_dir, new_dir)
    else:
        logging.warning(f"Directory {old_dir} does not exist.")


# Function to extract tar files
def extract_tar(tar_path, extract_to):
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extract_to)
        logging.info(f"Extracted tar file: {tar_path} to {extract_to}")
    except Exception as e:
        logging.error(f"Failed to extract tar file: {tar_path} - {e}")


# Extract train and dev tar files
extract_tar(train_tar_path, train_audio_dir)
extract_tar(dev_tar_path, dev_audio_dir)

# Rename directories
rename_directory(
    os.path.join(train_audio_dir, "flac_T"), os.path.join(train_audio_dir, "flac")
)
rename_directory(
    os.path.join(dev_audio_dir, "flac_D"), os.path.join(dev_audio_dir, "flac")
)

print("Extraction complete.")
