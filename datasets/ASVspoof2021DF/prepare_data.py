import zipfile
import os
import shutil
import tarfile

base_dir = "./datasets/ASVspoof2021DF"
zip_file_name = "LA.zip"
zip_file_path = os.path.join(base_dir, zip_file_name)

tar_gz_files = [
    "ASVspoof2021_DF_eval_part00.tar.gz",
    "ASVspoof2021_DF_eval_part01.tar.gz",
    "ASVspoof2021_DF_eval_part02.tar.gz",
    "ASVspoof2021_DF_eval_part03.tar.gz",
]

extract_dir = base_dir
audio_dir = os.path.join(extract_dir, "LA")

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)


for file in tar_gz_files:
    try:
        print(f"Extracting {file}...")
        with tarfile.open(os.path.join(base_dir, file), "r:gz") as tar:
            tar.extractall(path=base_dir)
        print(f"{file} extracted successfully.")
    except Exception as e:
        print(f"Failed to extract {file}: {e}")


raw_keys = "DF-keys-full.tar.gz"
audio_dir_DF = os.path.join(extract_dir, "ASVspoof2021_DF_eval")

try:
    print(f"Extracting {raw_keys}...")
    with tarfile.open(os.path.join(base_dir, raw_keys), "r:gz") as tar:
        tar.extractall(path=base_dir)
    print(f"{raw_keys} extracted successfully.")
except Exception as e:
    print(f"Failed to extract {raw_keys}: {e}")

raw_protocol_eval_dir = os.path.join(extract_dir, "keys")


def write_to_file(file_path, partition):
    destination_folder = os.path.join(extract_dir, partition)
    os.makedirs(destination_folder, exist_ok=True)

    protocol_file_name = (
        f"ASVspoof2019.LA.cm.{partition}.trl.txt"
        if partition in ["dev", "eval"]
        else "ASVspoof2019.LA.cm.train.trn.txt"
    )

    with open(file_path, "w") as file:
        protocol_path = os.path.join(
            audio_dir, "ASVspoof2019_LA_cm_protocols", protocol_file_name
        )
        protocol_lines = open(protocol_path).readlines()

        file.write("id|label|vocoder|language|speaker\n")  # Write the header
        for line in protocol_lines:
            tokens = line.strip().split(" ")
            file_id = f"{tokens[1]}.flac"
            label = "fake" if tokens[4] == "spoof" else "real"
            vocoder = "real" if tokens[3] == "-" else tokens[3]
            language = "English"
            speaker = tokens[0]

            extension_folder = os.path.join(destination_folder, "flac")
            os.makedirs(extension_folder, exist_ok=True)

            source_path = os.path.join(
                audio_dir, f"ASVspoof2019_LA_{partition}", "flac", file_id
            )
            destination_path = os.path.join(extension_folder, file_id)
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
            else:
                print(f"File not found: {source_path}")

            file.write(
                f"/ASVspoof2021DF/{partition}/flac/{tokens[1]}.flac|{label}|{vocoder}|{language}|{speaker}\n"
            )


def write_to_file_eval(file_path, partition):
    with open(file_path, "w") as file:
        protocol_path = os.path.join(
            raw_protocol_eval_dir, "DF", "CM", "trial_metadata.txt"
        )
        protocol_lines = open(protocol_path).readlines()
        # The protocols look like this:
        #  [0]        [1]        [2]     [3]    [4]  [5]
        # LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -

        file.write("id|label|vocoder|language|speaker\n")  # Write the header
        for line in protocol_lines:
            tokens = line.strip().split(" ")
            label = "fake" if tokens[5] == "spoof" else "real"
            vocoder = "real" if tokens[4] == "-" else tokens[4]
            language = "English"
            speaker = tokens[0]
            file.write(
                f"/ASVspoof2021DF/{partition}/flac/{tokens[1]}.flac|{label}|{vocoder}|{language}|{speaker}\n"
            )


train_file_path = os.path.join(extract_dir, "train_meta.txt")
dev_file_path = os.path.join(extract_dir, "dev_meta.txt")
eval_file_path = os.path.join(extract_dir, "eval_meta.txt")

write_to_file(train_file_path, "train")
write_to_file(dev_file_path, "dev")
write_to_file_eval(eval_file_path, "eval")

if os.path.exists(audio_dir):
    shutil.rmtree(audio_dir)
    print(f"Deleted folder: {audio_dir}")

if os.path.exists(raw_protocol_eval_dir):
    shutil.rmtree(raw_protocol_eval_dir)
    print(f"Deleted folder: {raw_protocol_eval_dir}")

if os.path.exists(audio_dir_DF):
    shutil.move(
        os.path.join(audio_dir_DF, "flac"),
        os.path.join(extract_dir, "eval", "flac"),
    )
    shutil.rmtree(audio_dir_DF)
    print(f"Deleted folder: {audio_dir_DF}")

print("Meta files created:", os.listdir(extract_dir))
