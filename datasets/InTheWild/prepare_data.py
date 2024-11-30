import zipfile
import os
import pandas as pd
import shutil

# Set up directories and paths
base_dir = "./datasets/InTheWild"
zip_file_name = "release_in_the_wild.zip"
zip_file_path = os.path.join(base_dir, zip_file_name)
audio_dir = os.path.join(base_dir, "release_in_the_wild")

# Extract the zip file
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(base_dir)

# Load the metadata
csv_file_path = os.path.join(audio_dir, "meta.csv")
meta_df = pd.read_csv(csv_file_path)

# Function to write data to text file and move files
def write_to_file(df, file_path, partition):
    dest_folder = os.path.join(base_dir, partition)
    os.makedirs(dest_folder, exist_ok=True)
    with open(file_path, "w") as file:
        file.write("id|label|vocoder|language|speaker\n")
        for _, row in df.iterrows():
            file_id = row["file"]
            label = "fake" if row["label"] == "spoof" else "real"
            vocoder = "None" if row["label"] == "spoof" else "real"
            speaker = row["speaker"]

            ext_folder = os.path.join(dest_folder, file_id.split(".")[-1])
            os.makedirs(ext_folder, exist_ok=True)

            shutil.move(
                os.path.join(audio_dir, file_id), os.path.join(ext_folder, file_id)
            )
            file.write(f"/InTheWild/{partition}/wav/{row["file"]}|{label}|{vocoder}|English|{speaker}\n")


# Paths for output text files
eval_file_path = os.path.join(base_dir, "eval_meta.txt")

# Write data to text files and move files
write_to_file(meta_df, eval_file_path, "eval")

# Function to print statistics of partitions
def print_partition_info(file_path, partition_name):
    df = pd.read_csv(file_path, sep="|")
    fake_count = len(df[df["label"] == "fake"])
    real_count = len(df[df["label"] == "real"])
    print(f"{partition_name} - Fake: {fake_count}, Real: {real_count}")


# Print statistics for partitions
print_partition_info(eval_file_path, "eval")

# Clean up the extracted folder
if os.path.exists(audio_dir):
    shutil.rmtree(audio_dir)
    print(f"Deleted folder: {audio_dir}")

# Verify creation of meta files
print("Meta files created:", os.listdir(base_dir))
