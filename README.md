# SIGNL: Spatio-Temporal Vision Graph Non-Contrastive Learning

This repository contains the implementation of SIGNL, submitted to the 41st IEEE International Conference on Data Engineering (ICDE) 2025.


# Installation

This code requires Python 3.9 or higher.

```bash
pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip3 install -r requirements.txt
```

# Dataset

## ASVspoof 2021 DF
- Download `LA.zip` from ASVspoof 2019 for train and dev sets [here](https://datashare.ed.ac.uk/handle/10283/3336) and save it into `./datasets/ASVspoof2021/`
- Download all `.tar.gz` files for eval sets [here](https://zenodo.org/records/4835108) and save them into `./datasets/ASVspoof2021/`
- Run: `python ./datasets/ASVspoof2021/prepare_data.py`

## ASVspoof 5
- Request the dataset from [https://www.asvspoof.org/](https://www.asvspoof.org/)
- Save `flac_T.tar` and `flac_D.tar` into `./datasets/ASVspoof5/`
- Run: `python ./datasets/ASVspoof5/prepare_data.py`

## CFAD
- Download all CFAD's zip files from [here](https://zenodo.org/records/8122764) and save them into `./datasets/CFAD/`
- Run: `python ./datasets/CFAD/prepare_data.py`

## InTheWild
- Download `release_in_the_wild.zip` [here](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa) and save it to `./datasets/InTheWild/`
- Run: `python ./datasets/InTheWild/prepare_data.py`

# Evaluation using pre-trained model

- Download encoder file.
   - Download [here](https://drive.google.com/drive/folders/16F1vfRSpuRWV4bj9xwHhtzXIPdRHpYbo?usp=drive_link) and save to `./models/`.  
   - Look for filenames starting with **"pretrained_"** (e.g., `pretrained_W2VSIGNL_CFAD_ep100_bs96_lb100.ckpt`). `lb100` indicates that the model was built with full label information.

- Run command:  
    ```bash
    python main.py --eval_cls True --dataset <dataset_name> --encoder_file <encoder_file>
    ```

- Examples:
   - Full labels:  
     ```bash
     python main.py --eval_cls True --dataset CFAD --encoder pretrained_SIGNL_CFAD_ep100_bs96_lb100.ckpt
     ```
   - 5% labels:  
     ```bash
     python main.py --eval_cls True --dataset CFAD --encoder pretrained_SIGNL_CFAD_ep100_bs96_lb5.ckpt
     ```

# Downstream training using pre-trained encoders

- Download encoder file
   - Download [here](https://drive.google.com/drive/folders/16F1vfRSpuRWV4bj9xwHhtzXIPdRHpYbo?usp=drive_link) and save to `./models/`.  
   - Look for filenames starting with **"encoder_"** (e.g., `encoder_W2VSIGNL_CFAD_ep100_bs96.ckpt`).

- Run command:  
    ```bash
    python main.py --training_type classifier --dataset <dataset_name> --encoder_file <encoder_file> --epoch <number_of_epochs> --label_ratio <label_availability_ratio>
    ```
- Example:
    ```bash
    python main.py --training_type classifier --dataset CFAD --encoder encoder_SIGNL_CFAD_ep100_bs96.ckpt --epoch 100 --label_ratio 0.8
    ```  

# Run from scratch

Perform pre-training and downstream training from scratch.

- **Pre-train the encoders**  
    - Run command:
    ```bash
    python main.py --training_type encoder --dataset <dataset_name> --epoch <number_of_epochs>
    ```

    - Example:
    ```bash
    python main.py --training_type encoder --dataset CFAD --epoch 100
    ```
- **Downstream training**  
    - Get the recently pre-trained encoder file under `./models/` (e.g., `encoder_W2VSIGNL_CFAD_ep100_bs96.ckpt`).

    - Start the downstream training:
    ```bash
    python main.py --training_type classifier --dataset CFAD --encoder encoder_SIGNL_CFAD_ep100_bs96.ckpt --epoch 100 --label_ratio 0.8
    ```

# Results
## In-Domain Results
![Alt text](results/indomain.png?raw=true "results")

## Cross-Domain Results
![Alt text](results/crossdomain.png?raw=true "results")

# Cite this work
TBD

