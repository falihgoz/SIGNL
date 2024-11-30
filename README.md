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

- Download `LA.zip` from ASVspoof 2019 for train and dev sets [here](https://datashare.ed.ac.uk/handle/10283/3336) and save it into `./datasets/ASVspoof2021DF/`
- Download all `.tar.gz` files for eval sets [here](https://zenodo.org/records/4835108) and save them into `./datasets/ASVspoof2021DF/`
- Download `DF-keys-full.tar.gz` for eval set [here](https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz) and save it into `./datasets/ASVspoof2021DF/`
- Run: `python ./datasets/ASVspoof2021DF/prepare_data.py`

## ASVspoof 5

- Request the dataset from [https://www.asvspoof.org/](https://www.asvspoof.org/)
- Save `flac_T.tar` and `flac_D.tar` into `./datasets/ASVspoof5/`
- Run: `python ./datasets/ASVspoof5/prepare_data.py`

## CFAD

- Download all CFAD's zip files from [here](https://zenodo.org/records/8122764) and save them into `./datasets/CFAD/`
- Unzip all files into the same folder `./datasets/CFAD/`
- Run: `python ./datasets/CFAD/prepare_data.py`

## InTheWild

- Download `release_in_the_wild.zip` [here](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa) and save it to `./datasets/InTheWild/`
- Run: `python ./datasets/InTheWild/prepare_data.py`

# Evaluation using pre-trained model

- Download encoder file.

  - Download [here](https://drive.google.com/drive/folders/16F1vfRSpuRWV4bj9xwHhtzXIPdRHpYbo?usp=drive_link) and save to `./models/`.
  - Look for filenames starting with **"pretrained\_"** (e.g., `pretrained_W2VSIGNL_CFAD_ep100_bs96_lb100.ckpt`). `lb100` indicates that the model was built with full label information.

- Run command:

  ```bash
  python main.py --cls_eval True --dataset <dataset_name> --encoder_file <encoder_file>
  ```

- Examples:
  - Full labels & in-domain evaluation:
    ```bash
    $ python main.py --cls_eval True --dataset CFAD --encoder pretrained_SIGNL_CFAD_ep100_bs96_lb100.ckpt
    Testing DataLoader 0: 100%|██████████| 657/657 [09:03<00:00,  1.21it/s]
    ####### EVALUATION #######
    True Negatives (tn): 38501
    False Positives (fp): 3499
    False Negatives (fn): 1750
    True Positives (tp): 19250
    Accuracy: 0.9166825396825397
    Equal Error Rate (EER): 8.33%
    ```
    
  - 5% labels & in-domain evaluation:
    ```bash
    $ python main.py --cls_eval True --dataset CFAD --encoder pretrained_SIGNL_CFAD_ep100_bs96_lb5.ckpt
    Testing DataLoader 0: 100%|██████████| 657/657 [02:41<00:00,  4.06it/s]
    ####### EVALUATION #######
    True Negatives (tn): 37872
    False Positives (fp): 4128
    False Negatives (fn): 2066
    True Positives (tp): 18934
    Accuracy: 0.9016825396825396
    Equal Error Rate (EER): 9.84%
    ```
  
  - 5% labels & cross-domain evaluation:
    ```bash
    python main.py --cls_eval True --dataset InTheWild --encoder pretrained_SIGNL_CFAD_ep100_bs96_lb5.ckpt
    Testing DataLoader 0: 100%|██████████| 332/332 [01:03<00:00,  5.21it/s]
    ####### EVALUATION #######
    True Negatives (tn): 10787
    False Positives (fp): 1029
    False Negatives (fn): 1740
    True Positives (tp): 18223
    Accuracy: 0.9128669876333427
    Equal Error Rate (EER): 8.72%
    ```

# Downstream training using pre-trained encoders

- Download encoder file

  - Download [here](https://drive.google.com/drive/folders/16F1vfRSpuRWV4bj9xwHhtzXIPdRHpYbo?usp=drive_link) and save to `./models/`.
  - Look for filenames starting with **"encoder\_"** (e.g., `encoder_W2VSIGNL_CFAD_ep100_bs96.ckpt`).

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
