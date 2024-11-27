import numpy as np
import torch
import torch.utils.data as data
from collections import defaultdict
import random
import os
import librosa
import warnings
from transformers import Wav2Vec2FeatureExtractor
from configs.configs import Config

warnings.filterwarnings("ignore", category=UserWarning)

class AudioDataset(data.Dataset):
    def __init__(self, protocol_dir, partition, args):
        super(AudioDataset, self).__init__()

        self.partition = partition
        self.features = []
        self.max_audio_len = int(args.max_audio_len)
        self.base_path = Config.base_path
        self.model = args.model
        self.visual_type = args.visual_type
        self.wav2vec_model = "facebook/wav2vec2-xls-r-300m"
        self.wav2vec2_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav2vec_model)

        print("Reading ", protocol_dir)
        protocol_lines = open(protocol_dir).readlines()[1:]

        for line in protocol_lines:
            parts = line.strip().split("|")
            feature_path = parts[0]
            label = 0 if parts[1].lower() == "fake" else 1
            vocoder = parts[2]
            speaker = parts[3]
            self.features.append((feature_path, label, vocoder, speaker))

        if args.label_ratio < 1 and partition != "eval":
            data_by_attack_id = defaultdict(list)
            for row in self.features:
                data_by_attack_id[row[2]].append(row)
            sampled_features = []
            for sub_features in data_by_attack_id.values():
                sample_size = max(1, int(args.label_ratio * len(sub_features)))
                sampled_features.extend(random.sample(sub_features, sample_size))
            self.features = sampled_features

    def pad(self, x, max_len=128000):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]

        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (num_repeats))[:max_len]
        return padded_x

    def load_feature(self, feature_path):
        feature_path_complete = os.path.join(self.base_path, feature_path[1:])
        feature, sr = librosa.load(feature_path_complete, sr=16000)
        feature = self.pad(feature, max_len=self.max_audio_len)
        feature = self.wav2vec2_extractor(
            feature, feature_size=1, return_tensors="pt", sampling_rate=sr, adding=True
        )
        feature = feature.input_values
        return feature

    def __getitem__(self, index):
        feature_path, label, vocoder, _ = self.features[index]
        feature = self.load_feature(feature_path)
        return feature, torch.tensor(label), vocoder, feature_path

    def __len__(self):
        return len(self.features)