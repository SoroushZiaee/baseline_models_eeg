from typing import (
    Dict,
    Tuple,
    Union,
)

import os
import pickle

import torch
from torch.utils.data import Dataset

import numpy as np


def read_pickle_files(path: str):
    with open(path, "rb") as fin:
        return pickle.load(fin)


class CSPFeature(Dataset):
    def __init__(self, feature_path: str, label_path: str):
        self.features = read_pickle_files(feature_path)
        self.labels = read_pickle_files(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx].ravel(), self.labels[idx]
