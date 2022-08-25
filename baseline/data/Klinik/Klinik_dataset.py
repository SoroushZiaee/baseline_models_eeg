from typing import (
    Dict,
    Tuple,
    Union,
)

import os
import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset


class KlinikDataset(Dataset):
    def __init__(
        self,
        eeg_electrode_positions: Dict[str, Tuple[int, int]],
        data_path: str,
        meta_data=None,
        length=1,
        transforms=None,
    ):

        self.eeg_electrode_positions = eeg_electrode_positions
        self.data_path = data_path

        if meta_data is None:
            self.meta_data = pd.read_csv(os.path.join(self.data_path, "meta_data.csv"))
        else:
            self.meta_data = meta_data

        df_label_1 = self.meta_data[self.meta_data.label.eq(1)]
        df_label_0 = self.meta_data[self.meta_data.label.eq(0)]

        self.meta_data = shuffle(
            pd.concat(
                (
                    df_label_0.sample(n=len(df_label_1), replace=False),
                    df_label_1,
                ),
                axis=0,
            )
        )

        self.meta_data.reset_index(drop=True, inplace=True)

        self.labels = self.meta_data["label"]

        self.length = length
        self.transforms = transforms

    def __repr__(self):
        return "KlinikDataset"

    def get_class_distribution(self):
        return self.meta_data["label"].value_counts().to_list()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx: int) -> Union[dict, torch.Tensor]:

        meta_data = self.meta_data.iloc[idx]

        # shape -> (num_channels, n_times)
        # eeg_data = np.load(
        #     os.path.join(self.data_path, os.path.split(meta_data["file_pathe"])[1])
        # )

        eeg_data = np.load(os.path.join("/", meta_data["file_pathe"]))

        eeg_data = eeg_data[:21, :]

        label = int(meta_data["label"])

        wav = {
            key: np.expand_dims(eeg_data[i], axis=0)
            for i, key in enumerate(self.eeg_electrode_positions.keys())
        }

        if self.transforms is not None:
            wav, label = self.transforms(wav, label)

        return wav, label

    def get_class(self, index):
        return self.meta_data.iloc[index]["label"]

    def subset(self, indices):
        return self.__class__(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            meta_data=self.meta_data.iloc[indices],
            length=self.length,
            transforms=self.transforms,
        )

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = torch.vstack([item[1] for item in batch]).squeeze()

        return [imgs, trgts]
