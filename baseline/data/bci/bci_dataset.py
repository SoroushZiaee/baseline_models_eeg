from typing import (
    Dict,
    Tuple,
    Union,
)
import os
import random

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import warnings


class BCI2aDataset(Dataset):
    def __init__(
        self,
        eeg_electrode_positions: Dict[str, Tuple[int, int]],
        data_path: str,
        meta_data=None,
        transforms=None,
    ):
        self.eeg_electrode_positions = eeg_electrode_positions
        self.data_path = data_path

        if meta_data is None:
            self.meta_data = pd.read_csv(os.path.join(self.data_path, "metadata.csv"))
        else:
            self.meta_data = meta_data

        self.transforms = transforms

    def get_ptients(self):
        return self.meta_data["patient"].values

    def get_labels(self):
        return self.meta_data["label"].values

    def get_sampling_rate(self):
        return 250

    def get_resampling_rate(self):
        return 256

    def get_class_distribution(self):
        return self.meta_data["label"].value_counts()

    def __len__(self) -> int:
        return len(self.meta_data["file_name"])

    def __getitem__(self, idx: int) -> Union[dict, torch.Tensor]:

        meta_data = self.meta_data.iloc[idx]

        # shape -> (num_channels, n_times)
        eeg_data = np.load(
            os.path.join(self.data_path, "train" + "/" + meta_data["file_name"])
        )

        info = mne.create_info(
            list(self.eeg_electrode_positions.keys()),
            sfreq=self.get_sampling_rate(),
            ch_types="eeg",
        )

        warnings.simplefilter("ignore")

        eeg_data = (
            mne.io.RawArray(eeg_data, info, verbose=0)
            .filter(l_freq=2, h_freq=None, verbose=0)
            .resample(self.get_resampling_rate())
        )

        eeg_data = eeg_data.get_data()

        label = meta_data["label"] - 1

        wav = {
            key: np.expand_dims(eeg_data[i], axis=0)
            for i, key in enumerate(self.eeg_electrode_positions.keys())
        }

        if self.transforms is not None:
            wav, label = self.transforms(wav, label)

        return wav, label

    def subset(self, indices):
        return self.__class__(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            meta_data=self.meta_data.iloc[indices],
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
