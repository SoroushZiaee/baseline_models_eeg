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


class LEEDataset(Dataset):
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

        self.length = length
        self.transforms = transforms

    def get_ptients(self):
        return self.meta_data["patient_id"].values

    def get_sampling_rate(self):
        return 1000

    def get_resampling_rate(self):
        return 256

    def to_int_label(self, label: str) -> int:
        return 0 if label == "right" else 1

    def get_class_distribution(self):
        return self.meta_data["label"].value_counts()

    def __len__(self) -> int:
        return len(self.meta_data["file_pathes"])

    def __getitem__(self, idx: int) -> Union[dict, torch.Tensor]:

        meta_data = self.meta_data.iloc[idx]

        # shape -> (num_channels, n_times)
        eeg_data = np.load(os.path.join(self.data_path, meta_data["file_pathes"]))

        info = mne.create_info(
            list(self.eeg_electrode_positions.keys()),
            sfreq=self.get_sampling_rate(),
            ch_types="eeg",
        )

        rand_start = random.randint(
            0,
            eeg_data.shape[1] * (self.get_resampling_rate() / self.get_sampling_rate())
            - int(self.length * self.get_resampling_rate()),
        )

        eeg_data = (
            mne.io.RawArray(eeg_data, info, verbose=0)
            .filter(l_freq=2, h_freq=None)
            .resample(self.get_resampling_rate())
            .get_data(
                start=rand_start,
                stop=rand_start + self.length * self.get_resampling_rate(),
            )
        )

        label = self.to_int_label(meta_data["label"])

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
