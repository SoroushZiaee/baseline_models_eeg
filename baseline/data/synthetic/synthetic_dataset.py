import torch
import numpy as np
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)


class EEGSyntheticDataset(Dataset):
    """ """

    def __init__(
        self,
        eeg_electrode_positions: Dict[str, Tuple[int, int]],
        transforms=None,
        data=None,
    ):
        self.eeg_electrode_positions = eeg_electrode_positions
        self.transforms = transforms

        if data is None:
            # distribution of class 1
            # from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
            rv1 = multivariate_normal([4, 2], [[2.0, 0], [0, 2.0]])
            x, y = np.mgrid[-6:6:0.01, -5:5:0.01]
            pos = np.dstack((x, y))

            # distribution of class 2
            rv2 = multivariate_normal([-4, -2], [[2.0, 0], [0, 2.0]])
            x, y = np.mgrid[-6:6:0.01, -5:5:0.01]
            pos = np.dstack((x, y))

            # Generating 100 samples from class 1
            self.eeg_data = {key: [] for key in eeg_electrode_positions.keys()}
            self.labels = []
            for i in range(100):
                # Generating 1 sec recording
                for key, value in eeg_electrode_positions.items():
                    self.eeg_data[key].append(
                        np.abs(np.random.normal(rv1.pdf(value), 0.01, 256))
                    )
                self.labels.append(1)

            # Generating another 100 samples from class 2
            for i in range(100):
                # Generating 1 sec recording
                for key, value in eeg_electrode_positions.items():
                    self.eeg_data[key].append(
                        np.abs(np.random.normal(rv2.pdf(value), 0.01, 256))
                    )
                self.labels.append(2)

            for key, value in eeg_electrode_positions.items():
                self.eeg_data[key] = np.expand_dims(np.array(self.eeg_data[key]), 1)
            self.labels = np.array(self.labels)
        else:
            self.eeg_data = data[0]
            self.labels = data[1]

        self.classes = [1, 2]
        self.cls_idx_map = {1: 0, 2: 1}

        self.indices = list(range(len(self.labels)))

    def class_to_index(self, cls):
        return self.cls_idx_map[cls]

    def index_to_class(self, index):
        return self.classes[index]

    def get_class(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wav = {key: self.eeg_data[key][idx, :, :] for key in self.eeg_data.keys()}

        label = self.get_class(idx)
        label = self.class_to_index(label)

        if self.transforms is not None:
            wav, label = self.transforms(wav, label)

        return wav, label

    def subset(self, indices):
        data = {}
        for key, value in self.eeg_data.items():
            data[key] = self.eeg_data[key][indices, :, :]
        return self.__class__(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data=(data, self.labels[indices]),
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
