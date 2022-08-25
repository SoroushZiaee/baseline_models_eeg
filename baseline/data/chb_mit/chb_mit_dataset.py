import imp
import torch
from torch.utils.data import Dataset
import random
import os
import pandas as pd
import numpy as np
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from .utils import CbhEdfFileMNE, ChbLabelWrapper, Patient


class CHBMITDataset(Dataset):
    """ """

    def __init__(
        self,
        eeg_electrode_positions: Dict[str, Tuple[int, int]],
        patient_list=None,
        data_path="/home/ec2-user/SageMaker/data/CHB-MIT Scalp EEG Database",
        transforms=None,
        length=1,
        verbose=False,
        dataframe=None,
        subset=False,
    ):

        if not subset:
            # UnChanged Block
            self.eeg_electrode_positions = eeg_electrode_positions
            self.patient_list = patient_list
            self.transforms = transforms
            self._data_path = data_path
            self._length = length
            self.verbose = verbose

            for i, patient in enumerate(patient_list):

                # Create a dataframe
                if not i:
                    self._meta_data = Patient(
                        self._data_path, patient, verbose=verbose
                    ).generate_metadata()

                else:
                    self._meta_data = pd.concat(
                        (
                            self._meta_data,
                            Patient(
                                self._data_path, patient, verbose=verbose
                            ).generate_metadata(),
                        ),
                        axis=0,
                    )

            self._meta_data.reset_index(drop=True, inplace=True)
            self._meta_data.to_csv(
                "patient_metadata.csv", sep=",", header=True, index=False
            )
            self.labels = self._meta_data["label"]

        else:
            self.eeg_electrode_positions = eeg_electrode_positions
            self.patient_list = patient_list
            self.transforms = transforms
            self._data_path = data_path
            self._length = length
            self.verbose = verbose

            self._meta_data = dataframe.copy()
            self._meta_data.reset_index(drop=True, inplace=True)

    def class_to_index(self, cls):
        return self.cls_idx_map[cls]

    def index_to_class(self, index):
        return self.classes[index]

    def get_class(self, index):
        return self._meta_data.loc[index, "label"]

    def get_class_distribution(self):
        return self._meta_data["label"].value_counts().to_list()

    def __len__(self):
        return len(self._meta_data)

    def __getitem__(self, idx):
        single_meta_data = self._meta_data.iloc[idx]
        label = self.get_class(idx)

        patient_id = single_meta_data["patient_name"]

        start = single_meta_data["start"]
        end = single_meta_data["end"]

        temp_cbh = CbhEdfFileMNE(
            os.path.join(self._data_path, single_meta_data["filename"]),
            patient_id,
            verbose=False,
        )
        rand_start = random.randint(
            start, end - int(self._length * temp_cbh.get_sampling_rate())
        )

        # shape -> (num_channels, n_times)
        eeg_data = temp_cbh.get_data(start=rand_start, length=self._length)

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
            patient_list=self.patient_list,
            data_path=self._data_path,
            transforms=self.transforms,
            length=self._length,
            verbose=self.verbose,
            dataframe=self._meta_data.iloc[indices],
            subset=True,
        )

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = torch.vstack([item[1] for item in batch]).squeeze()

        return [imgs, trgts]
