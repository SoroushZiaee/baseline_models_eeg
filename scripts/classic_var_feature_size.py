from distutils.command import config
import os
import functools
import itertools

from parso import parse
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch import nn

import wandb


try:
    from dask.distributed import Client, progress
    import dask.bag as db

    # Make it True
    using_dask = False
except Exception:
    using_dask = False

from baseline.utils.transforms import *
from baseline.utils.utils import *
from baseline.configs.utils import eeg_electrode_configs, read_json_config


def prepare_transforms(worker_configs):
    transforms = [LabelToDict()]
    worker_transform = []
    # transforms = []
    for worker in worker_configs:
        name = worker["name"]
        if name == "wte":
            transforms.append(WTE())
            worker_transform.append(WTE())
        elif name == "wpte":
            transforms.append(WPTE())
            worker_transform.append(WPTE())
        elif name == "psd":
            transforms.append(PSD())
            worker_transform.append(PSD())

    transforms.append(FlattenChannel())
    transforms.append(ConcatenateWorker(worker_transform))

    # transforms.append(ToTensor(device=torch.device("cpu")))

    return Compose(transforms)


def extract_sample_and_transform(args):
    dataset, idx = args
    sample_w, sample_l = dataset[idx]

    return sample_w, sample_l, idx


def prepare_data(opts, dataset, extract_sample_and_transform, name="data"):

    X = []
    Y = []

    if not using_dask:
        for i in tqdm(range(len(dataset))):
            sample_w, sample_l, _ = extract_sample_and_transform((dataset, i))

            if opts["feature"]:
                X.append(sample_w)
                Y.append(sample_l)
            else:
                X.append(sample_l["concat"])
                Y.append(sample_l["label"])

        X = np.array(X)
        Y = np.array(Y)

        # Delete Nan from data
        if np.isnan(np.sum(X)):
            mask = find_nan_idx(X)
            print(f"mask true nan: {mask[mask == True].shape}")
            X = X[~mask]
            Y = Y[~mask]

        # Delete inf from data
        if np.isinf(np.sum(X)):
            mask = find_inf_idx(X)
            print(f"mask true inf: {mask[mask == True].shape}")
            X = X[~mask]
            Y = Y[~mask]

    return X, Y


def train(opts):
    print(f"Feature status : {opts['feature']}")
    transform = None if opts["feature"] else prepare_transforms(opts["workers"])

    dataset = select_dataset(opts, transform=transform)
    print(f"Dataset : {dataset}, Transformer : {transform}")

    # classes = np.array([dataset.get_class(i) for i in range(len(dataset))])

    # groups = np.array([dataset.get_class(i) for i in range(len(dataset))])

    X, Y = prepare_data(
        opts,
        dataset,
        extract_sample_and_transform,
        # name=f"{opts['transform']}_{opts['dataset']}",
    )

    print(f"X: {X.shape}, Y : {Y.shape}")

    for train_idx, test_idx in get_splitter(
        splitter=opts["data_splitter"], n_splits=opts["n_splits"]
    )(list(range(len(X))), Y, Y):

        wandb_run = wandb.init(
            project=opts["proj"], group=opts["wb_group"], reinit=True
        )
        wandb.config.update(opts)

        print(f"Train Samples : {len(train_idx)}")
        print(f"Test Samples : {len(test_idx)}")

        clf = select_model(opts)
        print(f"Model : {clf}")
        # clf = BatchVotingClassifier(clf)
        fit_eval(clf, X, Y, train_idx, test_idx, dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    # For specifying on WANDB
    parser.add_argument("--proj", type=str, default="test_project")
    parser.add_argument("--dataset", type=str, default="KlinikDataset")
    parser.add_argument("--data_path", type=str, default="/data/")
    parser.add_argument("--feature", type=bool, default=False)
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument(
        "--channel_config",
        type=str,
        default="../baseline/configs/eeg_recording_standard/international_10_20_21.py",
    )

    parser.add_argument(
        "--conf_path",
        type=str,
        default="../baseline/configs/run_configs/baseline.json",
    )

    # data spliter params
    parser.add_argument("--data_splitter", type=str, default="train_test_split")
    parser.add_argument("--va_split", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--n_repeats", type=int, default=10)

    return parser.parse_args()


# sourcery skip: dict-assign-update-to-union
if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)

    if not conf["feature"]:
        (
            conf["eeg_electrode_positions"],
            conf["eeg_electrods_plane_shape"],
        ) = eeg_electrode_configs(conf["channel_config"])

    configs = read_json_config(conf["conf_path"])

    for model, transform, splitter in itertools.product(
        configs["models"], configs["transforms"], configs["data_splitters"]
    ):

        temp_conf = conf.copy()
        temp_conf.update(model)
        temp_conf.update(transform)
        temp_conf.update(splitter)
        temp_conf[
            "wb_group"
        ] = f"{model['model']}_{'_'.join(worker['name'] for worker in transform['workers'])}_{splitter['data_splitter']}"

        train(temp_conf)
