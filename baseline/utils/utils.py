import numpy as np
import os

from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

import wandb

from baseline.data.Klinik import KlinikDataset
from baseline.data.bci import BCI2aDataset
from baseline.data.csp_feature import CSPFeature


def select_dataset(opts, transform=None):
    if opts["dataset"] == "KlinikDataset":
        dataset = KlinikDataset(
            eeg_electrode_positions=opts["eeg_electrode_positions"],
            data_path=opts["data_path"],
            meta_data=None,
            length=opts["length"],
            transforms=transform,
        )

    if opts["dataset"] == "BCI2aDataset":
        dataset = BCI2aDataset(
            eeg_electrode_positions=opts["eeg_electrode_positions"],
            data_path=opts["data_path"],
            meta_data=None,
            transforms=transform,
        )

    if opts["dataset"] == "CSPFeature":
        dataset = CSPFeature(
            feature_path=os.path.join(opts["data_path"], "features.pkl"),
            label_path=os.path.join(opts["data_path"], "labels.pkl"),
        )

    return dataset


def get_splitter(splitter="", n_splits=None, n_repeats=None):
    cv = getattr(model_selection, splitter)

    if splitter == "train_test_split":

        def wrapper(x, y, group):
            print(f"X : {len(x)}")
            print(f" Y : {len(y)}")
            print(f"Cross Validation Type : {type(cv)}")

            train_idx, test_idx = cv(
                x,
                stratify=y,
                test_size=0.2,
            )

            yield train_idx, test_idx

        return wrapper

    if splitter in ["LeaveOneGroupOut"]:
        return cv().split

    if splitter in ["GroupKFold", "StratifiedGroupKFold", "StratifiedKFold"]:
        return cv(n_splits=n_splits if n_splits is not None else 4).split

    if splitter in ["RepeatedStratifiedKFold"]:
        return cv(
            n_splits=n_splits if n_splits is not None else 4,
            n_repeats=n_repeats if n_repeats is not None else 10,
        ).split

    return None


def select_model(opts):
    if opts["model"] == "svm":
        return svm.SVC(decision_function_shape="ovo", probability=True)
    elif opts["model"] == "BalancedRandomForestClassifier":
        return BalancedRandomForestClassifier(
            n_estimators=100, max_depth=4, n_jobs=os.cpu_count()
        )

    elif opts["model"] == "XGBoost":
        params = opts.copy()
        params.pop("model", None)
        return XGBClassifier(use_label_encoder=False, **params)
    elif opts["model"] == "GaussianNaiveBayes":
        return GaussianNB()
    elif opts["model"] == "K-NearestNeighbores":
        return KNeighborsClassifier()
    else:
        print(f"unknown model type {opts['model']}")
        return None


def fit_eval(clf, X, Y, train_idx, test_idx, dataset):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(f"The data shape : {X_train.shape}")

    print("*** Training the Model...")
    clf.fit(X_train, Y_train)
    print("Done.")

    preds = evaluation("Prediction on Train data", clf, X_train, Y_train)

    temp_log = {"train accuracy": accuracy_score(preds, Y_train)}
    temp_log["train F1 (micro)"] = f1_score(preds, Y_train, average="micro")
    temp_log["train F1 (macro)"] = f1_score(preds, Y_train, average="macro")

    preds = evaluation("Prediction on Test data", clf, X_test, Y_test)

    temp_log["test accuracy"] = accuracy_score(preds, Y_test)
    temp_log["test F1 (micro)"] = f1_score(preds, Y_test, average="micro")
    temp_log["test F1 (macro)"] = f1_score(preds, Y_test, average="macro")

    if temp_log:
        wandb.log(temp_log)


# TODO Rename this here and in `fit_eval`
def evaluation(arg0, clf, x, y):
    # classes = list(map(str, dataset.classes))

    print(arg0)
    result = clf.predict(x)
    print("Done.")
    print(classification_report(y, result, zero_division=0))
    return result


def find_nan_idx(x):
    nan_idx = np.argwhere(np.isnan(x))
    trials = np.array(list(set([i[0] for i in nan_idx])))
    mask_trials = np.zeros(x.shape[0], dtype=bool)
    mask_trials[trials] = True

    return mask_trials


def find_inf_idx(x):
    inf_idx = np.argwhere(np.isinf(x))
    trials = np.array(list(set([i[0] for i in inf_idx])))
    print(f"{trials = }")
    mask_trials = np.zeros(x.shape[0], dtype=bool)
    mask_trials[trials] = True

    return mask_trials
