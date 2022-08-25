from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from torch import Tensor
import torch
import pywt

from functools import partial
from pytorch_lightning.utilities.apply_func import apply_to_collection
import mne

import numpy as np
import mne
import pandas as pd
import pickle

from hampel import hampel

import logging

logger = logging.getLogger(__name__)


class ToTensor(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, data, label):
        data = apply_to_collection(
            data,
            dtype=(np.ndarray, int, float, np.int64),
            function=lambda a: torch.tensor(a).float(),
        )
        label = apply_to_collection(
            label,
            dtype=(np.ndarray, int, float, np.int64),
            function=lambda a: torch.tensor(a).float(),
        )

        return data, label


class LabelToDict(object):
    def __call__(self, data, label):
        return data, {"label": label}

    def __repr__(self):
        return "LabelToDict"


class ZNorm(object):
    def __init__(
        self,
        stats: str,
        mode: str = "min-max",
        max_clip_val: int = 0,
        min_clip_val: int = None,
    ):
        self.stats_name = stats
        self.mode = mode
        self.min_clip_val = min_clip_val if min_clip_val is not None else -max_clip_val
        self.max_clip_val = max_clip_val
        with open(stats, "rb") as stats_f:
            self.stats = pickle.load(stats_f)

    def __call__(self, pkg: Tuple[Dict[str, Tensor], List[int]], target: Any):
        for k, st in self.stats.items():
            if k in pkg:
                if self.mode == "min-max":
                    minx = st["min"].unsqueeze(0).to(pkg[k].device)
                    maxx = st["max"].unsqueeze(0).to(pkg[k].device)
                    pkg[k] = (pkg[k] - minx) / (maxx - minx)
                if self.mode == "mean-std":
                    mean = st["mean"].unsqueeze(0).to(pkg[k].device)
                    std = st["std"].unsqueeze(0).to(pkg[k].device)
                    pkg[k] = (pkg[k] - mean) / std
                if self.max_clip_val > 0 or self.min_clip_val is not None:
                    pkg[k] = torch.clip(
                        pkg[k], min=self.min_clip_val, max=self.max_clip_val
                    )
            else:
                raise ValueError(f"couldn't find stats key {k} in package")
        return pkg, target


class Compose(object):
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, data: Any, target: Any):
        for t in self.transforms:
            data, target = t(data, target)
        return data, target

    def __repr__(self):
        return "Compose(" + f"-".join(str(t) for t in self.transforms) + ")"


class FlattenChannel(object):
    def __call__(self, data: Any, target: Any):
        for i, worker in enumerate(target.keys()):

            if worker == "label":
                pass
            else:
                flatten_worker = []
                for j, ch in enumerate(target[worker].keys()):
                    flatten_worker.append(target[worker][ch])

                target[worker] = np.array(flatten_worker)
                # print(f"target flattened : {target[worker].shape}")

        return data, target

    def __repr__(self):
        return f"FlattenChannel"


class ConcatenateWorker(object):
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, data: Any, target: Any):

        concatenate_worker = []

        for i, worker in enumerate(target.keys()):
            if worker == "label":
                pass
            else:
                # print(f"worker : {target[worker].shape}")
                concatenate_worker.append(target[worker])

        target["concat"] = np.concatenate(concatenate_worker, axis=-1)

        return data, target

    def __repr__(self):
        return f"ConcatenateWorker({'-'.join(str(t) for t in self.transforms)})"


class WTE(object):
    def __init__(self, level=4, wavelet="db1", name="wte"):
        self.level = level
        self.wavelet = wavelet
        self.name = name

    def __call__(
        self,
        data: Dict[str, np.ndarray],
        label: Dict[str, Union[Any, Dict[str, np.ndarray]]],
    ):

        label[self.name] = apply_to_collection(
            data,
            dtype=np.ndarray,
            function=partial(
                self.wavelet_transform_energy, level=self.level, wavelet=self.wavelet
            ),
        )

        return data, label

    def __repr__(self):
        attrs = "(level={}, wavelet={})".format(self.level, self.wavelet)
        return self.__class__.__name__ + attrs

    @staticmethod
    def wavelet_transform_energy(signal: np.ndarray, level: int, wavelet: str = "db1"):
        """calculates wavelet transform energy of a 1d signal

        Parameters
        ----------
        signal : numpy.ndarray
            raw signal. (eg. audio signal)
        level : int
            wavelet transform maximum level
        wavelet : str, optional
            wavelet type. one of the type available in :code:`pywt.wavelist()`,
             by default "db1"

        Returns
        -------
        numpy.ndarray
            The energy vector with shape of (level + 1,)

        Notes
        -----
        The WT energy can be calculated in different ways. Here we have implemented
        the method proposed in [1] equation (1) and [2] equation (2):

        .. math::
            \tilde{\mathbf{E}}_{\mathbf{V}_{j}} =
            \frac{\sum_{n} (\mathbf{w}_{j,n})^2}{\sum_{j=1}^{J_{max}} \sum_{n} (\mathbf{w}_{j,n})^2}

        Where :math:`\mathbf{w}_{j,n}` are the coefficients generated by DWT at the
        jth decomposition level.

        .. [1] K. Qian et al., “A bag of wavelet features for snore sound classification,”
           Ann. Biomed. Eng., vol. 47, no. 4, pp. 1000–1011, 2019.
        .. [2] Qian, K., C. Janott, Z. Zhang, C. Heiser, and B. Schuller.
           Wavelet features for classification of VOTE snore sounds.
           In: Proceedings of ICASSP, Shanghai, China, 2016, pp.221–225.
        """
        wt = pywt.wavedec(data=signal, wavelet=wavelet, mode="symmetric", level=level)

        ps_wt = np.array([np.sum(np.power(wt_j, 2)) for wt_j in wt])
        energy_vector = ps_wt / np.sum(ps_wt)

        return energy_vector


class WPTE(object):
    def __init__(
        self,
        level=4,
        wavelet="db1",
        include_raw=True,
        name="wpte",
    ):
        self.level = level
        self.wavelet = wavelet
        self.include_raw = include_raw
        self.name = name

    def __call__(
        self,
        data: Dict[str, np.ndarray],
        label: Dict[str, Union[Any, Dict[str, np.ndarray]]],
    ):

        label[self.name] = apply_to_collection(
            data,
            dtype=np.ndarray,
            function=partial(
                self.wavelet_packet_transform_energy,
                maxlevel=self.level,
                wavelet=self.wavelet,
                include_raw=self.include_raw,
            ),
        )

        return data, label

    def __repr__(self):
        attrs = "(level={}, wavelet={})".format(self.level, self.wavelet)
        return self.__class__.__name__ + attrs

    @staticmethod
    def wavelet_packet_energy(wpt_subspace: np.ndarray):
        """calculates the energy of a single subband from subspaces given by wpt

        Parameters
        ----------
        wpt_subspace : numpy.ndarray
            coefficients calculated by WPT from the signal at subspace V
            which is the kth subband at jth level.

        Returns
        -------
        int
            The energy of WPT sub space

        Notes
        -----
        The WPT energy can be calculated in different ways. Here we have implemented
        the method proposed in [1] equation (2):

        .. math:: \tilde{\mathbf{E}}_{\mathbf{V}_{j,k}} = log(\sqrt{\frac{\sum_{n=1}^{N_{j,k}} (\mathbf{w}_{j,k,n})^2}{N_{j,k}}})

        Where :math:`\mathbf{w}_{j,k,n}` represents the coefficients calculated by
        WPT from the signal at the subspace :math:`\mathbf{V}_{j,k}`.
        :math:`N_{j,k}` is the total number of wavelet coefficients in the kth subband
        at the jth level.

        .. [1] K. Qian et al., “A bag of wavelet features for snore sound classification,”
           Ann. Biomed. Eng., vol. 47, no. 4, pp. 1000–1011, 2019.
        """
        energy = np.log(
            np.sqrt(np.sum(np.power(wpt_subspace, 2) / wpt_subspace.shape[0]))
        )
        return energy

    @staticmethod
    def wavelet_packet_transform_energy(
        signal: np.ndarray,
        maxlevel: int,
        wavelet: str = "db1",
        include_raw: bool = True,
    ):
        """calculates wavelet packet transform energy of a 1d signal

        Parameters
        ----------
        signal : numpy.ndarray
            raw signal. (eg. audio signal)
        maxlevel : int
            wavelet packet transform maximum level
        wavelet : str, optional
            wavelet type. one of the types available in :code:`pywt.wavelist()`,
             by default "db1"

        Returns
        -------
        numpy.ndarray
            The energy vector with shape of (:math:`2^{maxlevel + 1} - 1`,)
        """
        energy_vector = []

        wp = pywt.WaveletPacket(
            data=signal, wavelet=wavelet, mode="symmetric", maxlevel=maxlevel
        )
        if include_raw:
            energy = WPTE.wavelet_packet_energy(signal)
            energy_vector.append(energy)

        for row in range(1, maxlevel + 1):
            for i in [node.path for node in wp.get_level(row, "freq")]:
                energy = WPTE.wavelet_packet_energy(wp[i].data)
                energy_vector.append(energy)

        return np.array(energy_vector)


class PSD(object):
    def __init__(
        self,
        sfreq=256,
        fmin=0,
        fmax=np.inf,
        n_fft=256,
        n_overlap=128,
        n_per_seg=256,
        average="mean",
        verbose=0,
        windowed=False,
        unit="Hz",  # Can be bin
        name="psd",
    ):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.n_per_seg = n_per_seg
        self.average = average
        self.windowed = windowed
        self.unit = unit
        self.verbose = verbose
        self.name = name

    def __call__(
        self,
        data: Dict[str, np.ndarray],
        label: Dict[str, Union[Any, Dict[str, np.ndarray]]],
    ):

        label[self.name] = apply_to_collection(
            data,
            dtype=np.ndarray,
            function=partial(
                self.power_spectral_density,
                sfreq=self.sfreq,
                fmin=self.fmin,
                fmax=self.fmax,
                n_fft=self.n_fft,
                n_overlap=self.n_overlap,
                n_per_seg=self.n_per_seg,
                average=self.average,
                verbose=self.verbose,
                windowed=self.windowed,
                unit=self.unit,
            ),
        )

        return data, label

    def __repr__(self):
        attrs = "(sfreq={}, n_fft={}, n_overlap={}, n_per_seg={})".format(
            self.sfreq, self.n_fft, self.n_overlap, self.n_per_seg
        )
        return self.__class__.__name__ + attrs

    @staticmethod
    def power_spectral_density(
        signal: np.ndarray,
        sfreq: int = 256,
        fmin: float = 0,
        fmax: float = np.inf,
        n_fft: int = 256,
        n_overlap: int = 128,
        n_per_seg: int = 256,
        average: str = "mean",
        verbose: int = 0,
        windowed: bool = False,
        unit: str = "Hz",
    ):

        if windowed:
            average = None

        data, _ = mne.time_frequency.psd_array_welch(
            signal,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            n_overlap=n_overlap,
            n_per_seg=n_per_seg,
            average=average,
            verbose=verbose,
        )

        if not windowed:
            data = (10 * np.log10(data * sfreq / n_fft)).flatten()
            # data = data.flatten()

        # The Shape of data should be [trial, window, psds]
        # Handle Window data
        if unit == "bin" and windowed:
            data = np.apply_along_axis(
                lambda x: 10 * np.log10(x * sfreq / n_fft),
                axis=2,
                arr=np.transpose(data, (0, 2, 1)),
            )

        return data
