import os
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
import datetime
import numpy as np
import re
import glob
import pandas as pd
import warnings
import mne


from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)


class CbhEdfFileMNE(object):
    def __init__(self, filename, patient_id=None, verbose=None):

        self._filename = filename
        self._patient_id = patient_id

        # ignore error of the getting
        warnings.simplefilter("ignore")
        self._raw = mne.io.read_raw_edf(self._filename, preload=False, verbose=verbose)

    def get_filename(self):
        return self._filename

    def get_n_channels(self):
        """
        Number of channels
        """

        return len(self._raw.ch_names)

    def get_n_data_points(self):
        """
        Number of data points
        """

        return self._raw.n_times

    def get_channel_names(self):
        """
        Names of channels
        """
        return self._raw.ch_names

    def get_file_duration(self):
        """
        Returns the file duration in seconds
        """

        return self._raw.n_times / self._raw.info["sfreq"]

    def get_sampling_rate(self):
        """
        Get the frequency
        """

        return self._raw.info["sfreq"]

    def get_data(self, channel_id_list: list = None, start=None, end=None, length=None):
        """
        Get raw data for a single channel or multiple channel or whole channel
        Parameters
        ----------
        start : an integer
            Start time in timepoint

        end : an integer
            End time in timepoint

        length: an integer
            Indicate the length of the clips in seconds
        """

        if start is None:
            if channel_id_list == None or len(channel_id_list) == 0:
                return self._raw.get_data()
            else:
                return self._raw[channel_id_list, :][0]

        # Define the end
        if end is None:
            if length is None:
                end = int(self.get_n_data_points())
            else:
                end = start + length * self.get_sampling_rate()

        if channel_id_list == None or len(channel_id_list) == 0:
            return self._raw[:, start:end][0]

        return self._raw[channel_id_list, start:end][0]

    def get_start_datetime(self):
        """
        Get the starting date and time
        """
        return self._raw.info["meas_date"]

    def get_end_datetime(self):
        return self.get_start_datetime() + datetime.timedelta(
            seconds=self.get_file_duration()
        )


class ChbLabelWrapper:
    """
    Class for handling the labels
    """

    def __init__(self, filename, patient_id):
        self._patient_id = patient_id
        self._filename = filename
        self._file = open(filename, "r")
        self._parse_file(self._file)
        self._file.close()

    def _parse_file(self, file_obj):
        """
        Parse the file object
        :param file_obj: Opened file object
        """
        # Split file into blocks
        data = file_obj.read()
        blocks = data.split("\n\n")

        # Block zero
        self._frequency = self._parse_frequency(blocks[0])

        # Block one
        self._channel_names = self._parse_channel_names(blocks[1])

        # Block two-N
        self._metadata_store = self._parse_file_metadata(blocks[2:])

    def _parse_frequency(self, frequency_block):
        """
        Frequency block parsing with format
        'Data Sampling Rate: ___ Hz'
        :param frequency_block: Frequency block
        :return: Parses and returns the frequency value in Hz
        """
        pattern = re.compile("Data Sampling Rate: (.*?) Hz")
        result = pattern.search(frequency_block)
        # Check if there is a match or not
        if result is None:
            raise ValueError(
                'Frequency block does not contain the correct string ("Data Sampling Rate: __ Hz")'
            )
        result = int(result.group(1))
        return result

    def _parse_channel_names(self, channel_block):
        """
        Get channel names from the blocks
        :param channel_block: List of Channel names
        :return: Returns the channel names as a list of strings
        """
        # Split by line
        lines = channel_block.split("\n")
        pattern = re.compile("Channel [0-9]{1,}: (.*?)$")

        output_channel_list = []
        for line in lines:
            channel_name = pattern.search(line)
            if channel_name is not None:
                channel_name = channel_name.group(1)
                output_channel_list.append(channel_name)

        return output_channel_list

    def _parse_metadata(self, metadata_block, output_metadata):
        """
        Parse a single seizure metadata block
        \TODO Replace individual file metadata with a named structure
        :param metadata_block:
        :return:
        """
        # Search first line for seizure file pattern
        pattern_filename = re.compile("File Name: (.*?)$")

        # Patient 24 doesn't have these two sentences
        pattern_start_time = re.compile("File Start Time: (.*?)$")
        pattern_end_time = re.compile("File End Time: (.*?)$")
        pattern_seizures = re.compile("Number of Seizures in File: (.*?)$")
        pattern_seizure_start = re.compile(
            "Seizure [0-9]{0,}[ ]{0,}Start Time: (.*?) seconds"
        )
        pattern_seizure_end = re.compile(
            "Seizure [0-9]{0,}[ ]{0,}End Time: (.*?) seconds"
        )

        if pattern_filename.search(metadata_block[0]) is not None:
            if self._patient_id == 24:
                file_metadata = dict()
                filename = pattern_filename.search(metadata_block[0]).group(1)
                file_metadata["n_seizures"] = int(
                    pattern_seizures.search(metadata_block[1]).group(1)
                )
                file_metadata["channel_names"] = self._channel_names
                file_metadata["sampling_rate"] = self.get_sampling_rate()
                seizure_intervals = []
                for i in range(file_metadata["n_seizures"]):
                    seizure_start = int(
                        pattern_seizure_start.search(metadata_block[2 + i * 2]).group(1)
                    )
                    seizure_end = int(
                        pattern_seizure_end.search(metadata_block[2 + i * 2 + 1]).group(
                            1
                        )
                    )
                    seizure_intervals.append((seizure_start, seizure_end))
                file_metadata["seizure_intervals"] = seizure_intervals
                output_metadata[filename] = file_metadata
            else:
                file_metadata = dict()
                filename = pattern_filename.search(metadata_block[0]).group(1)
                file_metadata["start_time"] = pattern_start_time.search(
                    metadata_block[1]
                ).group(1)
                file_metadata["end_time"] = pattern_end_time.search(
                    metadata_block[2]
                ).group(1)
                file_metadata["n_seizures"] = int(
                    pattern_seizures.search(metadata_block[3]).group(1)
                )
                file_metadata["channel_names"] = self._channel_names
                file_metadata["sampling_rate"] = self.get_sampling_rate()
                seizure_intervals = []
                for i in range(file_metadata["n_seizures"]):
                    seizure_start = int(
                        pattern_seizure_start.search(metadata_block[4 + i * 2]).group(1)
                    )
                    seizure_end = int(
                        pattern_seizure_end.search(metadata_block[4 + i * 2 + 1]).group(
                            1
                        )
                    )
                    seizure_intervals.append((seizure_start, seizure_end))
                file_metadata["seizure_intervals"] = seizure_intervals
                output_metadata[filename] = file_metadata
        else:
            import warnings

            warnings.warn(
                "Block didn't follow the pattern for a metadata file block", Warning
            )
            # Check channel names
            try:
                self._channel_names = self._parse_channel_names(
                    "\n".join(metadata_block)
                )
            except Exception as e:
                print("Failed to parse block as a channel names block")
                raise e
        return output_metadata

    def _parse_file_metadata(self, seizure_file_blocks):
        """
        Parse the file metadata list blocks to get the seizure intervals
        Note: These are not necessarily in file order, so always check against the filename before continuing.
        :param seizure_file_blocks: List of seizure file blocks
        """
        output_metadata = OrderedDict()
        for block in seizure_file_blocks:
            lines = block.split("\n")
            output_metadata = self._parse_metadata(lines, output_metadata)

        return output_metadata

    def get_sampling_rate(self):
        """
        Gets the sampling rate
        """
        return self._frequency

    def get_channel_names(self, filename):
        """
        Return the channel names
        """
        return self._metadata_store[filename]["channel_names"]

    def get_seizure_list(self):
        """
        Get list of seizure intervals for each file
        """
        return [
            metadata["seizure_intervals"]
            for filename, metadata in self._metadata_store.items()
        ]

    def get_file_metadata(self):
        """
        Get the metadata for all of the files
        """
        return self._metadata_store


class Patient:
    def __init__(self, data_path, id, verbose=None):
        self._id = id
        self._data_path = data_path

        # Change the code and add list at the beginning of the map to solve error
        self._edf_files = list(
            map(
                lambda filename: CbhEdfFileMNE(filename, self._id, verbose=verbose),
                sorted(
                    glob.glob(
                        os.path.join(
                            self._data_path,
                            "physionet.org/files/chbmit/1.0.0/chb%02d/*.edf" % self._id,
                        )
                    )
                ),
            )
        )
        self._cumulative_duration = [0]

        for file in self._edf_files[:-1]:
            self._cumulative_duration.append(
                self._cumulative_duration[-1] + file.get_file_duration()
            )

        self._duration = sum(self._cumulative_duration)

        self._seizure_list = ChbLabelWrapper(
            os.path.join(
                self._data_path,
                "physionet.org/files/chbmit/1.0.0/chb%02d/chb%02d-summary.txt"
                % (self._id, self._id),
            ),
            self._id,
        ).get_seizure_list()

        self._seizure_intervals = []

        for i, file in enumerate(self._seizure_list):
            for seizure in file:
                #                 begin = seizure[0] * self._edf_files[i].get_sampling_rate() + self._cumulative_duration[i]
                #                 end = seizure[1] * self._edf_files[i].get_sampling_rate() + self._cumulative_duration[i]
                begin = seizure[0] * self._edf_files[i].get_sampling_rate()
                end = seizure[1] * self._edf_files[i].get_sampling_rate()
                self._seizure_intervals.append((begin, end))

        # Sort the filenames to store correctly
        self._patients_metadata = [
            ("{:02d}".format(self._id), file.get_filename())
            for file in self._edf_files[:]
        ]
        self._signal_lengths = [file.get_n_data_points() for file in self._edf_files[:]]

    def get_channel_names(self):
        return self._edf_files[0].get_channel_names()

    def get_sampling_rate(self):
        return self._edf_files[0].get_sampling_rate()

    def get_eeg_data(self):
        data = []

        for i, file in enumerate(self._edf_files):
            print("Reading EEG data from file %s" % file._filename)
            #             if not i:
            #                 data = file.get_data()
            #             else:
            #                 data = np.vstack((data, file.get_data()))
            data.append(file.get_data())

        return data

    def get_seizures(self):
        return self._seizure_list

    # New Version
    def get_patients_metadata(self):
        return self._patients_metadata

    # New Version
    def get_patients_signal_lengths(self):
        return self._signal_lengths

    # New Version
    def get_seizures_datapoint(self) -> list:
        output = []

        for i, file in enumerate(self.get_seizures()):
            temp = []
            if len(file) != 0:
                for seizure in file:
                    temp.append(
                        (
                            int(seizure[0] * self._edf_files[i].get_sampling_rate()),
                            int(seizure[1] * self._edf_files[i].get_sampling_rate()),
                        )
                    )
            else:
                # Do nothing
                pass

            output.append(temp)

        return output

    # New Version
    def _map_tuple_integer(self, t: tuple):
        """
        Cast the double or float type to integer
        """
        return tuple(int(x) for x in t)

    def get_seizure_intervals(self):
        return list(map(lambda x: self._map_tuple_integer(x), self._seizure_intervals))

    def get_seizure_labels(self):
        labels = np.zeros(self._duration)

        for i, interval in enumerate(self._seizure_intervals):
            labels[interval[0] : interval[1]] = 1

        return labels

    def get_seizure_clips(self):
        clips = []
        data = self.get_eeg_data()
        labels = self.get_seizure_labels()

        for i in range(len(self._seizure_intervals)):
            if not i:
                left = 0
            else:
                left = (
                    self._seizure_intervals[i - 1][1] + self._seizure_intervals[i][0]
                ) / 2
            if i == len(self._seizure_intervals) - 1:
                right = -1
            else:
                right = (
                    self._seizure_intervals[i][1] + self._seizure_intervals[i + 1][0]
                ) / 2
            clips.append((data[left:right], labels[left:right]))

        return clips

    def generate_metadata(self) -> pd.DataFrame:

        inner_dict = {
            "patient_name": [],
            "record_id": [],
            "filename": [],
            "label": [],
            "start": [],
            "end": [],
        }

        seizure_intervals = self.get_seizures_datapoint()

        signal_lengths = self.get_patients_signal_lengths()

        for i, (patient_name, filename) in enumerate((self.get_patients_metadata())):

            record_id = int(re.compile("chb(\d+)_(\d+)").search(filename).group(2))
            intervals, labels = self.generate_sub_intervals(
                [signal_lengths[i]], [seizure_intervals[i]]
            )

            for interval, label in zip(intervals, labels):
                inner_dict["patient_name"].append(patient_name)
                inner_dict["record_id"].append(record_id)
                inner_dict["filename"].append(filename)
                inner_dict["label"].append(label)
                inner_dict["start"].append(interval[0])
                inner_dict["end"].append(interval[1])

        return pd.DataFrame(inner_dict)

    @staticmethod
    def generate_sub_intervals(signal_length: list, interval_seizures: list) -> list:
        # Initialize the variable
        labels = []
        intervals = []
        begin = 0

        for i, seizures in enumerate(interval_seizures):
            if len(seizures) != 0:
                # Handle if we have only one seizure on our dataset
                # Combine two "if"
                if len(seizures) == 1:
                    begin = 0
                    seizure = seizures[0]
                    # First segments
                    intervals.append((begin, seizure[0]))
                    labels.append(0)

                    # Second segments
                    intervals.append((seizure[0], seizure[1]))
                    labels.append(1)

                    # THird segments
                    intervals.append((seizure[1], signal_length[i]))
                    labels.append(0)
                # If we have more then calculate with the following algorithm change the begin and end variable at each iteration to create our segmentations
                else:
                    begin = 0

                    # Optimize the code
                    for j, seizure in enumerate(seizures):
                        # Define the end of interval
                        if j == len(seizures) - 1:
                            end = signal_length[i]
                        else:
                            end = seizures[j + 1][0]

                        # Just do for the first interval
                        if not j:
                            # First segments
                            intervals.append((begin, seizure[0]))
                            labels.append(0)

                        # Second segments
                        intervals.append((seizure[0], seizure[1]))
                        labels.append(1)

                        # THird segments
                        intervals.append((seizure[1], end))
                        labels.append(0)

                        begin = seizure[1]

            else:
                intervals.append((begin, signal_length[i]))
                labels.append(0)

        return intervals, labels
