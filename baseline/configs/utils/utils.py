from typing import Dict, Tuple, Union, List, Callable, Any
import json


def eeg_electrode_configs(
    conf_path: str,
) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int]]:
    config = {}
    with open(conf_path, "r") as f:
        lines = f.readlines()
        lines = "".join(lines)
        exec(lines, config)
        return config["eeg_electrode_positions"], config["eeg_electrods_plane_shape"]


def read_json_config(
    conf_path: str,
) -> List[Dict[str, Any]]:
    with open(conf_path, "r") as f:
        configs = json.load(f)
        return configs
