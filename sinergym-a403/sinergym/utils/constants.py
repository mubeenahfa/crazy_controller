"""Constants used in whole project."""

import os
from typing import List, Union

import numpy as np
import pkg_resources

# ---------------------------------------------------------------------------- #
#                               Generic constants                              #
# ---------------------------------------------------------------------------- #
# Sinergym Data path
PKG_DATA_PATH = pkg_resources.resource_filename(
    'sinergym', 'data/')
# Weekday encoding for simulations
WEEKDAY_ENCODING = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}
# Default start year (Non leap year please)
YEAR = 1991
# cwd
CWD = os.getcwd()

# Logger values (environment layer, simulator layer and modeling layer)
# LOG_ENV_LEVEL = 'INFO'
# LOG_SIM_LEVEL = 'INFO'
# LOG_MODEL_LEVEL = 'INFO'
# LOG_WRAPPERS_LEVEL = 'INFO'
# LOG_REWARD_LEVEL = 'INFO'
# LOG_COMMON_LEVEL = 'INFO'
# LOG_CALLBACK_LEVEL = 'INFO'
LOG_ENV_LEVEL = 'WARNING'
LOG_SIM_LEVEL = 'WARNING'
LOG_MODEL_LEVEL = 'WARNING'
LOG_WRAPPERS_LEVEL = 'WARNING'
LOG_REWARD_LEVEL = 'WARNING'
LOG_COMMON_LEVEL = 'WARNING'
LOG_CALLBACK_LEVEL = 'WARNING'
# LOG_FORMAT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"
LOG_FORMAT = "[%(name)s] (%(levelname)s) : %(message)s"

# ---------------------------------------------------------------------------- #
#              Custom Eplus discrete environments action mappings             #
# ---------------------------------------------------------------------------- #

# --------------------------------------A403----------------------------------- #
# -------------------------- ACTION MAPPINGS -------------------------- #
A403_MAPPINGS = {
    "FULL WINDOW FAN CONTROL": { #Window fan has 0.5,0.75 and 1.0 speeds.
        0 : [19, 21, 0.5, 0.0],
        1 : [19, 21, 0.5, 0.5],
        2 : [19, 21, 0.5, 0.75],
        3 : [19, 21, 0.5, 1.0],
        4 : [19, 21, 0.75, 0.0],
        5 : [19, 21, 0.75, 0.5],
        6 : [19, 21, 0.75, 0.75],
        7 : [19, 21, 0.75, 1.0],
        8 : [19, 21, 1.0, 0.0],
        9 : [19, 21, 1.0, 0.5],
        10 : [19, 21, 1.0, 0.75],
        11 : [19, 21, 1.0, 1.0],
        12 : [20, 23, 0.5, 0.0],
        13 : [20, 23, 0.5, 0.5],
        14 : [20, 23, 0.5, 0.75],
        15 : [20, 23, 0.5, 1.0],
        16 : [20, 23, 0.75, 0.0],
        17 : [20, 23, 0.75, 0.5],
        18 : [20, 23, 0.75, 0.75],
        19 : [20, 23, 0.75, 1.0],
        20 : [20, 23, 1.0, 0.0],
        21 : [20, 23, 1.0, 0.5],
        22 : [20, 23, 1.0, 0.75],
        23 : [20, 23, 1.0, 1.0],
        24 : [23, 26, 0.5, 0.0],
        25 : [23, 26, 0.5, 0.5],
        26 : [23, 26, 0.5, 0.75],
        27 : [23, 26, 0.5, 1.0],
        28 : [23, 26, 0.75, 0.0],
        29 : [23, 26, 0.75, 0.5],
        30 : [23, 26, 0.75, 0.75],
        31 : [23, 26, 0.75, 1.0],
        32 : [23, 26, 1.0, 0.0],
        33 : [23, 26, 1.0, 0.5], 
        34 : [23, 26, 1.0, 0.75],
        35 : [23, 26, 1.0, 1.0],
        36 : [5 , 50, 0.0, 0.0], # OFF ACTION FOR HVAC AND WINDOW FAN
        37 : [5 , 50, 0.0, 0.5],
        38 : [5 , 50, 0.0, 0.75],
        39 : [5 , 50, 0.0, 1.0]
    },
    "ONLY FAN":{
            0: [ 5, 50, 0.0, 0.0],
            1: [ 5, 50, 0.0, 0.5],
            2: [ 5, 50, 0.0, 0.75],
            3: [ 5, 50, 0.0, 1.0],
    },
    "ONLY HVAC":{
        0 : [21, 22, 1.0, 0.0],
        1 : [22, 23, 1.0, 0.0],
        2 : [23, 24, 1.0, 0.0],
        3 : [24, 25, 1.0, 0.0],
        4 : [25, 26, 1.0, 0.0],
        5 : [26, 27, 1.0, 0.0],
        6 : [27, 28, 1.0, 0.0],
        7 : [28, 29, 1.0, 0.0],
        8 : [29, 30, 1.0, 0.0],
        9 : [5 , 50, 0.0, 0.0],  
    },   
    "CLASSIC WINDOW 0.5":{
        0 : [19,21,0.5,0.0],
        1 : [19,21,0.5,0.5],
        2 : [19,21,0.75,0.0],
        3 : [19,21,0.75,0.5],
        4 : [19,21,1.0,0.0],
        5 : [19,21,1.0,0.5],
        6 : [21,23,0.5,0.0],
        7 : [21,23,0.5,0.5],
        8 : [21,23,0.75,0.0],
        9 : [21,23,0.75,0.5],
        10 : [21,23,1.0,0.0],
        11 : [21,23,1.0,0.5],
        12 : [23,26,0.5,0.0],
        13 : [23,26,0.5,0.5],
        14 : [23,26,0.75,0.0],
        15 : [23,26,0.75,0.5],
        16 : [23,26,1.0,0.0],
        17 : [23,26,1.0,0.5],
        18 : [5,50,0.0,0.0],
        19 : [5,50,0.0,0.5]
    },
    "CLASSIC WINDOW 0.25":{
        0 : [19,21,0.5,0.0],
        1 : [19,21,0.5,0.25],
        2 : [19,21,0.75,0.0],
        3 : [19,21,0.75,0.25],
        4 : [19,21,1.0,0.0],
        5 : [19,21,1.0,0.25],
        6 : [21,23,0.5,0.0],
        7 : [21,23,0.5,0.25],
        8 : [21,23,0.75,0.0],
        9 : [21,23,0.75,0.25],
        10 : [21,23,1.0,0.0],
        11 : [21,23,1.0,0.25],
        12 : [23,26,0.5,0.0],
        13 : [23,26,0.5,0.25],
        14 : [23,26,0.75,0.0],
        15 : [23,26,0.75,0.25],
        16 : [23,26,1.0,0.0],
        17 : [23,26,1.0,0.25],
        18 : [5,50,0.0,0.0],
        19 : [5,50,0.0,0.25]
    },
    "WINDOW FAN SPEED CONTROL":{ # Used for all baselines
        0 : [19, 21, 0.5, 0.0],
        1 : [19, 21, 0.5, 0.5],
        2 : [19, 21, 0.5, 0.75],
        3 : [19, 21, 0.5, 1.0],
        4 : [19, 21, 0.75, 0.0],
        5 : [19, 21, 0.75, 0.5],
        6 : [19, 21, 0.75, 0.75],
        7 : [19, 21, 0.75, 1.0],
        8 : [19, 21, 1.0, 0.0],
        9 : [19, 21, 1.0, 0.5],
        10 : [19, 21, 1.0, 0.75],
        11 : [19, 21, 1.0, 1.0],
        12 : [21, 23, 0.5, 0.0],
        13 : [21, 23, 0.5, 0.5],
        14 : [21, 23, 0.5, 0.75],
        15 : [21, 23, 0.5, 1.0],
        16 : [21, 23, 0.75, 0.0],
        17 : [21, 23, 0.75, 0.5],
        18 : [21, 23, 0.75, 0.75],
        19 : [21, 23, 0.75, 1.0],
        20 : [21, 23, 1.0, 0.0],
        21 : [21, 23, 1.0, 0.5],
        23 : [21, 23, 1.0, 0.75],
        23 : [21, 23, 1.0, 1.0],
        24 : [23, 26, 0.5, 0.0],
        25 : [23, 26, 0.5, 0.5],
        26 : [23, 26, 0.5, 0.75],
        27 : [23, 26, 0.5, 1.0],
        28 : [23, 26, 0.75, 0.0],
        29 : [23, 26, 0.75, 0.5],
        30 : [23, 26, 0.75, 0.75],
        31 : [23, 26, 0.75, 1.0],
        32 : [23, 26, 1.0, 0.0],
        33 : [23, 26, 1.0, 0.5],
        34 : [23, 26, 1.0, 0.75],
        35 : [23, 26, 1.0, 1.0],
        36: [5,50,0.0,0.0], # OFF ACTION FOR HVAC AND WINDOW FAN
        37: [5,50,0.0,0.5],
        38: [5,50,0.0,0.75],
        39: [5,50,0.0,1.0]

    },
    "PMV_MODEL_FULL":{
        0 : [21, 22, 0.5, 0.0],
        1 : [21, 22, 0.5, 0.5],
        2 : [21, 22, 0.5, 0.75],
        3 : [21, 22, 0.5, 1.0],
        4 : [21, 22, 0.75, 0.0],
        5 : [21, 22, 0.75, 0.5],
        6 : [21, 22, 0.75, 0.75],
        7 : [21, 22, 0.75, 1.0],
        8 : [21, 22, 1.0, 0.0],
        9 : [21, 22, 1.0, 0.5],
        10 : [21, 22, 1.0, 0.75],
        11 : [21, 22, 1.0, 1.0],
        12 : [22, 23, 0.5, 0.0],
        13 : [22, 23, 0.5, 0.5],
        14 : [22, 23, 0.5, 0.75],
        15 : [22, 23, 0.5, 1.0],
        16 : [22, 23, 0.75, 0.0],
        17 : [22, 23, 0.75, 0.5],
        18 : [22, 23, 0.75, 0.75],
        19 : [22, 23, 0.75, 1.0],
        20 : [22, 23, 1.0, 0.0],
        21 : [22, 23, 1.0, 0.5],
        22 : [22, 23, 1.0, 0.75],
        23 : [22, 23, 1.0, 1.0],
        24 : [23, 24, 0.5, 0.0],
        25 : [23, 24, 0.5, 0.5],
        26 : [23, 24, 0.5, 0.75],
        27 : [23, 24, 0.5, 1.0],
        28 : [23, 24, 0.75, 0.0],
        29 : [23, 24, 0.75, 0.5],
        30 : [23, 24, 0.75, 0.75],
        31 : [23, 24, 0.75, 1.0],
        32 : [23, 24, 1.0, 0.0],
        33 : [23, 24, 1.0, 0.5],
        34 : [23, 24, 1.0, 0.75],
        35 : [23, 24, 1.0, 1.0],
        36 : [24, 25, 0.5, 0.0],
        37 : [24, 25, 0.5, 0.5],
        38 : [24, 25, 0.5, 0.75],
        39 : [24, 25, 0.5, 1.0],
        40 : [24, 25, 0.75, 0.0],
        41 : [24, 25, 0.75, 0.5],
        42 : [24, 25, 0.75, 0.75],
        43 : [24, 25, 0.75, 1.0],
        44 : [24, 25, 1.0, 0.0],
        45 : [24, 25, 1.0, 0.5],
        46 : [24, 25, 1.0, 0.75],
        47 : [24, 25, 1.0, 1.0],
        48 : [25, 26, 0.5, 0.0],
        49 : [25, 26, 0.5, 0.5],
        50 : [25, 26, 0.5, 0.75],
        51 : [25, 26, 0.5, 1.0],
        52 : [25, 26, 0.75, 0.0],
        53 : [25, 26, 0.75, 0.5],
        54 : [25, 26, 0.75, 0.75],
        55 : [25, 26, 0.75, 1.0],
        56 : [25, 26, 1.0, 0.0],
        57 : [25, 26, 1.0, 0.5],
        58 : [25, 26, 1.0, 0.75],
        59 : [25, 26, 1.0, 1.0],
        60 : [5 , 50, 0.0, 0.0], 
        61 : [5 , 50, 0.0, 0.5],
        62 : [5 , 50, 0.0, 0.75],
        63 : [5 , 50, 0.0, 1.0]
    },
    "PMV_MODEL_MULTI_FAN":{
        0 : [21, 22, 1.0, 0.0],
        1 : [21, 22, 1.0, 0.5],
        2 : [21, 22, 1.0, 0.75],
        3 : [21, 22, 1.0, 1.0],
        4 : [22, 23, 1.0, 0.0],
        5 : [22, 23, 1.0, 0.5],
        6 : [22, 23, 1.0, 0.75],
        7 : [22, 23, 1.0, 1.0],
        8 : [23, 24, 1.0, 0.0],
        9 : [23, 24, 1.0, 0.5],
        10 : [23, 24, 1.0, 0.75],
        11 : [23, 24, 1.0, 1.0],
        12 : [24, 25, 1.0, 0.0],
        13 : [24, 25, 1.0, 0.5],
        14 : [24, 25, 1.0, 0.75],
        15 : [24, 25, 1.0, 1.0],
        16 : [25, 26, 1.0, 0.0],
        17 : [25, 26, 1.0, 0.5],
        18 : [25, 26, 1.0, 0.75],
        19 : [25, 26, 1.0, 1.0],
        20 : [26, 27, 1.0, 0.0],
        21 : [26, 27, 1.0, 0.5],
        22 : [26, 27, 1.0, 0.75],
        23 : [26, 27, 1.0, 1.0],
        24 : [27, 28, 1.0, 0.0],
        25 : [27, 28, 1.0, 0.5],
        26 : [27, 28, 1.0, 0.75],
        27 : [27, 28, 1.0, 1.0],
        28 : [28, 29, 1.0, 0.0],
        29 : [28, 29, 1.0, 0.5],
        30 : [28, 29, 1.0, 0.75],
        31 : [28, 29, 1.0, 1.0],
        32 : [29, 30, 1.0, 0.0],
        33 : [29, 30, 1.0, 0.5],
        34 : [29, 30, 1.0, 0.75],
        35 : [29, 30, 1.0, 1.0],
        36 : [5 , 50, 0.0, 0.0],
        37 : [5 , 50, 0.0, 0.5],
        38 : [5 , 50, 0.0, 0.75],
        39 : [5 , 50, 0.0, 1.0]
    },
    
}

# ---------------------COMBINED ACTION KEY MAPPER FOR MULTI AGENT CASE--------------------------------- #
# Action dictionary
reduced_actions = {
    0: [21, 23, 1.0, 0.0],
    1: [21, 23, 1.0, 1.0],
    2: [23, 26, 1.0, 0.0],
    3: [23, 26, 1.0, 1.0],
    4: [5, 50, 0.0, 0.0],
    5: [5, 50, 0.0, 1.0],
}

fan_map = {
    0: 0.0,   # Off
    1: 0.25,  # Low
    2: 0.5,   # Medium
    3: 1.0    # High
}

hvac_map = {
    0: [5, 50, 0.0],     # Off
    1: [23, 26, 1.0],    # Summer
    2: [21, 23, 1.0]     # Winter
}

from itertools import product

def generate_combined_action_dict(fan_map: dict, hvac_map: dict) -> dict:
    combined_actions = {}
    idx = 0

    for hvac_action, fan_action in product(hvac_map.values(), fan_map.values()):
        t_min, t_max, hvac_flag = hvac_action
        fan_flag = fan_action
        combined_actions[idx] = [t_min, t_max, hvac_flag, fan_flag]
        idx += 1

    return combined_actions

def get_combined_action_key(fan_action: int, 
                            hvac_action: int, 
                            action_dict: dict, 
                            fan_map: dict, 
                            hvac_map: dict) -> int:
    """
    Returns the key from the action_dict that matches the combination of fan and hvac actions.

    :param fan_action: int, key from fan_map (e.g. 0 or 1)
    :param hvac_action: int, key from hvac_map (e.g. 0, 1, 2)
    :param action_dict: dict, e.g. {0: [21, 23, 1.0, 0.0], ...}
    :param fan_map: dict, e.g. {0: 0.0, 1: 1.0}
    :param hvac_map: dict, e.g. {0: [5, 50, 0.0], 1: [23, 26, 1.0], 2: [21, 23, 1.0]}
    :return: int, key from action_dict
    """
    if fan_action not in fan_map:
        raise ValueError(f"Invalid fan_action: {fan_action}")
    if hvac_action not in hvac_map:
        raise ValueError(f"Invalid hvac_action: {hvac_action}")

    # Extract action representation
    fan_flag = fan_map[fan_action]
    t_min, t_max, hvac_flag = hvac_map[hvac_action]

    target_action = [t_min, t_max, hvac_flag, fan_flag]

    # Search for matching action in dict
    for key, value in action_dict.items():
        if value == target_action:
            return key

    raise ValueError("Combined action not found in the provided dictionary.")





# -------------------------- MAPPING FUNCTION -------------------------- #
def get_a403_action_mapping(env_type: str, action) -> List[float]:
    """Generic mapper to resolve action index to control values."""
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = A403_MAPPINGS.get(env_type)
    if mapping is None:
        raise ValueError(f"Unknown environment type: {env_type}")

    if action not in mapping:
        raise IndexError(f"Invalid action {action} for mapping {env_type}")

    return mapping[action]
# ---------------------------------------------------------------------------- #
#              Default Eplus discrete environments action mappings             #
# ---------------------------------------------------------------------------- #

# -------------------------------------5ZONE---------------------------------- #

#ACITON_CONFIG_NAME = "ONLY HVAC"
ACITON_CONFIG_NAME = "PMV_MODEL_MULTI_FAN"
#ACITON_CONFIG_NAME = "WINDOW FAN SPEED CONTROL"
def DEFAULT_A403SMALLFANGER_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)
def DEFAULT_A403LARGEFANGER_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)
def DEFAULT_A403MEDIUMFANGER_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_A403MEDIUMWINDOW_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_A403SMALL_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_A403MEDIUM_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_A403LARGE_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_A403NEW_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_A403V3_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_A403_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)
# -------------------------------------5ZONE---------------------------------- #
def DEFAULT_A403_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping(ACITON_CONFIG_NAME, action)

def DEFAULT_5ZONE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [12, 30],
        1: [13, 30],
        2: [14, 29],
        3: [15, 28],
        4: [16, 28],
        5: [17, 27],
        6: [18, 26],
        7: [19, 25],
        8: [20, 24],
        9: [21, 23.25]
    }

    return mapping[action]


# ----------------------------------DATACENTER--------------------------------- #

def DEFAULT_DATACENTER_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# ----------------------------------WAREHOUSE--------------------------------- #


def DEFAULT_WAREHOUSE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# ----------------------------------OFFICE--------------------------------- #


def DEFAULT_OFFICE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# ----------------------------------OFFICEGRID---------------------------- #


def DEFAULT_OFFICEGRID_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30, 0.0, 0.0],
        1: [16, 29, 0.0, 0.0],
        2: [17, 28, 0.0, 0.0],
        3: [18, 27, 0.0, 0.0],
        4: [19, 26, 0.0, 0.0],
        5: [20, 25, 0.0, 0.0],
        6: [21, 24, 0.0, 0.0],
        7: [22, 23, 0.0, 0.0],
        8: [22, 22.5, 0.0, 0.0],
        9: [21, 22.5, 0.0, 0.0]
    }

    return mapping[action]

# ----------------------------------SHOP--------------------- #


def DEFAULT_SHOP_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# -------------------------------- AUTOBALANCE ------------------------------- #


def DEFAULT_RADIANT_DISCRETE_FUNCTION(
        action: Union[np.ndarray, List[int]]) -> List[float]:
    action[5] += 25
    return list(action)