import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from src.exception import CustomException
from src.logger import logger


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        logger.error(CustomException(e, sys))


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        logger.error(CustomException(e, sys))
        
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array) # saves numpy array in .npy file
    except Exception as e:
        logger.error(CustomException(e, sys))
    
def save_object(file_path: str, obj: object) -> None:
    """
    Serialise an object file and save the binary
    """
    logger.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj) # we serialize any object file like, and save the binary (for example in .pkl format)

        logger.info("Exited the save_object method of utils")

    except Exception as e:
        logger.error(CustomException(e, sys))