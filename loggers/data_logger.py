# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Callable, List, Dict
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.types import DataFrame

from configs.main_config import MainConfig

from datetime import datetime
import pkg_resources
import subprocess
import numpy as np
import importlib
import json
import uuid
import os
import sys


from mpi4py import MPI
import h5py





class DataLogger:
    """ Parallel trajectories Logger using HDF5: https://docs.h5py.org
    """
    def __init__(self, 
                config: MainConfig
                ) -> None:
        self.config = config
        self._pause = True
        self._data_frames = []

        self.configure_paths()


    def configure_paths(self) -> None:
        """Configure paths for logging, filename format: init_log_name + curr_datetime + uid
        """
        uid = str(uuid.uuid4())

        curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = '_'.join([self.config.log_name, curr_datetime, uid, '.h5'])
        self.log_path = os.path.join(self.config.log_folder_path, log_name)

        return 


    def add_data(self, data: dict) -> None:
        """ Write data paraller to the log files 

        Args:
            data (dict): Dictionary representing the data to be logged at this time index.

        """
        # h5py.File(self.log_path, 'w', driver='mpio', comm=MPI.COMM_WORLD )for parallel data preprocessing

        if not os.path.isfile(self.log_path): # log does not exis 
            with h5py.File(self.log_path, 'w') as file:

                for key, value in data.items():
                    if isinstance(value, int) or isinstance(value, float) == int:
                        shape = (1,)
                        maxshape = (None,)
                    elif isinstance(value, np.ndarray):
                        shape = (1,) + value.shape
                        maxshape = (None,) + value.shape
                    elif isinstance(value, list) or isinstance(value, tuple):
                        shape =  (1,) + np.array(value).shape
                        maxshape = (None,) + value.shape

                    file.create_dataset(name = key, shape = shape,  maxshape = maxshape)
                    file[key][0] = value # add fist frame
        else:
            with h5py.File(self.log_path, 'a') as file:
                for key, value in data.items():
                    
                    dataset = file[key]

                    # resize for new element for element 
                    if isinstance(value, int) or isinstance(value, float):
                        num_values = dataset.shape[0]
                        new_shape = (num_values + 1,)

                        dataset.resize(new_shape)
                    elif isinstance(value, np.ndarray):
                        old_shape = dataset.shape
                        num_values = dataset.shape[0] 
                        new_shape = (num_values + 1,) + dataset.shape[1:]

                        dataset.resize(new_shape)

                    dataset[num_values] = value

        return

    def get_num_of_data_frames(self) -> int:
        """

        Returns:
            int: the number of data frames collected/ retrieved in the data logger.
        """

        return len(self._data_frames)

    def pause(self) -> None:
        """Pauses data collection.
        """
        self._pause = True
        return

    def start(self) -> None:
        """Resumes/ starts data collection.
        """
        self._pause = False
        return

    def is_started(self) -> bool:
        """
        Returns:
            bool: True if data collection is started/ resumed. False otherwise.
        """
        return not self._pause

    def reset(self) -> None:
        """Clears the data in the logger.
        """
        self._pause = True

        return

    def load(self, log_path: str) -> None:
        """Loads data from dataset to read back a previous saved data or to resume recording data from another time step.

        Args:
            log_path (str): path of the hdf5 file to be used to load the data.
        """

        return
