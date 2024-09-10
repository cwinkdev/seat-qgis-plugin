#!/usr/bin/python

# pylint: disable=too-many-arguments


"""
/***************************************************************************.

 velocity_module.py
 Copyright 2023, Integral Consulting Inc. All rights reserved.

 PURPOSE: module for calcualting velocity (larval motility) change from a velocity stressor

 PROJECT INFORMATION:
 Name: SEAT - Spatial and Environmental Assessment Toolkit
 Number: C1308

 AUTHORS
  Timothy Nelson (tnelson@integral-corp.com)
  Sam McWilliams (smcwilliams@integral-corp.com)
  Eben Pendelton

 NOTES (Data descriptions and any script specific notes)
	1. called by stressor_receptor_calc.py
"""

import os
from typing import Optional, Tuple, List, Dict, NamedTuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from seat.modules.stressor_utils import (
    estimate_grid_spacing,
    create_structured_array_from_unstructured,
    calc_receptor_array,
    trim_zeros,
    create_raster,
    numpy_array_to_raster,
    bin_layer,
    classify_layer_area,
    classify_layer_area_2nd_constraint,
    resample_structured_grid,
    secondary_constraint_geotiff_to_numpy,
)


@dataclass
class PreparedData:
    """
    Data structure to store prepared velocity stressor data.

    Attributes
    ----------
    gridtype : str
        Type of grid ('structured' or 'unstructured').
    xcor : np.ndarray
        X-coordinates array.
    ycor : np.ndarray
        Y-coordinates array.
    mag_nodev : np.ndarray
        Magnitude data array for 'no device' runs.
    mag_dev : np.ndarray
        Magnitude data array for 'with device' runs.
    data_frame : pd.DataFrame
        DataFrame containing file and run information if multiple files are found; otherwise, None.
    """

    gridtype: str
    xcor: np.ndarray
    ycor: np.ndarray
    mag_nodev: np.ndarray
    mag_dev: np.ndarray
    data_frame: pd.DataFrame


@dataclass
class MotilityAndDifferences:
    """
    Data structure to store motility differences and critical velocities.

    Attributes
    ----------
    mag_diff : np.ndarray
        Magnitude differences between 'with device' and 'no device' runs.
    motility_nodev : np.ndarray
        Motility data array for 'no device' runs.
    motility_dev : np.ndarray
        Motility data array for 'with device' runs.
    motility_diff : np.ndarray
        Differences in motility between 'with device' and 'no device' runs.
    velcrit : np.ndarray
        Critical velocity values.
    """

    mag_diff: NDArray[np.float64]
    motility_nodev: NDArray[np.float64]
    motility_dev: NDArray[np.float64]
    motility_diff: NDArray[np.float64]
    velcrit: NDArray[np.float64]


@dataclass
class StructuredArrays:
    """
    Data structure to store arrays for structured grids after processing.

    Attributes
    ----------
    mag_diff_struct : np.ndarray
        Structured array of magnitude differences between 'with device' and 'no device' runs.
    mag_combined_dev_struct : np.ndarray
        Structured array of combined magnitudes for 'with device' runs.
    mag_combined_nodev_struct : np.ndarray
        Structured array of combined magnitudes for 'no device' runs.
    motility_nodev_struct : np.ndarray
        Structured array of motility for 'no device' runs.
    motility_dev_struct : np.ndarray
        Structured array of motility for 'with device' runs.
    motility_diff_struct : np.ndarray
        Structured array of motility differences.
    velcrit_struct : np.ndarray
        Structured array of critical velocities.
    """

    mag_diff_struct: np.ndarray
    mag_combined_dev_struct: np.ndarray
    mag_combined_nodev_struct: np.ndarray
    motility_nodev_struct: np.ndarray
    motility_dev_struct: np.ndarray
    motility_diff_struct: np.ndarray
    velcrit_struct: np.ndarray


class GridData(NamedTuple):
    """
    Represents grid data for velocity stressor calculations.

    Attributes
    ----------
    rx : np.ndarray
        X-coordinates array for the grid.
    ry : np.ndarray
        Y-coordinates array for the grid.
    dx : float
        X-coordinate spacing.
    dy : float
        Y-coordinate spacing.
    gridtype : str
        Type of grid ('structured' or 'unstructured').
    crs : int
        Coordinate Reference System (CRS) identifier.
    """

    rx: NDArray[np.float64]
    ry: NDArray[np.float64]
    dx: float
    dy: float
    gridtype: str
    crs: int


class FileProcessor:
    """
    Handles file validation, listing, and loading operations for NetCDF files.

    Provides methods to validate directories, list NetCDF files, and
    load and sort files for 'no device'
    and 'with device' runs.
    """

    @staticmethod
    def validate_and_list_files(
        fpath_nodev: str, fpath_dev: str
    ) -> Tuple[List[str], List[str]]:
        """
        Validate the directories and list NetCDF files in them.

        Parameters
        ----------
        fpath_nodev : str
            Directory path to the baseline/no device model run NetCDF files.
        fpath_dev : str
            Directory path to the with device model run NetCDF files.

        Returns
        -------
        files_nodev : list
            List of filenames for the 'no device' runs.
        files_dev : list
            List of filenames for the 'with device' runs.
        """
        if not os.path.exists(fpath_nodev):
            raise FileNotFoundError(f"The directory {fpath_nodev} does not exist.")
        if not os.path.exists(fpath_dev):
            raise FileNotFoundError(f"The directory {fpath_dev} does not exist.")

        files_nodev = [i for i in os.listdir(fpath_nodev) if i.endswith(".nc")]
        files_dev = [i for i in os.listdir(fpath_dev) if i.endswith(".nc")]

        return files_nodev, files_dev

    @staticmethod
    def load_and_sort_files(fpath_nodev: str, fpath_dev: str) -> PreparedData:
        """
        Load and sort NetCDF files for 'no device' and 'with device' runs and
        process their data.

        Parameters
        ----------
        fpath_nodev : str
            Directory path to the baseline/no device model run NetCDF files.
        fpath_dev : str
            Directory path to the with device model run NetCDF files.

        Returns
        -------
        gridtype : str
            Grid type ('structured' or 'unstructured').
        xcor : np.ndarray
            X-coordinates array.
        ycor : np.ndarray
            Y-coordinates array.
        mag_nodev : np.ndarray
            Magnitude data array for 'no device' runs.
        mag_dev : np.ndarray
            Magnitude data array for 'with device' runs.
        data_frame : pd.DataFrame or None
            DataFrame containing file and run information if multiple
            files are found, otherwise None.
        """

        files_nodev, files_dev = file_processor.validate_and_list_files(
            fpath_nodev, fpath_dev
        )

        if len(files_nodev) == 1 and len(files_dev) == 1:
            gridtype, xcor, ycor, mag_nodev, mag_dev = (
                grid_processor.process_single_file_case(
                    files_nodev, files_dev, fpath_nodev, fpath_dev
                )
            )
            data_frame = None  # No DataFrame needed for a single file case
        elif len(files_nodev) == len(files_dev):
            prepared_data = grid_processor.process_multiple_files_case(
                files_nodev, files_dev, fpath_nodev, fpath_dev
            )
            gridtype = prepared_data.gridtype
            xcor = prepared_data.xcor
            ycor = prepared_data.ycor
            mag_nodev = prepared_data.mag_nodev
            mag_dev = prepared_data.mag_dev
            data_frame = prepared_data.data_frame
        else:
            raise ValueError(
                f"Number of device runs ({len(files_dev)}) must be the same as no device runs "
                f"({len(files_nodev)})."
            )

        return PreparedData(gridtype, xcor, ycor, mag_nodev, mag_dev, data_frame)


class GridProcessor:
    """
    Processes grid-related operations for structured and unstructured grids.

    Provides methods for checking grid types, processing single or multiple files,
    estimating grid spacing,
    and converting unstructured data to structured arrays.
    """

    @staticmethod
    def check_grid_define_vars(dataset: Dataset) -> tuple[str, str, str, str, str]:
        """
        Determins the type of grid and corresponding velocity variable name and coordiante names

        Parameters
        ----------
        dataset : netdcf (.nc) dataset
            netdcf (.nc) dataset.

        Returns
        -------
        gridtype : string
            "structured" or "unstructured".
        xvar : str
            name of x-coordinate variable.
        yvar : str
            name of y-coordiante variable.
        uvar : str
            name of x-coordinate velocity variable.
        vvar : str
            name of y-coordinate velocity variable.
        """
        data_vars = list(dataset.variables)
        if "U1" in data_vars:
            gridtype = "structured"
            uvar = "U1"
            vvar = "V1"
            try:
                xvar, yvar = dataset.variables[uvar].coordinates.split()
            except AttributeError:
                xvar = "XCOR"
                yvar = "YCOR"
        else:
            gridtype = "unstructured"
            uvar = "ucxa"
            vvar = "ucya"
            xvar, yvar = dataset.variables[uvar].coordinates.split()
        return gridtype, xvar, yvar, uvar, vvar

    @staticmethod
    def process_single_file_case(
        files_nodev: List[str],
        files_dev: List[str],
        fpath_nodev: str,
        fpath_dev: str,
    ) -> Tuple[
        str,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Process the single file case for both 'no device' and 'with device' runs.

        Parameters
        ----------
        files_nodev : list
            List of filenames for the 'no device' runs.
        files_dev : list
            List of filenames for the 'with device' runs.
        fpath_nodev : str
            Directory path to the baseline/no device model run NetCDF files.
        fpath_dev : str
            Directory path to the with device model run NetCDF files.

        Returns
        -------
        gridtype : str
            Grid type ('structured' or 'unstructured').
        xcor : np.ndarray
            X-coordinates array.
        ycor : np.ndarray
            Y-coordinates array.
        mag_nodev : np.ndarray
            Magnitude data array for 'no device' runs.
        mag_dev : np.ndarray
            Magnitude data array for 'with device' runs.
        """

        # Helper function to read variables and compute magnitudes
        def read_and_compute_magnitude(
            file_path: str,
            file_name: str,
            xvar: str,
            yvar: str,
            uvar: str,
            vvar: str,
        ):
            with Dataset(os.path.join(file_path, file_name)) as dataset:
                xcor = dataset.variables[xvar][:].data
                ycor = dataset.variables[yvar][:].data
                u = dataset.variables[uvar][:].data
                v = dataset.variables[vvar][:].data
                magnitude = np.sqrt(u**2 + v**2)
            return xcor, ycor, magnitude

        # Read grid definitions and variables

        with Dataset(os.path.join(fpath_dev, files_dev[0])) as file_dev_present:
            (gridtype, xvar, yvar, uvar, vvar) = grid_processor.check_grid_define_vars(
                file_dev_present
            )

        # Read and compute magnitudes for 'with device' and 'no device' runs
        _, _, mag_dev = read_and_compute_magnitude(
            fpath_dev, files_dev[0], xvar, yvar, uvar, vvar
        )
        xcor, ycor, mag_nodev = read_and_compute_magnitude(
            fpath_nodev, files_nodev[0], xvar, yvar, uvar, vvar
        )

        return gridtype, xcor, ycor, mag_nodev, mag_dev

    @staticmethod
    def process_multiple_files_case(
        files_nodev: List[str],
        files_dev: List[str],
        fpath_nodev: str,
        fpath_dev: str,
    ) -> PreparedData:
        """
        Process the multiple files case for both 'no device' and 'with device' runs.

        Parameters
        ----------
        files_nodev : list
            List of filenames for the 'no device' runs.
        files_dev : list
            List of filenames for the 'with device' runs.
        fpath_nodev : str
            Directory path to the baseline/no device model run NetCDF files.
        fpath_dev : str
            Directory path to the with device model run NetCDF files.

        Returns
        -------
        PreparedData
            An object containing grid type, coordinates, magnitudes, and file/run information.
        """
        # Prepare and sort run data
        data_frame = DataPreparation.prepare_run_data(files_nodev, files_dev)

        # Read the first file to define grid type and coordinates
        with Dataset(
            os.path.join(fpath_dev, data_frame["files_dev"].iloc[0])
        ) as dataset:
            gridtype, xvar, yvar, uvar, vvar = grid_processor.check_grid_define_vars(
                dataset
            )

            # Directly extract coordinates from the dataset
            xcor, ycor = (
                dataset.variables[xvar][:].data,
                dataset.variables[yvar][:].data,
            )

        # Compute magnitudes for 'no device' and 'with device' runs
        mag_nodev, mag_dev = DataPreparation.compute_all_magnitudes(
            data_frame, fpath_nodev, fpath_dev, uvar, vvar
        )

        # Return prepared data directly
        return PreparedData(gridtype, xcor, ycor, mag_nodev, mag_dev, data_frame)

    @staticmethod
    def estimate_grid_spacing_for_unstructured(
        xcor: NDArray[np.float64], ycor: NDArray[np.float64], nsamples: int = 100
    ) -> float:
        """
        Estimate grid spacing for unstructured grid data.

        Parameters
        ----------
        xcor : np.ndarray
            X-coordinates array.
        ycor : np.ndarray
            Y-coordinates array.
        nsamples : int
            Number of samples to use for estimation.

        Returns
        -------
        float
            Estimated grid spacing.
        """
        return estimate_grid_spacing(xcor, ycor, nsamples=nsamples)

    @staticmethod
    def create_structured_arrays(
        xcor: NDArray[np.float64],
        ycor: NDArray[np.float64],
        dxdy: float,
        *arrays: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[NDArray[np.float64]]]:
        """
        Create structured arrays from unstructured data.

        Parameters
        ----------
        xcor : np.ndarray
            X-coordinates array.
        ycor : np.ndarray
            Y-coordinates array.
        dxdy : float
            Grid spacing.
        arrays : list of np.ndarray
            Arrays to convert to structured format.

        Returns
        -------
        rx : np.ndarray
            Structured X-coordinates array.
        ry : np.ndarray
            Structured Y-coordinates array.
        structured_arrays : list of np.ndarray
            List of structured arrays.
        """
        rx, ry = None, None
        structured_arrays = []
        for array in arrays:
            # Ensure create_structured_array_from_unstructured returns structured data correctly
            rx, ry, struct_array = create_structured_array_from_unstructured(
                xcor, ycor, array, dxdy, flatness=0.2
            )
            structured_arrays.append(struct_array)
        return rx, ry, structured_arrays

    @staticmethod
    def process_structured_grid(
        xcor: NDArray[np.float64],
        ycor: NDArray[np.float64],
        mag_combined_nodev: NDArray[np.float64],
        mag_combined_dev: NDArray[np.float64],
        mag_diff: NDArray[np.float64],
        motility_nodev: NDArray[np.float64],
        motility_dev: NDArray[np.float64],
        motility_diff: NDArray[np.float64],
        velcrit: NDArray[np.float64],
    ) -> Tuple[
        Dict[str, NDArray[np.float64]],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
    ]:
        """
        Process structured grid data and prepare output arrays.

        Returns
        -------
        dict_of_arrays : dict
            Dictionary of processed arrays for structured grid.
        rx : np.ndarray
            X-coordinates array.
        ry : np.ndarray
            Y-coordinates array.
        dx : float
            X-coordinate spacing.
        dy : float
            Y-coordinate spacing.
        """
        motility_classification = vel_stress_calc.classify_motility(
            motility_dev, motility_nodev
        )
        dx = np.nanmean(np.diff(xcor[:, 0]))
        dy = np.nanmean(np.diff(ycor[0, :]))
        rx = xcor
        ry = ycor
        dict_of_arrays = {
            "velocity_magnitude_without_devices": mag_combined_nodev,
            "velocity_magnitude_with_devices": mag_combined_dev,
            "velocity_magnitude_difference": mag_diff,
            "motility_without_devices": motility_nodev,
            "motility_with_devices": motility_dev,
            "motility_difference": motility_diff,
            "motility_classified": motility_classification,
            "critical_velocity": velcrit,
        }
        return dict_of_arrays, rx, ry, dx, dy

    @staticmethod
    def process_unstructured_grid(
        xcor: NDArray[np.float64],
        ycor: NDArray[np.float64],
        mag_combined_nodev: NDArray[np.float64],
        mag_combined_dev: NDArray[np.float64],
        mag_diff: NDArray[np.float64],
        motility_nodev: NDArray[np.float64],
        motility_dev: NDArray[np.float64],
        motility_diff: NDArray[np.float64],
        velcrit: NDArray[np.float64],
    ) -> Tuple[
        Dict[str, NDArray[np.float64]],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
    ]:
        """
        Process unstructured grid data and prepare output arrays.

        Returns
        -------
        dict_of_arrays : dict
            Dictionary of processed arrays for unstructured grid.
        rx : np.ndarray
            X-coordinates array.
        ry : np.ndarray
            Y-coordinates array.
        dx : float
            X-coordinate spacing.
        dy : float
            Y-coordinate spacing.
        """
        # Step 1: Prepare structured data using a helper function
        dx, dy, rx, ry, structured_data = DataPreparation.prepare_structured_data(
            xcor,
            ycor,
            mag_diff,
            mag_combined_dev,
            mag_combined_nodev,
            motility_nodev,
            motility_dev,
            motility_diff,
            velcrit,
        )

        # Step 2: Classify motility and create output dictionary
        dict_of_arrays = MotilityHandler.classify_motility_and_create_output(
            structured_data
        )

        return dict_of_arrays, rx, ry, dx, dy


class DataPreparation:
    """
    Prepares and handles data necessary for velocity stressor calculations.

    Provides methods to prepare run data, compute magnitudes for different runs,
    load or calculate probabilities,
    initialize combined arrays, and load and prepare data for further processing.
    """

    @staticmethod
    def prepare_run_data(files_nodev: List[str], files_dev: List[str]) -> pd.DataFrame:
        """
        Prepare and sort run data based on run numbers extracted from filenames.

        Parameters
        ----------
        files_nodev : list
            List of filenames for the 'no device' runs.
        files_dev : list
            List of filenames for the 'with device' runs.

        Returns
        -------
        data_frame : pd.DataFrame
            DataFrame containing sorted file and run information.
        """
        # Extract run numbers from file names
        run_num_nodev = np.array(
            [int(file.split(".")[0].split("_")[-2]) for file in files_nodev]
        )
        run_num_dev = np.array(
            [int(file.split(".")[0].split("_")[-2]) for file in files_dev]
        )

        # Adjust file order if necessary
        if np.any(run_num_nodev != run_num_dev):
            files_dev = [
                files_dev[np.flatnonzero(run_num_dev == ri)[0]] for ri in run_num_nodev
            ]

        # Create DataFrame with sorted runs
        return pd.DataFrame(
            {
                "files_nodev": files_nodev,
                "run_num_nodev": run_num_nodev,
                "files_dev": files_dev,
                "run_num_dev": run_num_dev,
            }
        ).sort_values(by="run_num_dev")

    @staticmethod
    def compute_all_magnitudes(
        data_frame: pd.DataFrame,
        fpath_nodev: str,
        fpath_dev: str,
        uvar: str,
        vvar: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitudes for 'no device' and 'with device' runs.

        Parameters
        ----------
        data_frame : pd.DataFrame
            DataFrame containing file and run information.
        fpath_nodev : str
            Directory path to the baseline/no device model run NetCDF files.
        fpath_dev : str
            Directory path to the with device model run NetCDF files.
        uvar : str
            U-component variable name in NetCDF files.
        vvar : str
            V-component variable name in NetCDF files.

        Returns
        -------
        mag_nodev : np.ndarray
            Magnitude data array for 'no device' runs.
        mag_dev : np.ndarray
            Magnitude data array for 'with device' runs.
        """

        def compute_magnitudes(file_path: str, file_name: str) -> np.ndarray:
            with Dataset(os.path.join(file_path, file_name)) as dataset:
                u = dataset.variables[uvar][:].data
                v = dataset.variables[vvar][:].data
                return np.sqrt(u**2 + v**2)

        # Compute magnitudes for each run
        mag_nodev = np.array(
            [
                compute_magnitudes(fpath_nodev, row.files_nodev)
                for _, row in data_frame.iterrows()
            ]
        )
        mag_dev = np.array(
            [
                compute_magnitudes(fpath_dev, row.files_dev)
                for _, row in data_frame.iterrows()
            ]
        )

        return mag_nodev, mag_dev

    @staticmethod
    def load_or_calculate_probabilities(
        probabilities_file: str, mag_dev_shape: Tuple[int], data_frame: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Load probabilities from a CSV file or calculate them based on the number of runs.

        Parameters
        ----------
        probabilities_file : str
            File path to probabilities/boundary condition *.csv file.
        mag_dev_shape : Tuple[int]
            Shape of the magnitude data array for 'with device' runs.
        data_frame : pd.DataFrame
            Data frame containing run information.

        Returns
        -------
        bc_probability : pd.DataFrame
            DataFrame containing run numbers and their corresponding probabilities.
        """
        if probabilities_file != "":
            if not os.path.exists(probabilities_file):
                raise FileNotFoundError(
                    f"The file {probabilities_file} does not exist."
                )
            # Load BC file with probabilities and find appropriate probability
            bc_probability = pd.read_csv(probabilities_file, delimiter=",")
            bc_probability["run_num"] = bc_probability["run number"] - 1
            bc_probability = bc_probability.sort_values(by="run number")
            bc_probability["probability"] = bc_probability["% of yr"].values / 100

            # Exclude rows based on the 'Exclude' column
            if "Exclude" in bc_probability.columns:
                bc_probability = bc_probability[
                    ~(
                        (bc_probability["Exclude"] == "x")
                        | (bc_probability["Exclude"] == "X")
                    )
                ]
        else:  # assume run_num in file name is return interval
            bc_probability = pd.DataFrame()
            # Generate sequential run numbers from zero
            bc_probability["run_num"] = np.arange(0, mag_dev_shape[0])
            # Calculate probabilities assuming run_num in name is the return interval
            bc_probability["probability"] = 1 / data_frame.run_num_dev.to_numpy()
            bc_probability["probability"] = (
                bc_probability["probability"] / bc_probability["probability"].sum()
            )  # rescale to ensure total sum is 1

        return bc_probability

    @staticmethod
    def initialize_combined_arrays(
        gridtype: str, mag_nodev: NDArray[np.float64], mag_dev: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Initialize combined arrays for magnitude data based on grid type.

        Parameters
        ----------
        gridtype : str
            Grid type ('structured' or 'unstructured').
        mag_nodev : np.ndarray
            Magnitude data array for 'no device' runs.
        mag_dev : np.ndarray
            Magnitude data array for 'with device' runs.

        Returns
        -------
        mag_combined_nodev : np.ndarray
            Initialized combined magnitude array for 'no device' runs.
        mag_combined_dev : np.ndarray
            Initialized combined magnitude array for 'with device' runs.
        """
        if gridtype == "structured":
            mag_combined_nodev = np.zeros(np.shape(mag_nodev[0, :, :]))
            mag_combined_dev = np.zeros(np.shape(mag_dev[0, :, :]))
        else:
            mag_combined_nodev = np.zeros(np.shape(mag_nodev)[-1])
            mag_combined_dev = np.zeros(np.shape(mag_dev)[-1])

        return mag_combined_nodev, mag_combined_dev

    @staticmethod
    def load_and_prepare_data(fpath_nodev: str, fpath_dev: str) -> Tuple[
        str,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        Optional[pd.DataFrame],
    ]:
        """
        Load and prepare data for velocity stressor calculation.

        Parameters
        ----------
        fpath_nodev : str
            Directory path to the baseline/no device model run NetCDF files.
        fpath_dev : str
            Directory path to the with device model run NetCDF files.

        Returns
        -------
        gridtype : str
            Grid type ('structured' or 'unstructured').
        xcor : np.ndarray
            X-coordinates array.
        ycor : np.ndarray
            Y-coordinates array.
        mag_nodev : np.ndarray
            Magnitude data array for 'no device' runs.
        mag_dev : np.ndarray
            Magnitude data array for 'with device' runs.
        data_frame : pd.DataFrame or None
            DataFrame containing file and run information if
            multiple files are found, otherwise None.
        """
        prepared_data = file_processor.load_and_sort_files(fpath_nodev, fpath_dev)
        gridtype = prepared_data.gridtype
        xcor = prepared_data.xcor
        ycor = prepared_data.ycor
        mag_nodev = prepared_data.mag_nodev
        mag_dev = prepared_data.mag_dev
        data_frame = prepared_data.data_frame

        # Ensure data_frame is handled properly
        if data_frame is None:
            pass

        if gridtype == "structured" and (xcor[0, 0] == 0) & (xcor[-1, 0] == 0):
            # At least for some runs the boundary has 0 coordinates. Check and fix.
            xcor, ycor, mag_nodev, mag_dev = trim_zeros(xcor, ycor, mag_nodev, mag_dev)

        return PreparedData(gridtype, xcor, ycor, mag_nodev, mag_dev, data_frame)

    @staticmethod
    def prepare_structured_data(
        xcor: NDArray[np.float64],
        ycor: NDArray[np.float64],
        mag_diff: NDArray[np.float64],
        mag_combined_dev: NDArray[np.float64],
        mag_combined_nodev: NDArray[np.float64],
        motility_nodev: NDArray[np.float64],
        motility_dev: NDArray[np.float64],
        motility_diff: NDArray[np.float64],
        velcrit: NDArray[np.float64],
    ) -> Tuple[
        float, float, NDArray[np.float64], NDArray[np.float64], StructuredArrays
    ]:
        """
        Estimate grid spacing and convert all required arrays to structured format.

        Parameters
        ----------
        xcor, ycor : np.ndarray
            X and Y coordinates arrays.
        mag_diff,
        mag_combined_dev,
        mag_combined_nodev,
        motility_nodev,
        motility_dev,
        motility_diff,
        velcrit : np.ndarray
            Arrays to be converted to structured format.

        Returns
        -------
        dx, dy : float
            Estimated grid spacing.
        rx, ry : np.ndarray
            Structured X and Y coordinates arrays.
        structured_data : StructuredArrays
            Initialized StructuredArrays object with all converted structured arrays.
        """
        # Estimate grid spacing
        dx = dy = grid_processor.estimate_grid_spacing_for_unstructured(xcor, ycor)

        # Convert all required arrays to structured format using a single loop
        rx, ry, structured_arrays = grid_processor.create_structured_arrays(
            xcor,
            ycor,
            dx,
            mag_diff,
            mag_combined_dev,
            mag_combined_nodev,
            motility_nodev,
            motility_dev,
            motility_diff,
            velcrit,
        )

        # Initialize structured data
        structured_data = StructuredArrays(
            mag_diff_struct=structured_arrays[0],
            mag_combined_dev_struct=structured_arrays[1],
            mag_combined_nodev_struct=structured_arrays[2],
            motility_nodev_struct=structured_arrays[3],
            motility_dev_struct=structured_arrays[4],
            motility_diff_struct=structured_arrays[5],
            velcrit_struct=structured_arrays[6],
        )

        return dx, dy, rx, ry, structured_data


class MotilityHandler:
    """
    Handles motility-related operations for velocity stressors.

    Provides methods to handle motility arrays and classify motility for
    creating output dictionaries.
    """

    @staticmethod
    def handle_motility_arrays(
        xcor: NDArray[np.float64],
        ycor: NDArray[np.float64],
        dxdy: float,
        motility_nodev: Optional[NDArray[np.float64]],
        motility_dev: Optional[NDArray[np.float64]],
        motility_diff: Optional[NDArray[np.float64]],
        velcrit: Optional[NDArray[np.float64]],
        template_shape: Tuple[int, int],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Handle motility and velocity critical arrays for unstructured grids.

        Parameters
        ----------
        xcor, ycor, dxdy : same as previous functions.
        motility_nodev, motility_dev, motility_diff, velcrit : same as original function.
        template_shape : Tuple[int, int]
            Shape of the template array for initializing arrays with NaN if required.

        Returns
        -------
        motility_nodev_struct,
            motility_dev_struct,
            motility_diff_struct,
            velcrit_struct : np.ndarray
            Structured arrays for motility and velocity critical values.
        """
        if motility_nodev is not None and motility_dev is not None:
            _, _, motility_nodev_struct = create_structured_array_from_unstructured(
                xcor, ycor, motility_nodev, dxdy, flatness=0.2
            )
            _, _, motility_dev_struct = create_structured_array_from_unstructured(
                xcor, ycor, motility_dev, dxdy, flatness=0.2
            )
            _, _, motility_diff_struct = create_structured_array_from_unstructured(
                xcor, ycor, motility_diff, dxdy, flatness=0.2
            )
            _, _, velcrit_struct = create_structured_array_from_unstructured(
                xcor, ycor, velcrit, dxdy, flatness=0.2
            )
        else:
            motility_nodev_struct = np.full(template_shape, np.nan)
            motility_dev_struct = np.full(template_shape, np.nan)
            motility_diff_struct = np.full(template_shape, np.nan)
            velcrit_struct = np.full(template_shape, np.nan)

        return (
            motility_nodev_struct,
            motility_dev_struct,
            motility_diff_struct,
            velcrit_struct,
        )

    @staticmethod
    def classify_motility_and_create_output(
        structured_data: StructuredArrays,
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Classify motility and create output dictionary for unstructured grid.

        Returns
        -------
        dict_of_arrays : dict
            Dictionary of processed arrays for unstructured grid.
        """
        motility_classification = vel_stress_calc.classify_motility(
            structured_data.motility_dev_struct,
            structured_data.motility_nodev_struct,
        )
        motility_classification = np.where(
            np.isnan(structured_data.mag_diff_struct), -100, motility_classification
        )

        return {
            "velocity_magnitude_without_devices": structured_data.mag_combined_nodev_struct,
            "velocity_magnitude_with_devices": structured_data.mag_combined_dev_struct,
            "velocity_magnitude_difference": structured_data.mag_diff_struct,
            "motility_without_devices": structured_data.motility_nodev_struct,
            "motility_with_devices": structured_data.motility_dev_struct,
            "motility_difference": structured_data.motility_diff_struct,
            "motility_classified": motility_classification,
            "critical_velocity": structured_data.velcrit_struct,
        }


class VelocityStressorCalculator:
    """
    Calculates velocity stressors based on processed data.

    Provides methods for applying value selection, combining magnitudes with probabilities, and
    calculating motility and differences between 'with device' and 'no device' runs.
    """

    @staticmethod
    def classify_motility(
        motility_parameter_dev: NDArray[np.float64],
        motility_parameter_nodev: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        classifies larval motility from device runs to no device runs.

        Parameters
        ----------
        motility_parameter_dev : Array
            motility parameter (vel/vel_crit) for with device runs.
        motility_parameter_nodev : TYPE
            motility parameter (vel/vel_crit) for without (baseline) device runs.

        Returns
        -------
        motility_classification : array
            Numerically classified array where,
            3 = New Motility
            2 = Increased Motility
            1 = Reduced Motility
            0 = No Change
            -1 = Motility Stops
        """

        motility_classification = np.zeros(motility_parameter_dev.shape)
        # Motility Stops
        motility_classification = np.where(
            (
                (motility_parameter_dev < motility_parameter_nodev)
                & (motility_parameter_nodev >= 1)
                & (motility_parameter_dev < 1)
            ),
            -1,
            motility_classification,
        )
        # Reduced Motility (Tw<Tb) & (Tw-Tb)>1
        motility_classification = np.where(
            (
                (motility_parameter_dev < motility_parameter_nodev)
                & (motility_parameter_nodev >= 1)
                & (motility_parameter_dev >= 1)
            ),
            1,
            motility_classification,
        )
        # Increased Motility (Tw>Tb) & (Tw-Tb)>1
        motility_classification = np.where(
            (
                (motility_parameter_dev > motility_parameter_nodev)
                & (motility_parameter_nodev >= 1)
                & (motility_parameter_dev >= 1)
            ),
            2,
            motility_classification,
        )
        # New Motility
        motility_classification = np.where(
            (
                (motility_parameter_dev > motility_parameter_nodev)
                & (motility_parameter_nodev < 1)
                & (motility_parameter_dev >= 1)
            ),
            3,
            motility_classification,
        )
        # NoChange or NoMotility = 0
        return motility_classification

    @staticmethod
    def apply_value_selection(
        mag_dev: NDArray[np.float64],
        mag_nodev: NDArray[np.float64],
        value_selection: Optional[str],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Apply a value selection method (e.g., Maximum, Mean, Final Timestep)
        to the magnitude arrays.

        Parameters
        ----------
        mag_dev : np.ndarray
            Magnitude data array for 'with device' runs.
        mag_nodev : np.ndarray
            Magnitude data array for 'no device' runs.
        value_selection : str or None
            Value selection method. Options: 'Maximum', 'Mean', 'Final Timestep', or None.

        Returns
        -------
        mag_dev : np.ndarray
            Processed magnitude data array for 'with device' runs.
        mag_nodev : np.ndarray
            Processed magnitude data array for 'no device' runs.
        """
        if value_selection == "Maximum":
            mag_dev = np.nanmax(mag_dev, axis=1)  # max over time
            mag_nodev = np.nanmax(mag_nodev, axis=1)  # max over time
        elif value_selection == "Mean":
            mag_dev = np.nanmean(mag_dev, axis=1)  # mean over time
            mag_nodev = np.nanmean(mag_nodev, axis=1)  # mean over time
        elif value_selection == "Final Timestep":
            mag_dev = mag_dev[:, -1, :]  # last time step
            mag_nodev = mag_nodev[:, -1, :]  # last time step
        else:
            mag_dev = np.nanmax(mag_dev, axis=1)  # default to max over time
            mag_nodev = np.nanmax(mag_nodev, axis=1)  # default to max over time

        return mag_dev, mag_nodev

    @staticmethod
    def combine_magnitudes_with_probabilities(
        mag_nodev: NDArray[np.float64],
        mag_dev: NDArray[np.float64],
        bc_probability: pd.DataFrame,
        gridtype: str,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Combine magnitudes using probabilities for 'no device' and 'with device' runs.

        Parameters
        ----------
        mag_nodev : np.ndarray
            Magnitude data array for 'no device' runs.
        mag_dev : np.ndarray
            Magnitude data array for 'with device' runs.
        bc_probability : pd.DataFrame
            DataFrame containing run numbers and their corresponding probabilities.
        gridtype : str
            Grid type ('structured' or 'unstructured').

        Returns
        -------
        mag_combined_nodev : np.ndarray
            Combined magnitude array for 'no device' runs.
        mag_combined_dev : np.ndarray
            Combined magnitude array for 'with device' runs.
        """

        # Initialize combined arrays based on grid type
        data_preparation = DataPreparation()
        mag_combined_nodev, mag_combined_dev = (
            data_preparation.initialize_combined_arrays(gridtype, mag_nodev, mag_dev)
        )

        # Combine magnitudes using calculated probabilities
        for run_number, prob in zip(
            bc_probability["run_num"].values, bc_probability["probability"].values
        ):
            mag_combined_nodev = mag_combined_nodev + prob * mag_nodev[run_number, :]
            mag_combined_dev = mag_combined_dev + prob * mag_dev[run_number, :]

        return mag_combined_nodev, mag_combined_dev

    @staticmethod
    def calculate_motility_and_differences(
        mag_combined_nodev: NDArray[np.float64],
        mag_combined_dev: NDArray[np.float64],
        receptor_filename: Optional[str],
        xcor: NDArray[np.float64],
        ycor: NDArray[np.float64],
        latlon: bool,
    ) -> MotilityAndDifferences:
        """
        Calculate differences in magnitudes and motility metrics based on combined data.

        Parameters
        ----------
        mag_combined_nodev : np.ndarray
            Combined magnitude array for 'no device' runs.
        mag_combined_dev : np.ndarray
            Combined magnitude array for 'with device' runs.
        receptor_filename : str, optional
            File path to the receptor file (*.csv or *.tif). Default is None.
        xcor : np.ndarray
            X-coordinates array.
        ycor : np.ndarray
            Y-coordinates array.
        latlon : bool
            True if coordinates are lat/lon. Default is True.

        Returns
        -------
        mag_diff : np.ndarray
            Difference in magnitudes between 'with device' and 'no device' runs.
        motility_nodev : np.ndarray
            Motility array for 'no device' runs.
        motility_dev : np.ndarray
            Motility array for 'with device' runs.
        motility_diff : np.ndarray
            Difference in motility between 'with device' and 'no device' runs.
        velcrit : np.ndarray
            Critical velocity array.
        """
        mag_diff = mag_combined_dev - mag_combined_nodev
        velcrit = calc_receptor_array(
            receptor_filename, xcor, ycor, latlon=latlon, mask=~np.isnan(mag_diff)
        )
        motility_nodev = mag_combined_nodev / velcrit
        motility_dev = mag_combined_dev / velcrit
        motility_diff = motility_dev - motility_nodev

        return MotilityAndDifferences(
            mag_diff, motility_nodev, motility_dev, motility_diff, velcrit
        )

    @staticmethod
    def calculate_velocity_stressors(
        fpath_nodev: str,
        fpath_dev: str,
        probabilities_file: str,
        receptor_filename: Optional[str] = None,
        latlon: bool = True,
        value_selection: Optional[str] = None,
    ) -> Tuple[
        Dict[str, NDArray[np.float64]],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
        str,
    ]:
        """
        Calculate velocity stressors based on 'no device' and 'with device' runs.
        """

        # Step 1: Load and prepare data
        prepared_data = DataPreparation.load_and_prepare_data(fpath_nodev, fpath_dev)

        # Step 2: Load or calculate probabilities
        bc_probability = DataPreparation.load_or_calculate_probabilities(
            probabilities_file,
            prepared_data.mag_dev.shape,
            prepared_data.data_frame,
        )

        # Step 3: Apply value selection if needed
        if np.ndim(prepared_data.mag_nodev) == 5:  # Depth-averaged for structured array
            prepared_data.mag_dev = np.nanmean(prepared_data.mag_dev, axis=2)
            prepared_data.mag_nodev = np.nanmean(prepared_data.mag_nodev, axis=2)

        mag_dev, mag_nodev = vel_stress_calc.apply_value_selection(
            prepared_data.mag_dev, prepared_data.mag_nodev, value_selection
        )

        # Step 4: Combine magnitudes with probabilities
        mag_combined_nodev, mag_combined_dev = (
            vel_stress_calc.combine_magnitudes_with_probabilities(
                mag_nodev, mag_dev, bc_probability, prepared_data.gridtype
            )
        )

        # Step 5: Calculate differences and motility
        motility_and_differences = vel_stress_calc.calculate_motility_and_differences(
            mag_combined_nodev,
            mag_combined_dev,
            receptor_filename,
            prepared_data.xcor,
            prepared_data.ycor,
            latlon,
        )

        # Step 6: Process grids and prepare output arrays
        if prepared_data.gridtype == "structured":
            return (
                *grid_processor.process_structured_grid(
                    prepared_data.xcor,
                    prepared_data.ycor,
                    mag_combined_nodev,
                    mag_combined_dev,
                    motility_and_differences.mag_diff,
                    motility_and_differences.motility_nodev,
                    motility_and_differences.motility_dev,
                    motility_and_differences.motility_diff,
                    motility_and_differences.velcrit,
                ),
                prepared_data.gridtype,
            )

        return (
            *grid_processor.process_unstructured_grid(
                prepared_data.xcor,
                prepared_data.ycor,
                mag_combined_nodev,
                mag_combined_dev,
                motility_and_differences.mag_diff,
                motility_and_differences.motility_nodev,
                motility_and_differences.motility_dev,
                motility_and_differences.motility_diff,
                motility_and_differences.velcrit,
            ),
            prepared_data.gridtype,
        )


class RasterCreator:
    """
    Handles creation of raster files from numpy arrays for velocity stressor calculations.

    This class provides methods to create raster files from numpy arrays based on grid data
    and save them to a specified output path.

    Attributes
    ----------
    output_path : str
        Directory to save output files.
    grid_data : GridData
        Contains grid data such as coordinates, spacing, grid type, and CRS.

    Methods
    -------
    create_raster(array_name, numpy_array)
        Create a raster file from a numpy array and save it to the output path.
    calculate_bounds()
        Calculate bounds for raster creation based on CRS and grid data.
    prepare_numpy_array(array, gridtype)
        Prepare the numpy array by flipping and transposing it based on the grid type.
    create_raster_from_arrays(dict_of_arrays, use_numpy_arrays)
        Create raster files from multiple numpy arrays and save them to the specified output path.
    """

    def __init__(self, output_path: str, grid_data: "GridData"):
        self.output_path = output_path
        self.grid_data = grid_data

    def create_raster(self, array_name: str, numpy_array: NDArray[np.float64]) -> str:
        """
        Create a raster file from a numpy array and save it to the output path.
        """
        raster_path = os.path.join(self.output_path, array_name + ".tif")
        bounds = self.calculate_bounds()
        rows, cols = numpy_array.shape

        # Create output raster
        output_raster = create_raster(raster_path, cols, rows, nbands=1)

        # Write the numpy array to the raster
        numpy_array_to_raster(
            output_raster,
            numpy_array,
            bounds,
            [self.grid_data.dx, self.grid_data.dy],
            self.grid_data.crs,
            raster_path,
        )

        # Close the raster
        output_raster = None

        return raster_path

    def calculate_bounds(self) -> List[float]:
        """
        Calculate bounds based on CRS and grid data.
        """
        if self.grid_data.crs == 4326:
            rxx = np.where(
                self.grid_data.rx > 180, self.grid_data.rx - 360, self.grid_data.rx
            )
            return [
                rxx.min() - self.grid_data.dx / 2,
                self.grid_data.ry.max() - self.grid_data.dy / 2,
            ]

        return [
            self.grid_data.rx.min() - self.grid_data.dx / 2,
            self.grid_data.ry.max() - self.grid_data.dy / 2,
        ]

    @staticmethod
    def prepare_numpy_array(
        array: NDArray[np.float64], gridtype: str
    ) -> NDArray[np.float64]:
        """
        Prepare the numpy array by flipping and transposing it based on the grid type.
        """
        if gridtype == "structured":
            return np.flip(np.transpose(array), axis=0)
        return np.flip(array, axis=0)

    def create_raster_from_arrays(
        self,
        dict_of_arrays: Dict[str, NDArray[np.float64]],
        use_numpy_arrays: List[str],
    ) -> List[str]:
        """
        Create raster files from numpy arrays and save them to the specified output path.
        """
        output_rasters = []

        for use_numpy_array in use_numpy_arrays:
            # Prepare the numpy array based on the grid type
            numpy_array = self.prepare_numpy_array(
                dict_of_arrays[use_numpy_array], self.grid_data.gridtype
            )

            # Create the raster
            raster_path = self.create_raster(use_numpy_array, numpy_array)
            output_rasters.append(raster_path)

        return output_rasters


grid_processor = GridProcessor()
file_processor = FileProcessor()
vel_stress_calc = VelocityStressorCalculator()


def calculate_area_statistics(
    output_path: str,
    crs: int,
    receptor_filename: Optional[str] = None,
    secondary_constraint_filename: Optional[str] = None,
) -> None:
    """
    Perform area calculations and generate statistics files for velocity change.

    Parameters
    ----------
    output_path : str
        Directory to save output files.
    crs : int
        Coordinate Reference System / EPSG code.
    receptor_filename : str, optional
        File path to the receptor file (*.csv or *.tif). Default is None.
    secondary_constraint_filename : str, optional
        File path to the secondary constraint file (*.tif). Default is None.
    """
    # Calculate area for velocity magnitude difference
    bin_layer(
        os.path.join(output_path, "velocity_magnitude_difference.tif"),
        receptor_filename=None,
        receptor_names=None,
        latlon=crs == 4326,
    ).to_csv(
        os.path.join(output_path, "velocity_magnitude_difference.csv"), index=False
    )

    # Calculate area with secondary constraints if provided
    if secondary_constraint_filename:
        bin_layer(
            os.path.join(output_path, "velocity_magnitude_difference.tif"),
            receptor_filename=os.path.join(output_path, "velocity_risk_layer.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="risk layer",
        ).to_csv(
            os.path.join(
                output_path, "velocity_magnitude_difference_at_velocity_risk_layer.csv"
            ),
            index=False,
        )

    # Calculate area for velocity changes with critical velocity if receptor is provided
    if receptor_filename:
        bin_layer(
            os.path.join(output_path, "velocity_magnitude_difference.tif"),
            receptor_filename=os.path.join(output_path, "critical_velocity.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="critical velocity",
        ).to_csv(
            os.path.join(
                output_path, "velocity_magnitude_difference_at_critical_velocity.csv"
            ),
            index=False,
        )

        bin_layer(
            os.path.join(output_path, "motility_difference.tif"),
            receptor_filename=None,
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
        ).to_csv(os.path.join(output_path, "motility_difference.csv"), index=False)

        bin_layer(
            os.path.join(output_path, "motility_difference.tif"),
            receptor_filename=os.path.join(output_path, "critical_velocity.tif"),
            receptor_names=None,
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="critical velocity",
        ).to_csv(
            os.path.join(output_path, "motility_difference_at_critical_velocity.csv"),
            index=False,
        )

        classify_layer_area(
            os.path.join(output_path, "motility_classified.tif"),
            at_values=[-3, -2, -1, 0, 1, 2, 3],
            value_names=[
                "New Deposition",
                "Increased Deposition",
                "Reduced Deposition",
                "No Change",
                "Reduced Erosion",
                "Increased Erosion",
                "New Erosion",
            ],
            latlon=crs == 4326,
        ).to_csv(os.path.join(output_path, "motility_classified.csv"), index=False)

        classify_layer_area(
            os.path.join(output_path, "motility_classified.tif"),
            receptor_filename=os.path.join(output_path, "critical_velocity.tif"),
            at_values=[-3, -2, -1, 0, 1, 2, 3],
            value_names=[
                "New Deposition",
                "Increased Deposition",
                "Reduced Deposition",
                "No Change",
                "Reduced Erosion",
                "Increased Erosion",
                "New Erosion",
            ],
            limit_receptor_range=[0, np.inf],
            latlon=crs == 4326,
            receptor_type="critical velocity",
        ).to_csv(
            os.path.join(output_path, "motility_classified_at_critical_velocity.csv"),
            index=False,
        )

        # Additional classifications with secondary constraints
        if secondary_constraint_filename:
            bin_layer(
                os.path.join(output_path, "motility_difference.tif"),
                receptor_filename=os.path.join(output_path, "velocity_risk_layer.tif"),
                receptor_names=None,
                limit_receptor_range=[0, np.inf],
                latlon=crs == 4326,
                receptor_type="risk layer",
            ).to_csv(
                os.path.join(
                    output_path, "motility_difference_at_velocity_risk_layer.csv"
                ),
                index=False,
            )

            classify_layer_area_2nd_constraint(
                raster_to_sample=os.path.join(output_path, "motility_classified.tif"),
                secondary_constraint_filename=os.path.join(
                    output_path, "velocity_risk_layer.tif"
                ),
                at_raster_values=[-3, -2, -1, 0, 1, 2, 3],
                at_raster_value_names=[
                    "New Deposition",
                    "Increased Deposition",
                    "Reduced Deposition",
                    "No Change",
                    "Reduced Erosion",
                    "Increased Erosion",
                    "New Erosion",
                ],
                limit_constraint_range=[0, np.inf],
                latlon=crs == 4326,
                receptor_type="risk layer",
            ).to_csv(
                os.path.join(
                    output_path, "motility_classified_at_velocity_risk_layer.csv"
                ),
                index=False,
            )


def prepare_arrays_and_grid_data(
    dev_present_file: str,
    dev_notpresent_file: str,
    probabilities_file: str,
    receptor_filename: Optional[str],
    crs: int,
    value_selection: Optional[str],
) -> Tuple[Dict[str, NDArray[np.float64]], GridData]:
    """
    Prepare arrays and grid data for velocity stressor calculation.
    """
    # Calculate velocity stressors
    dict_of_arrays, rx, ry, dx, dy, gridtype = (
        vel_stress_calc.calculate_velocity_stressors(
            fpath_nodev=dev_notpresent_file,
            fpath_dev=dev_present_file,
            probabilities_file=probabilities_file,
            receptor_filename=receptor_filename,
            latlon=crs == 4326,
            value_selection=value_selection,
        )
    )

    # Create a GridData instance
    grid_data = GridData(rx=rx, ry=ry, dx=dx, dy=dy, gridtype=gridtype, crs=crs)

    return dict_of_arrays, grid_data


def determine_numpy_arrays_to_use(receptor_filename: Optional[str], _) -> List[str]:
    """
    Determine which numpy arrays to use for raster creation.
    """
    if receptor_filename:
        return [
            "velocity_magnitude_without_devices",
            "velocity_magnitude_with_devices",
            "velocity_magnitude_difference",
            "motility_without_devices",
            "motility_with_devices",
            "motility_difference",
            "motility_classified",
            "critical_velocity",
        ]
    return [
        "velocity_magnitude_without_devices",
        "velocity_magnitude_with_devices",
        "velocity_magnitude_difference",
    ]


def handle_secondary_constraint(
    secondary_constraint_filename: Optional[str],
    dict_of_arrays: Dict[str, NDArray[np.float64]],
    rx: NDArray[np.float64],
    ry: NDArray[np.float64],
) -> None:
    """
    Handle secondary constraints and update the dictionary of arrays if provided.
    """
    if secondary_constraint_filename:
        if not os.path.exists(secondary_constraint_filename):
            raise FileNotFoundError(
                f"The file {secondary_constraint_filename} does not exist."
            )
        rrx, rry, constraint = secondary_constraint_geotiff_to_numpy(
            secondary_constraint_filename
        )
        dict_of_arrays["velocity_risk_layer"] = resample_structured_grid(
            rrx, rry, constraint, rx, ry, interpmethod="nearest"
        )


def create_rasters(
    output_path: str,
    grid_data: GridData,
    dict_of_arrays: Dict[str, NDArray[np.float64]],
    use_numpy_arrays: List[str],
) -> List[str]:
    """
    Create rasters from arrays using the RasterCreator class.
    """
    raster_creator = RasterCreator(output_path, grid_data)
    return raster_creator.create_raster_from_arrays(dict_of_arrays, use_numpy_arrays)


def run_velocity_stressor(
    dev_present_file: str,
    dev_notpresent_file: str,
    probabilities_file: str,
    crs: int,
    output_path: str,
    receptor_filename: Optional[str] = None,
    secondary_constraint_filename: Optional[str] = None,
    value_selection: Optional[str] = None,
) -> Dict[str, str]:
    """
    Creates GeoTIFFs and area change statistics files for velocity change.
    """
    os.makedirs(
        output_path, exist_ok=True
    )  # Create output directory if it doesn't exist

    # Step 1: Prepare arrays and grid data
    dict_of_arrays, grid_data = prepare_arrays_and_grid_data(
        dev_present_file,
        dev_notpresent_file,
        probabilities_file,
        receptor_filename,
        crs,
        value_selection,
    )

    # Step 2: Determine which numpy arrays to use for raster creation
    use_numpy_arrays = determine_numpy_arrays_to_use(receptor_filename, dict_of_arrays)

    # Step 3: Handle secondary constraint if provided
    handle_secondary_constraint(
        secondary_constraint_filename, dict_of_arrays, grid_data.rx, grid_data.ry
    )
    if secondary_constraint_filename:
        use_numpy_arrays.append("velocity_risk_layer")

    # Step 4: Create rasters from arrays
    output_rasters = create_rasters(
        output_path, grid_data, dict_of_arrays, use_numpy_arrays
    )

    # Step 5: Generate output dictionary of raster paths
    output = {
        os.path.basename(raster_path).split(".")[0]: raster_path
        for raster_path in output_rasters
    }
    return output
