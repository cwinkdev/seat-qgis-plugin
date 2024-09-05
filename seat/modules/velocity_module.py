#!/usr/bin/python

# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches

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
from typing import Optional, Tuple, List, Dict
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


# Single File Processing
def validate_and_list_files(fpath_nodev: str, fpath_dev: str) -> Tuple[List[str], List[str]]:
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

def process_single_file_case(
        files_nodev: List[str],
        files_dev: List[str],
        fpath_nodev: str,
        fpath_dev: str) -> Tuple[
            str,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]]:
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
            vvar: str):
        with Dataset(os.path.join(file_path, file_name)) as dataset:
            xcor = dataset.variables[xvar][:].data
            ycor = dataset.variables[yvar][:].data
            u = dataset.variables[uvar][:].data
            v = dataset.variables[vvar][:].data
            magnitude = np.sqrt(u**2 + v**2)
        return xcor, ycor, magnitude

    # Read grid definitions and variables
    with Dataset(os.path.join(fpath_dev, files_dev[0])) as file_dev_present:
        gridtype, xvar, yvar, uvar, vvar = check_grid_define_vars(file_dev_present)

    # Read and compute magnitudes for 'with device' and 'no device' runs
    _, _, mag_dev = read_and_compute_magnitude(fpath_dev, files_dev[0], xvar, yvar, uvar, vvar)
    xcor, ycor, mag_nodev = read_and_compute_magnitude(
        fpath_nodev,
        files_nodev[0],
        xvar,
        yvar,
        uvar,
        vvar
        )

    return gridtype, xcor, ycor, mag_nodev, mag_dev



# Multiple File Processing
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
    run_num_nodev = np.array([int(file.split(".")[0].split("_")[-2]) for file in files_nodev])
    run_num_dev = np.array([int(file.split(".")[0].split("_")[-2]) for file in files_dev])

    # Adjust file order if necessary
    if np.any(run_num_nodev != run_num_dev):
        files_dev = [files_dev[np.flatnonzero(run_num_dev == ri)[0]] for ri in run_num_nodev]

    # Create DataFrame with sorted runs
    return pd.DataFrame({
        "files_nodev": files_nodev,
        "run_num_nodev": run_num_nodev,
        "files_dev": files_dev,
        "run_num_dev": run_num_dev,
    }).sort_values(by="run_num_dev")

def compute_all_magnitudes(
        data_frame: pd.DataFrame,
        fpath_nodev: str,
        fpath_dev: str,
        uvar: str,
        vvar: str) -> Tuple[np.ndarray, np.ndarray]:
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
        [compute_magnitudes(fpath_nodev, row.files_nodev) for _, row in data_frame.iterrows()]
        )
    mag_dev = np.array(
        [compute_magnitudes(fpath_dev, row.files_dev) for _, row in data_frame.iterrows()]
        )

    return mag_nodev, mag_dev

def process_multiple_files_case(
        files_nodev: List[str],
        files_dev: List[str],
        fpath_nodev: str,
        fpath_dev: str) -> Tuple[
            str,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            pd.DataFrame]:
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
    data_frame : pd.DataFrame
        DataFrame containing file and run information.
    """
    # Prepare run data and get sorted DataFrame
    data_frame = prepare_run_data(files_nodev, files_dev)

    # Read the first file to define grid type and coordinates
    with Dataset(os.path.join(fpath_dev, data_frame['files_dev'].iloc[0])) as dataset:
        gridtype, xvar, yvar, uvar, vvar = check_grid_define_vars(dataset)
        xcor = dataset.variables[xvar][:].data
        ycor = dataset.variables[yvar][:].data

    # Compute magnitudes for 'no device' and 'with device' runs
    mag_nodev, mag_dev = compute_all_magnitudes(
        data_frame,
        fpath_nodev,
        fpath_dev,
        uvar,
        vvar
    )

    return gridtype, xcor, ycor, mag_nodev, mag_dev, data_frame



def load_and_sort_files(fpath_nodev: str, fpath_dev: str) -> Tuple[
    List[str],
    List[str],
    str,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[pd.DataFrame]
]:
    """
    Load and sort NetCDF files for 'no device' and 'with device' runs and process their data.

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
        DataFrame containing file and run information if multiple files are found, otherwise None.
    """
    files_nodev, files_dev = validate_and_list_files(fpath_nodev, fpath_dev)

    if len(files_nodev) == 1 and len(files_dev) == 1:
        gridtype, xcor, ycor, mag_nodev, mag_dev = process_single_file_case(
            files_nodev, files_dev, fpath_nodev, fpath_dev
        )
        data_frame = None  # No DataFrame needed for a single file case
    elif len(files_nodev) == len(files_dev):
        gridtype, xcor, ycor, mag_nodev, mag_dev, data_frame = process_multiple_files_case(
            files_nodev, files_dev, fpath_nodev, fpath_dev
        )
    else:
        raise ValueError(
            f"Number of device runs ({len(files_dev)}) must be the same as no device runs "
            f"({len(files_nodev)})."
        )

    return gridtype, xcor, ycor, mag_nodev, mag_dev, data_frame

def load_or_calculate_probabilities(
        probabilities_file: str,
        mag_dev_shape: Tuple[int],
        data_frame: pd.DataFrame) -> pd.DataFrame:
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
            raise FileNotFoundError(f"The file {probabilities_file} does not exist.")
        # Load BC file with probabilities and find appropriate probability
        bc_probability = pd.read_csv(probabilities_file, delimiter=",")
        bc_probability["run_num"] = bc_probability["run number"] - 1
        bc_probability = bc_probability.sort_values(by="run number")
        bc_probability["probability"] = bc_probability["% of yr"].values / 100

        # Exclude rows based on the 'Exclude' column
        if "Exclude" in bc_probability.columns:
            bc_probability = bc_probability[
                ~((bc_probability["Exclude"] == "x") | (bc_probability["Exclude"] == "X"))
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

def apply_value_selection(
        mag_dev: NDArray[np.float64],
        mag_nodev: NDArray[np.float64],
        value_selection: Optional[str]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply a value selection method (e.g., Maximum, Mean, Final Timestep) to the magnitude arrays.

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

def initialize_combined_arrays(
        gridtype: str,
        mag_nodev: NDArray[np.float64],
        mag_dev: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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


def calculate_velocity_stressors(
    fpath_nodev: str,
    fpath_dev: str,
    probabilities_file: str,
    receptor_filename: Optional[str] = None,
    latlon: bool = True,
    value_selection: Optional[str] = None,
) -> Tuple[
    List[NDArray[np.float64]],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    str,
]:
    """
    Calculate velocity stressors based on 'no device' and 'with device' runs.

    Parameters
    ----------
    fpath_nodev : str
        Directory path to the baseline/no device model run NetCDF files.
    fpath_dev : str
        Directory path to the with device model run NetCDF files.
    probabilities_file : str
        File path to probabilities/boundary condition *.csv file.
    receptor_filename : str, optional
        File path to the receptor file (*.csv or *.tif). Default is None.
    latlon : bool, optional
        True if coordinates are lat/lon. Default is True.
    value_selection : str, optional
        Temporal selection of shear stress (not currently used). Default is None.

    Returns
    -------
    dict_of_arrays : dict
        Dictionary containing arrays of velocity magnitudes, motility, and classifications.
    rx : np.ndarray
        X-coordinates array.
    ry : np.ndarray
        Y-coordinates array.
    dx : float
        X-spacing.
    dy : float
        Y-spacing.
    gridtype : str
        Grid type ('structured' or 'unstructured').
    """
    (
    gridtype,
    xcor,
    ycor,
    mag_nodev,
    mag_dev,
    data_frame
    ) = load_and_sort_files(fpath_nodev, fpath_dev)

    # Ensure data_frame is handled properly
    if data_frame is None:
        pass

    if gridtype == "structured":
        if (xcor[0, 0] == 0) & (xcor[-1, 0] == 0):
            # at least for some runs the boundary has 0 coordinates. Check and fix.
            xcor, ycor, mag_nodev, mag_dev = trim_zeros(xcor, ycor, mag_nodev, mag_dev)

    bc_probability = load_or_calculate_probabilities(probabilities_file, mag_dev.shape, data_frame)

    # ensure velocity is depth averaged for structured array [run_num, time, layer, x, y]
    #  and drop dimension
    if np.ndim(mag_nodev) == 5:
        mag_dev = np.nanmean(mag_dev, axis=2)
        mag_nodev = np.nanmean(mag_nodev, axis=2)

    (mag_dev, mag_nodev) = apply_value_selection(mag_dev, mag_nodev, value_selection)

    (
        mag_combined_nodev,
        mag_combined_dev
    ) = initialize_combined_arrays(gridtype, mag_nodev, mag_dev)

    for run_number, prob in zip(
        bc_probability["run_num"].values, bc_probability["probability"].values
    ):
        mag_combined_nodev = mag_combined_nodev + prob * mag_nodev[run_number, :]
        mag_combined_dev = mag_combined_dev + prob * mag_dev[run_number, :]

    mag_diff = mag_combined_dev - mag_combined_nodev
    velcrit = calc_receptor_array(
        receptor_filename, xcor, ycor, latlon=latlon, mask=~np.isnan(mag_diff)
    )
    motility_nodev = mag_combined_nodev / velcrit
    # motility_nodev = np.where(velcrit == 0, np.nan, motility_nodev)
    motility_dev = mag_combined_dev / velcrit
    # motility_dev = np.where(velcrit == 0, np.nan, motility_dev)
    # Calculate risk metrics over all runs

    motility_diff = motility_dev - motility_nodev

    if gridtype == "structured":
        motility_classification = classify_motility(motility_dev, motility_nodev)
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
    else:  # unstructured
        dxdy = estimate_grid_spacing(xcor, ycor, nsamples=100)
        dx = dxdy
        dy = dxdy
        rx, ry, mag_diff_struct = create_structured_array_from_unstructured(
            xcor, ycor, mag_diff, dxdy, flatness=0.2
        )
        _, _, mag_combined_dev_struct = create_structured_array_from_unstructured(
            xcor, ycor, mag_combined_dev, dxdy, flatness=0.2
        )
        _, _, mag_combined_nodev_struct = create_structured_array_from_unstructured(
            xcor, ycor, mag_combined_nodev, dxdy, flatness=0.2
        )
        if not ((receptor_filename is None) or (receptor_filename == "")):
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
            motility_nodev_struct = np.nan * mag_diff_struct
            motility_dev_struct = np.nan * mag_diff_struct
            motility_diff_struct = np.nan * mag_diff_struct
            velcrit_struct = np.nan * mag_diff_struct

        motility_classification = classify_motility(
            motility_dev_struct, motility_nodev_struct
        )

        motility_classification = np.where(
            np.isnan(mag_diff_struct), -100, motility_classification
        )

        dict_of_arrays = {
            "velocity_magnitude_without_devices": mag_combined_nodev_struct,
            "velocity_magnitude_with_devices": mag_combined_dev_struct,
            "velocity_magnitude_difference": mag_diff_struct,
            "motility_without_devices": motility_nodev_struct,
            "motility_with_devices": motility_dev_struct,
            "motility_difference": motility_diff_struct,
            "motility_classified": motility_classification,
            "critical_velocity": velcrit_struct,
        }
    return dict_of_arrays, rx, ry, dx, dy, gridtype


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
    creates geotiffs and area change statistics files for velocity change

    Parameters
    ----------
    dev_present_file : str
        Directory path to the baseline/no device model run netcdf files.
    dev_notpresent_file : str
        Directory path to the baseline/no device model run netcdf files.
    probabilities_file : str
        File path to probabilities/bondary condition *.csv file.
    crs : scalar
        Coordiante Reference System / EPSG code.
    output_path : str
        File directory to save output.
    receptor_filename : str, optional
        File path to the recetptor file (*.csv or *.tif). The default is None.
    secondary_constraint_filename: str, optional
        File path to the secondary constraint file (*.tif). The default is None.

    Returns
    -------
    output_rasters : dict
        key = names of output rasters, val = full path to raster:
    """

    os.makedirs(
        output_path, exist_ok=True
    )  # create output directory if it doesn't exist

    dict_of_arrays, rx, ry, dx, dy, gridtype = calculate_velocity_stressors(
        fpath_nodev=dev_notpresent_file,
        fpath_dev=dev_present_file,
        probabilities_file=probabilities_file,
        receptor_filename=receptor_filename,
        latlon=crs == 4326,
        value_selection=value_selection,
    )

    if not ((receptor_filename is None) or (receptor_filename == "")):
        use_numpy_arrays = [
            "velocity_magnitude_without_devices",
            "velocity_magnitude_with_devices",
            "velocity_magnitude_difference",
            "motility_without_devices",
            "motility_with_devices",
            "motility_difference",
            "motility_classified",
            "critical_velocity",
        ]
    else:
        use_numpy_arrays = [
            "velocity_magnitude_without_devices",
            "velocity_magnitude_with_devices",
            "velocity_magnitude_difference",
        ]

    if not (
        (secondary_constraint_filename is None) or (secondary_constraint_filename == "")
    ):
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
        use_numpy_arrays.append("velocity_risk_layer")

    numpy_array_names = [i + ".tif" for i in use_numpy_arrays]

    output_rasters = []
    for array_name, use_numpy_array in zip(numpy_array_names, use_numpy_arrays):
        if gridtype == "structured":
            numpy_array = np.flip(np.transpose(dict_of_arrays[use_numpy_array]), axis=0)
        else:
            numpy_array = np.flip(dict_of_arrays[use_numpy_array], axis=0)

        cell_resolution = [dx, dy]
        if crs == 4326:
            rxx = np.where(rx > 180, rx - 360, rx)
            bounds = [rxx.min() - dx / 2, ry.max() - dy / 2]
        else:
            bounds = [rx.min() - dx / 2, ry.max() - dy / 2]
        rows, cols = numpy_array.shape
        # create an ouput raster given the stressor file path
        output_rasters.append(os.path.join(output_path, array_name))
        output_raster = create_raster(
            os.path.join(output_path, array_name),
            cols,
            rows,
            nbands=1,
        )

        # post processing of numpy array to output raster
        numpy_array_to_raster(
            output_raster,
            numpy_array,
            bounds,
            cell_resolution,
            crs,
            os.path.join(output_path, array_name),
        )
        output_raster = None

    # Area calculations pull form rasters to ensure uniformity
    bin_layer(
        os.path.join(output_path, "velocity_magnitude_difference.tif"),
        receptor_filename=None,
        receptor_names=None,
        latlon=crs == 4326,
    ).to_csv(
        os.path.join(output_path, "velocity_magnitude_difference.csv"), index=False
    )
    if not (
        (secondary_constraint_filename is None) or (secondary_constraint_filename == "")
    ):
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
    if not ((receptor_filename is None) or (receptor_filename == "")):
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

        if not (
            (secondary_constraint_filename is None)
            or (secondary_constraint_filename == "")
        ):
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
    output = {}

    for val in output_rasters:
        output[os.path.basename(os.path.normpath(val)).split(".")[0]] = val
    return output
