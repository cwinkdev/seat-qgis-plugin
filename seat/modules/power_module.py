# power_module.py

# pylint: disable=too-many-lines

"""
This module provides functionalities related to power calculations.

AUTHORS:
    - Timothy R. Nelson (TRN) - tnelson@integral-corp.com

NOTES:
    1. .OUT files and .pol files must be in the same folder.
    2. .OUT file is format sensitive.

CHANGE HISTORY:
    - 2022-08-01: TRN - File created.
    - 2023-08-05: TRN - Comments and slight edits.
"""


import io
import re
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.figure import Figure


@dataclass
class DevicePowerPlotData:
    """
    Data class to hold information related to plotting device power.

    Attributes
    ----------
    ndx : float
        Spacing for x-axis ticks in the plot.
    mxy : float
        Maximum y-axis value for the plot.
    nr : int
        Number of rows for subplots in the plot grid.
    nc : int
        Number of columns for subplots in the plot grid.
    fig : plt.Figure
        Matplotlib Figure object for the overall plot.
    axes_grid : np.ndarray
        Numpy array containing Axes objects for subplots in the grid.
    """

    ndx: float
    mxy: float
    nr: int
    nc: int
    fig: plt.Figure
    axes_grid: np.ndarray


@dataclass
class DevicePowerData:
    """
    Data class to hold information related to device power calculations and plotting.

    Attributes
    ----------
    devices : pd.DataFrame
        DataFrame containing power data per device for each scenario.
    power_plot_data : DevicePowerPlotData
        Instance of DevicePowerPlotData that holds plotting information, including
        the figure, axes grid, and plot parameters such as the number of rows and columns.
    device_power_year : np.ndarray
        Numpy array containing power values for each device over the course of a year,
        scaled by the hydrodynamic probabilities.
    """

    devices: pd.DataFrame
    power_plot_data: DevicePowerPlotData
    device_power_year: np.ndarray


class ObstacleData:
    """
    This class provides methods for handling obstacle data related to power device configuration.
    It includes methods for reading obstacle polygon files, calculating centroids, plotting obstacle
    locations, pairing devices based on centroids, and extracting device location data.

    Methods
    -------
    read_obstacle_polygon_file(power_device_configuration_file: str) ->
    Dict[str, NDArray[np.float64]]:
        Reads an obstacle polygon file and returns the xy coordinates of each obstacle.

    find_mean_point_of_obstacle_polygon(obstacles: Dict[str, NDArray[np.float64]]) ->
    NDArray[np.float64]:
        Calculates the centroid of each obstacle.

    plot_test_obstacle_locations(obstacles: Dict[str, NDArray[np.float64]]) -> Figure:
        Creates a plot of the spatial distribution and location of each obstacle.

    pair_devices(centroids: NDArray[np.float64]) -> NDArray[np.int32]:
        Determines the two intersecting obstacles that create a device
        by finding the closest centroid pairs.

    extract_device_location(
    obstacles: Dict[str, NDArray[np.float64]], device_index: List[List[int]]
    ) -> DataFrame:
        Creates a dictionary summary of each device location based
        on obstacle data and device indices.
    """

    @staticmethod
    def read_obstacle_polygon_file(
        power_device_configuration_file: str,
    ) -> Dict[str, NDArray[np.float64]]:
        """
        reads the obstacle polygon file

        Parameters
        ----------
        power_device_configuration_file : str
            filepath of .pol file.

        Returns
        -------
        obstacles : Dict
            xy of each obstacle.

        """
        try:
            with io.open(power_device_configuration_file, "r", encoding="utf-8") as inf:
                lines = inf.readlines()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"File not found: {power_device_configuration_file}"
            ) from exc

        ic = 0
        obstacles = {}
        while ic < len(lines) - 1:
            if "Obstacle" in lines[ic]:
                obstacle = lines[ic].strip()
                obstacles[obstacle] = {}
                ic += 1  # skip to next line
                nrows = int(lines[ic].split()[0])
                ic += 1  # skip to next line
                x = []
                y = []
                for _ in range(nrows):  # read polygon
                    xi, yi = [float(i) for i in lines[ic].split()]
                    x = np.append(x, xi)
                    y = np.append(y, yi)
                    ic += 1  # skip to next line
                obstacles[obstacle] = np.vstack((x, y)).T
            else:
                ic += 1
        return obstacles

    @staticmethod
    def find_mean_point_of_obstacle_polygon(
        obstacles: Dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Calculates the center of each obstacle.

        Parameters
        ----------
        obstacles : Dict
            x,y of each obstacle.

        Returns
        -------
        centroids : array
            Centroid of each obstacle.

        """
        centroids = np.empty((0, 3), dtype=int)
        for ic, obstacle in enumerate(obstacles.keys()):
            centroids = np.vstack(
                (
                    centroids,
                    [
                        ic,
                        np.nanmean(obstacles[obstacle][:, 0]),
                        np.nanmean(obstacles[obstacle][:, 1]),
                    ],
                )
            )
        return centroids

    @staticmethod
    def plot_test_obstacle_locations(
        obstacles: Dict[str, NDArray[np.float64]]
    ) -> Figure:
        """
        Creates a plot of the spatial distribution and location of each obstacle.

        Parameters
        ----------
        obstacles : Dict
            xy of each obstacle.

        Returns
        -------
        fig : pyplot figure handle
            pyplot figure handle.

        """
        fig, ax = plt.subplots(figsize=(10, 10))
        for obstacle in obstacles.keys():
            ax.plot(
                obstacles[obstacle][:, 0],
                obstacles[obstacle][:, 1],
                ".",
                markersize=3,
                alpha=0,
            )
            ax.text(
                obstacles[obstacle][0, 0],
                obstacles[obstacle][0, 1],
                f"{obstacle}",
                fontsize=8,
            )
            ax.text(
                obstacles[obstacle][1, 0],
                obstacles[obstacle][1, 1],
                f"{obstacle}",
                fontsize=8,
            )
        fig.tight_layout()
        return fig

    @staticmethod
    def pair_devices(centroids: NDArray[np.float64]) -> NDArray[np.int32]:
        """
        Determins the two intersecting obstacles to that create a device.

        Parameters
        ----------
        centroids : TYPE
            DESCRIPTION.

        Returns
        -------
        devices : TYPE
            DESCRIPTION.

        """
        devices = np.empty((0, 2), dtype=int)
        while len(centroids) > 0:
            # print(centroids)
            # must have dimensions M,N with each M [index, x , y]
            pair = PowerCalculatorTools.centroid_diffs(
                centroids[1:, :], centroids[0, :]
            )
            devices = np.vstack((devices, pair))
            centroids = centroids[~np.isin(centroids[:, 0], pair), :]
        return devices

    @staticmethod
    def extract_device_location(
        obstacles: Dict[str, NDArray[np.float64]], device_index: List[List[int]]
    ) -> DataFrame:
        """
        Creates a dictionary summary of each device location

        Parameters
        ----------
        obstacles : TYPE
            DESCRIPTION.
        device_index : TYPE
            DESCRIPTION.

        Returns
        -------
        devices_df : TYPE
            DESCRIPTION.

        """
        devices = {}
        for device, [ix1, ix2] in enumerate(device_index):
            key = f"{device+1:03.0f}"
            devices[key] = {}
            xy = obstacles[f"Obstacle {ix1+1}"]
            xy = np.vstack((xy, obstacles[f"Obstacle {ix2+1}"]))
            # create polygon from bottom left to upper right assuming rectangular
            x = xy[:, 0]
            y = xy[:, 1]
            devices[key]["polyx"] = [
                np.nanmin(x),
                np.nanmin(x),
                np.nanmax(x),
                np.nanmax(x),
            ]
            devices[key]["polyy"] = [
                np.nanmin(y),
                np.nanmax(y),
                np.nanmax(y),
                np.nanmin(y),
            ]
            devices[key]["lower_left"] = [np.nanmin(x), np.nanmin(y)]
            devices[key]["centroid"] = [np.nanmean(x), np.nanmean(y)]
            devices[key]["width"] = np.nanmax(x) - np.nanmin(x)
            devices[key]["height"] = np.nanmax(y) - np.nanmin(y)
        devices_df = pd.DataFrame.from_dict(devices, orient="index")
        return devices_df


class PowerFileData:
    """
    This class provides methods for handling power files related to power device calculations.
    It includes methods for reading power files, processing power files, sorting data files,
    and resetting data order.

    Methods
    -------
    read_power_file(datafile: str) -> Tuple[NDArray[np.float64], float]:
        Reads a power file and extracts the final set of converged data.

    read_and_process_power_files(
    power_files: str, bc_data: DataFrame
    ) -> Tuple[List[str], NDArray[np.float64], NDArray[np.float64]]:
        Reads and processes multiple power files, returning sorted data file paths,
        power values, and total power values.

    sort_data_files_by_runnumber(bc_data: DataFrame, datafiles: List[str]) -> List[str]:
        Sorts the power data files based on the run number specified
        in the hydrodynamic probabilities data.

    sort_bc_data_by_runnumber(bc_data: DataFrame) -> DataFrame:
        Sorts the hydrodynamic probabilities DataFrame by the 'run number' column.

    reset_bc_data_order(bc_data: DataFrame) -> Union[DataFrame, None]:
        Resets the order of the hydrodynamic probabilities DataFrame to its
        original order if 'original_order' column exists.
    """

    @staticmethod
    def read_power_file(datafile: str) -> Tuple[NDArray[np.float64], float]:
        """
        Read power file and extract final set of converged data

        Parameters
        ----------
        datafile : file path
            path and file name of power file.

        Returns
        -------
        Power : 1D Numpy Array [m]
            Individual data files for each observation [m].
        total_power : Scalar
            Total power from all observations.

        """
        with io.open(datafile, "r", encoding="utf-8") as inf:
            # = io.open(datafile, "r")  # Read datafile
            for line in inf:  # iterate through each line
                if re.match("Iteration:", line):
                    power_array = []
                    # If a new iteration is found, initalize varialbe
                    # or overwrite existing iteration
                else:  # data
                    # extract float variable from line
                    power = float(line.split("=")[-1].split("W")[0].strip())
                    power_array = np.append(
                        power_array, power
                    )  # append data for each observation
        total_power = np.nansum(power_array)  # Total power from all observations
        return power_array, total_power

    @staticmethod
    def read_and_process_power_files(
        power_files: str, bc_data: DataFrame
    ) -> Tuple[List[str], NDArray[np.float64], NDArray[np.float64]]:
        """
        Reads and processes power files.

        Parameters
        ----------
        power_files : str
            Path to the power files directory.
        bc_data : DataFrame
            DataFrame containing the hydrodynamic probabilities.

        Returns
        -------
        datafiles : List[str]
            Sorted list of power data file paths.
        power_array : NDArray[np.float64]
            Array of power values from each power file.
        total_power : NDArray[np.float64]
            Total power values from each power file.
        """
        datafiles_o = [s for s in os.listdir(power_files) if s.endswith(".OUT")]
        datafiles = PowerFileData.sort_data_files_by_runnumber(bc_data, datafiles_o)

        total_power = []
        ic = 0
        for datafile in datafiles:
            p, tp = PowerFileData.read_power_file(os.path.join(power_files, datafile))
            if ic == 0:
                power_array = np.empty((len(p), 0), float)
            power_array = np.append(power_array, p[:, np.newaxis], axis=1)
            total_power = np.append(total_power, tp)
            ic += 1

        return datafiles, power_array, total_power

    @staticmethod
    def sort_data_files_by_runnumber(
        bc_data: DataFrame, datafiles: List[str]
    ) -> List[str]:
        """
        Sorts the data files based on the run number specified in `bc_data`.

        Parameters
        ----------
        bc_data : pd.DataFrame
            DataFrame containing the 'run number' and other metadata.
        datafiles : list
            List of data file paths.

        Returns
        -------
        List[str]
            List of sorted data file paths based on the run number.
        """
        bc_data_sorted = PowerFileData.sort_bc_data_by_runnumber(bc_data.copy())
        return [datafiles[i] for i in bc_data_sorted.original_order.to_numpy()]

    @staticmethod
    def sort_bc_data_by_runnumber(bc_data: DataFrame) -> DataFrame:
        """
        Sorts the `bc_data` DataFrame by the 'run number' column.

        Parameters
        ----------
        bc_data : pd.DataFrame
            DataFrame containing the 'run number' and other metadata.

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame with an added 'original_order' column to track original indices.
        """
        bc_data["original_order"] = range(0, len(bc_data))
        return bc_data.sort_values(by="run number")

    @staticmethod
    def reset_bc_data_order(bc_data: DataFrame) -> Union[DataFrame, None]:
        """
        Resets the order of `bc_data` DataFrame to its original order
        if 'original_order' column exists.

        Parameters
        ----------
        bc_data : pd.DataFrame
            DataFrame containing the 'run number' and other metadata.

        Returns
        -------
        pd.DataFrame or None
            Sorted DataFrame if 'original_order' column exists, otherwise None.
        """
        if np.isin("original_order", bc_data.columns):
            bc_data = bc_data.sort()
        return bc_data


class PowerPlotter:
    """
    This class provides methods for plotting power data related to power devices.
    It includes methods for plotting and saving power summaries, creating power heatmaps,
    preparing device power plots, plotting device power scenarios, and
    processing and plotting device power.

    Methods
    -------
    plot_and_save_power_summaries(
    total_power_scaled: NDArray[np.float64],
    power_scaled: NDArray[np.float64],
    datafiles: List[str], save_path: str
    ) -> None:
        Plots and saves power summary figures for total scaled power, power per run obstacle,
        and total power for all obstacles.

    create_power_heatmap(device_power: DataFrame, crs: Optional[int] = None) -> Figure:
        Creates a heatmap of device locations with power values as color.

    prepare_device_power_plot(devices: pd.DataFrame) -> DevicePowerPlotData:
        Prepares the plot for device power data by creating subplots and calculating axis limits.

    plot_device_power_scenario(
    device_power_data: DevicePowerData, save_path: str, ic: int, col: str
    ) ->
    Tuple[DevicePowerData, str]:
        Plots the power data for a specific scenario and saves the plot to the specified path.

    process_and_plot_device_power(
    device_power_data: DevicePowerData,
    save_path: str,
    obstacles: Dict[str, np.ndarray],
    device_index: np.ndarray,
    crs: Optional[int] = None
    ) -> Tuple[DevicePowerData, str]:
        Processes device power data to compute total annual power per device,
        generates plots, and creates a heatmap of device power.
    """

    @staticmethod
    def plot_and_save_power_summaries(
        total_power_scaled: NDArray[np.float64],
        power_scaled: NDArray[np.float64],
        datafiles: List[str],
        save_path: str,
    ) -> None:
        """
        Plots and saves power summary figures.

        Parameters
        ----------
        total_power_scaled : NDArray[np.float64]
            Array of total scaled power values.
        power_scaled : NDArray[np.float64]
            Array of scaled power values for each scenario.
        datafiles : List[str]
            List of power data file names.
        save_path : str
            Path to save the generated figures.

        Returns
        -------
        None
        """
        # Plot total scaled power
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(
            np.arange(np.shape(total_power_scaled)[0]) + 1,
            np.log10(total_power_scaled),
            width=1,
            edgecolor="black",
        )
        ax.set_xlabel("Run Scenario")
        ax.set_ylabel("Power [$log_{10}(Watts)$]")
        ax.set_title("Total Power Annual")
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, "Total_Scaled_Power_Bars_per_Run.png"))

        # Plot scaled power per run obstacle
        subplot_grid_size = np.sqrt(np.shape(power_scaled)[1])
        fig, axes_grid = plt.subplots(
            np.round(subplot_grid_size).astype(int),
            np.ceil(subplot_grid_size).astype(int),
            sharex=True,
            sharey=True,
            figsize=(12, 10),
        )
        nr, nc = axes_grid.shape
        axes_grid = axes_grid.flatten()
        mxy = PowerCalculatorTools.roundup(np.log10(power_scaled.max().max()))
        ndx = np.ceil(power_scaled.shape[0] / 6)
        for ic in range(power_scaled.shape[1]):
            axes_grid[ic].bar(
                np.arange(np.shape(power_scaled)[0]) + 1,
                np.log10(power_scaled[:, ic]),
                width=1,
                edgecolor="black",
            )
            axes_grid[ic].set_title(f"{datafiles[ic]}", fontsize=8)
            axes_grid[ic].set_ylim([0, mxy])
            axes_grid[ic].set_xticks(np.arange(0, power_scaled.shape[0] + ndx, ndx))
            axes_grid[ic].set_xlim([0, power_scaled.shape[0] + 1])
        axes_grid = axes_grid.reshape(nr, nc)
        for ax in axes_grid[:, 0]:
            ax.set_ylabel("Power [$log_{10}(Watts)$]")
        for ax in axes_grid[-1, :]:
            ax.set_xlabel("Obstacle")
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, "Scaled_Power_Bars_per_run_obstacle.png"))

        # Plot total power for all obstacles
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(
            np.arange(np.shape(power_scaled)[0]) + 1,
            np.log10(np.sum(power_scaled, axis=1)),
            width=1,
            edgecolor="black",
        )
        ax.set_xlabel("Obstacle")
        ax.set_ylabel("Power [$log_{10}(Watts)$]")
        ax.set_title("Total Obstacle Power for all Runs")
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, "Total_Scaled_Power_Bars_per_obstacle.png"))

    @staticmethod
    def create_power_heatmap(
        device_power: DataFrame, crs: Optional[int] = None
    ) -> Figure:
        """
        Creates a heatmap of device location and power as cvalue.

        Parameters
        ----------
        device_power : dataframe
            device_power dataframe.
        crs : int
            Coordinate Reverence Systems EPSG number

        Returns
        -------
        fig : matplotlib figure handle
            matplotlib figure handle.

        """
        adjust_x = -360 if crs == 4326 else 0

        fig, ax = plt.subplots(figsize=(6, 4))
        lowerx = np.inf
        lowery = np.inf
        upperx = -np.inf
        uppery = -np.inf
        # cmap = ListedColormap(plt.get_cmap('Greens')(np.linspace(0.1, 1, 256)))
        # # skip too light colors
        cmap = ListedColormap(
            plt.get_cmap("turbo")(np.linspace(0.1, 1, 256))
        )  # skip too light colors
        # norm = plt.Normalize(device_power['Power [W]'].min(), device_power['Power [W]'].max())
        norm = plt.Normalize(
            0.9 * device_power["Power [W]"].min() * 1e-6,
            device_power["Power [W]"].max() * 1e-6,
        )
        for _, device in device_power.iterrows():
# # use https://stackoverflow.com/questions/10550477/how-do-i-set-color-to-rectangle-in-matplotlib
# # to create rectangles from the polygons above
# # scale color based on power device power range from 0 to max of array
# # This way the plot is array and grid independent, only based on centroid and device size,
# could make size variable if necessary.
            ax.add_patch(
                Rectangle(
                    (device.lower_left[0] + adjust_x, device.lower_left[1]),
                    np.nanmax([device.width, device.height]),
                    np.nanmax([device.width, device.height]),
                    color=cmap(norm(device["Power [W]"] * 1e-6)),
                )
            )
            lowerx = np.nanmin([lowerx, device.lower_left[0] + adjust_x])
            lowery = np.nanmin([lowery, device.lower_left[1]])
            upperx = np.nanmax([upperx, device.lower_left[0] + adjust_x + device.width])
            uppery = np.nanmax([uppery, device.lower_left[1] + device.height])
        xr = np.abs(np.max([lowerx, upperx]) - np.min([lowerx, upperx]))
        yr = np.abs(np.max([lowery, uppery]) - np.min([lowery, uppery]))
        ax.set_xlim([lowerx - 0.05 * xr, upperx + 0.05 * xr])
        ax.set_ylim([lowery - 0.05 * yr, uppery + 0.05 * yr])

        cb = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)
        cb.set_label("MW")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_xticks(np.linspace(lowerx, upperx, 5))
        ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=45)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%0.4f"))
        fig.tight_layout()
        return fig

    @staticmethod
    def prepare_device_power_plot(devices: pd.DataFrame) -> DevicePowerPlotData:
        """
        Prepares the plot for device power data.

        Parameters
        ----------
        devices : pd.DataFrame
            DataFrame of power per device per scenario.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray, int, int, float, float]
            A tuple containing:
            - fig: Matplotlib Figure object for plotting.
            - axes_grid: Array of Axes objects for subplots.
            - nr: Number of rows for subplots.
            - nc: Number of columns for subplots.
            - mxy: Maximum y-axis value for the plot.
            - ndx: Spacing for x-axis ticks in the plot.
        """
        subplot_grid_size = np.sqrt(devices.shape[1])
        fig, axes_grid = plt.subplots(
            np.round(subplot_grid_size).astype(int),
            np.ceil(subplot_grid_size).astype(int),
            sharex=True,
            sharey=True,
            figsize=(12, 10),
        )
        nr, nc = axes_grid.shape
        axes_grid = axes_grid.flatten()
        mxy = PowerCalculatorTools.roundup(np.log10(devices.max().max()))
        ndx = np.ceil(devices.shape[0] / 6)

        return DevicePowerPlotData(ndx, mxy, nr, nc, fig, axes_grid)

    @staticmethod
    def plot_device_power_scenario(
        device_power_data: DevicePowerData, save_path: str, ic: int, col: str
    ) -> Tuple[DevicePowerData, str]:
        """
        Plots the power data for a specific scenario and saves the plot to the specified path.

        Parameters
        ----------
        device_power_data : DevicePowerData
            An instance of DevicePowerData containing power data and plotting information.
        save_path : str
            Path to the directory where the output plot will be saved.
        ic : int
            Index of the subplot in the axes grid for plotting.
        col : str
            Name of the column in the devices DataFrame to plot.

        Returns
        -------
        Tuple[DevicePowerData, str]
            A tuple containing the updated DevicePowerData instance and the save path.
        """
        # Plot the power data for the specified scenario
        device_power_data.power_plot_data.axes_grid[ic].bar(
            np.arange(np.shape(device_power_data.devices[col])[0]) + 1,
            np.log10(device_power_data.devices[col].to_numpy()),
            width=1.0,
            edgecolor="black",
        )
        device_power_data.power_plot_data.axes_grid[ic].set_title(f"{col}", fontsize=8)
        device_power_data.power_plot_data.axes_grid[ic].set_ylim(
            [0, device_power_data.power_plot_data.mxy]
        )
        device_power_data.power_plot_data.axes_grid[ic].set_xticks(
            np.arange(
                0,
                device_power_data.devices.shape[0]
                + device_power_data.power_plot_data.ndx,
                device_power_data.power_plot_data.ndx,
            )
        )
        device_power_data.power_plot_data.axes_grid[ic].set_xlim(
            [0, device_power_data.devices.shape[0] + 1]
        )

        # Reshape the axes grid for proper layout
        device_power_data.power_plot_data.axes_grid = (
            device_power_data.power_plot_data.axes_grid.reshape(
                device_power_data.power_plot_data.nr,
                device_power_data.power_plot_data.nc,
            )
        )

        # Set y and x labels for subplots
        for ax in device_power_data.power_plot_data.axes_grid[:, 0]:
            ax.set_ylabel("Power [$log_{10}(Watts)$]")
        for ax in device_power_data.power_plot_data.axes_grid[-1, :]:
            ax.set_xlabel("Device")

        # Flatten the axes grid and adjust layout
        device_power_data.power_plot_data.axes_grid = (
            device_power_data.power_plot_data.axes_grid.flatten()
        )
        device_power_data.power_plot_data.fig.tight_layout()

        # Save the plot to the specified path
        device_power_data.power_plot_data.fig.savefig(
            os.path.join(save_path, "Scaled_Power_per_device_per_scenario.png")
        )

        return device_power_data, save_path

    @staticmethod
    def process_and_plot_device_power(
        device_power_data: DevicePowerData,
        save_path: str,
        obstacles: Dict[str, np.ndarray],
        device_index: np.ndarray,
        crs: Optional[int] = None,
    ) -> Tuple[DevicePowerData, str]:
        """
        Processes device power data to compute total annual power per device,
        generates plots for total power per device, and creates a heatmap of device power.

        Parameters
        ----------
        device_power_data : DevicePowerData
            An instance of DevicePowerData containing power data and plotting information.
        save_path : str
            Path to the directory where the output CSV and plots will be saved.
        obstacles : Dict[str, np.ndarray]
            Dictionary containing obstacle polygons with their respective coordinates.
        device_index : np.ndarray
            Numpy array representing paired device indices for processing.
        crs : Optional[int], optional
            Coordinate Reference System (CRS) EPSG code for plotting, by default None.

        Returns
        -------
        Tuple[DevicePowerData, str]
            A tuple containing the updated DevicePowerData instance and the save path.
        """
        # Calculate total annual power per device and save to CSV
        devices_total = pd.DataFrame({})
        devices_total["Power [W]"] = device_power_data.device_power_year.sum(axis=1)
        devices_total["Device"] = np.arange(1, len(devices_total) + 1)
        devices_total = devices_total.set_index("Device")
        devices_total.to_csv(os.path.join(save_path, "Power_per_device_annual.csv"))

        # Plot total scaled power per device and save plot
        device_power_data.power_plot_data.fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(
            devices_total.index,
            np.log10(devices_total["Power [W]"]),
            width=1,
            edgecolor="black",
        )
        ax.set_ylabel("Power [$log_{10}(Watts)$]")
        ax.set_xlabel("Device")
        device_power_data.power_plot_data.fig.savefig(
            os.path.join(save_path, "Total_Scaled_Power_per_Device_.png")
        )

        # Create and save a heatmap of device power
        device_power = ObstacleData.extract_device_location(obstacles, device_index)
        device_power["Power [W]"] = devices_total["Power [W]"].values
        device_power_data.power_plot_data.fig = PowerPlotter.create_power_heatmap(
            device_power, crs=crs
        )

        return device_power_data, save_path


class PowerCalculatorTools:
    """
    This class provides utility methods for calculating and processing power data for power devices.
    It includes methods for calculating and saving device power, processing device configurations,
    rounding up values, and determining closest centroid pairs.

    Methods
    -------
    calculate_and_save_device_power(
    power_array: np.ndarray,
    device_index: np.ndarray,
    bc_data: pd.DataFrame,
    datafiles: List[str],
    save_path: str
    ) -> DevicePowerData:
        Calculates the power per device based on the power array and device index,
        saves the device power data to a CSV file, and prepares for plotting.

    process_device_configuration(power_files: str, save_path: str) ->
    Tuple[np.ndarray, Dict[str, NDArray[np.float64]]]:
        Processes device configuration by reading obstacle polygon files, calculating centroids,
        pairing devices, and saving the configuration and plots.

    roundup(x: float, val: int = 2) -> float:
        Rounds up the number `x` to the nearest multiple of `val`.

    centroid_diffs(centroids: NDArray[np.float64], centroid: NDArray[np.float64]) -> List[int]:
        Determines the closest centroid pair based on the distances between centroids.
    """

    @staticmethod
    def calculate_and_save_device_power(
        power_array: np.ndarray,
        device_index: np.ndarray,
        bc_data: pd.DataFrame,
        datafiles: List[str],
        save_path: str,
    ) -> DevicePowerData:
        """
        Calculates the power per device based on the power array and device index,
        saves the device power data to a CSV file, and prepares for plotting.

        Parameters
        ----------
        power_array : np.ndarray
            Array of power values from each power file.
        device_index : np.ndarray
            Array representing paired device indices.
        bc_data : pd.DataFrame
            DataFrame containing hydrodynamic probabilities.
        datafiles : List[str]
            List of power data file names.
        save_path : str
            Path to save the generated CSV and plots.

        Returns
        -------
        DevicePowerPlotData
            A dataclass containing all the necessary data for device power plotting.
        """
        # Initialize device power array
        device_power = np.empty((0, np.shape(power_array)[1]), dtype=float)
        for ic0, ic1 in device_index:
            device_power = np.vstack(
                (device_power, power_array[ic0, :] + power_array[ic1, :])
            )

        # Calculate device power per year
        devices = pd.DataFrame({})
        device_power_year = device_power * bc_data["% of yr"].to_numpy()
        for ic, name in enumerate(datafiles):
            devices[name] = device_power_year[:, ic]
        devices["Device"] = np.arange(1, len(devices) + 1)
        devices = devices.set_index("Device")
        devices.to_csv(os.path.join(save_path, "Power_per_device_per_scenario.csv"))

        # Prepare for plotting using the helper function
        power_plot_data = PowerPlotter.prepare_device_power_plot(devices)

        return DevicePowerData(
            devices,
            power_plot_data,
            device_power_year,
        )

    @staticmethod
    def process_device_configuration(
        power_files: str, save_path: str
    ) -> Tuple[np.ndarray, Dict[str, NDArray[np.float64]]]:
        """
        Processes device configuration by reading obstacle polygon files, calculating centroids,
        pairing devices, and saving the configuration and plots.

        Parameters
        ----------
        power_files : str
            Path to the power files directory containing .pol files.
        save_path : str
            Path to the directory where output files and plots will be saved.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, NDArray[np.float64]]]
            A tuple containing:
            - device_index: An array representing paired device indices.
            - obstacles: A dictionary where keys are obstacle names and
              values are their xy coordinates.
        """
        # Find all .pol files in the power_files directory
        power_device_configuration_file = [
            s
            for s in os.listdir(power_files)
            if (s.endswith(".pol") or s.endswith(".Pol") or s.endswith(".POL"))
        ]

        # Ensure only one .pol file is present
        if len(power_device_configuration_file) > 0:
            assert (
                len(power_device_configuration_file) == 1
            ), "More than 1 *.pol file found"

            # Read obstacle polygon file
            obstacles = ObstacleData.read_obstacle_polygon_file(
                os.path.join(power_files, power_device_configuration_file[0])
            )

            # Plot and save obstacle locations
            fig = ObstacleData.plot_test_obstacle_locations(obstacles)
            fig.savefig(os.path.join(save_path, "Obstacle_Locations.png"))

            # Calculate centroids of the obstacles
            centroids = ObstacleData.find_mean_point_of_obstacle_polygon(obstacles)
            centroids_df = pd.DataFrame(data=centroids, columns=["obstacle", "X", "Y"])
            centroids_df["obstacle"] = centroids_df["obstacle"].astype(int)
            centroids_df = centroids_df.set_index(["obstacle"])

            # Pair devices based on centroids
            device_index = ObstacleData.pair_devices(centroids)
            device_index_df = pd.DataFrame(
                {
                    "Device_Number": range(device_index.shape[0]),
                    "Index 1": device_index[:, 0],
                    "Index 2": device_index[:, 1],
                    "X": centroids_df.loc[device_index[:, 0], "X"],
                    "Y": centroids_df.loc[device_index[:, 0], "Y"],
                }
            )
            device_index_df["Device_Number"] = device_index_df["Device_Number"] + 1
            device_index_df = device_index_df.set_index("Device_Number")
            device_index_df.to_csv(os.path.join(save_path, "Obstacle_Matching.csv"))

            # Plot and save device number locations
            fig, ax = plt.subplots(figsize=(10, 10))
            for device in device_index_df.index.values:
                ax.plot(
                    device_index_df.loc[device, "X"],
                    device_index_df.loc[device, "Y"],
                    ".",
                    alpha=0,
                )
                ax.text(
                    device_index_df.loc[device, "X"],
                    device_index_df.loc[device, "Y"],
                    device,
                    fontsize=8,
                )
            fig.savefig(os.path.join(save_path, "Device Number Location.png"))

        return device_index, obstacles

    @staticmethod
    def roundup(x: float, val: int = 2) -> float:
        """
        Rounds up the number `x` to the nearest multiple of `val`.

        Parameters
        ----------
        x : float
            The number to round up.
        val : int, optional
            The value to round to the nearest multiple of (default is 2).

        Returns
        -------
        float
            The rounded-up number.
        """
        return np.ceil(x / val) * val

    @staticmethod
    def centroid_diffs(
        centroids: NDArray[np.float64], centroid: NDArray[np.float64]
    ) -> List[int]:
        """
        Determines the closest centroid pair

        Parameters
        ----------
        centroids : Dict
            dimensions M,N with each M [index, x , y]
        centroid : array
            single x,y.

        Returns
        -------
        pair : list
            index of closest centroid.

        """

        diff = centroids[:, 1:] - centroid[1:]
        min_arg = np.nanargmin(np.abs(diff[:, -1] - diff[:, 0]))
        pair = [int(centroid[0]), int(centroids[min_arg, 0])]
        return pair


# # use https://stackoverflow.com/questions/10550477/how-do-i-set-color-to-rectangle-in-matplotlib
# # to create rectangles from the polygons above
# # scale color based on power device power range from 0 to max of array
# # This way the plot is array and grid independent, only based on centroid and device size,
# could make size variable if necessary.


def calculate_power(
    power_files: str,
    probabilities_file: str,
    save_path: Optional[str] = None,
    crs: Optional[int] = None,
) -> Tuple[DataFrame, DataFrame]:
    """
    Reads the power files and calculates the total annual power based on
    hydrodynamic probabilities in probabilities_file.
    Data are saved as a csv files.
    Three files are output:
        1) Total Power among all devices for each
        hydrodynamic conditions BC_probability_Annual_SETS_wPower.csv
        2) Power per device per hydordynamic scenario. Power_per_device_per_scenario.csv
        3) Total power per device during a year, scaled by $ of year in probabilities_file

    Parameters
    ----------
    fpath : file path
        Path to bc_file and power output files.
    probabilities_file : file name
        probabilities file name with extension.
    save_path: file path
        save directory

    Returns
    -------
    devices : Dataframe
        Scaled power per device per condition.
    devices_total : Dataframe
        Total annual power per device.

    """

    if not os.path.exists(power_files):
        raise FileNotFoundError(f"The directory {power_files} does not exist.")
    if not os.path.exists(probabilities_file):
        raise FileNotFoundError(f"The file {probabilities_file} does not exist.")

    bc_data = pd.read_csv(probabilities_file)

    # Read and process power files
    datafiles, power_array, total_power = PowerFileData.read_and_process_power_files(
        power_files, bc_data
    )

    power_scaled = bc_data["% of yr"].to_numpy() * power_array
    total_power_scaled = bc_data["% of yr"] * total_power

    # Summary of power given percent of year for each array
    # need to reorder total_power and Power to run roder in
    bc_data["Power_Run_Name"] = datafiles
    # bc_data['% of yr'] * total_power
    bc_data["Power [W]"] = total_power_scaled
    bc_data.to_csv(os.path.join(save_path, "BC_probability_wPower.csv"), index=False)

    PowerPlotter.plot_and_save_power_summaries(
        total_power_scaled, power_scaled, datafiles, save_path
    )

    device_index, obstacles = PowerCalculatorTools.process_device_configuration(
        power_files, save_path
    )

    device_power_data = PowerCalculatorTools.calculate_and_save_device_power(
        power_array, device_index, bc_data, datafiles, save_path
    )

    for ic, col in enumerate(device_power_data.devices.columns):
        # fig,ax = plt.subplots()
        device_power_data, save_path = PowerPlotter.plot_device_power_scenario(
            device_power_data, save_path, ic, col
        )

        # Sum power for the entire years (all datafiles) for each device

        device_power_data, save_path = PowerPlotter.process_and_plot_device_power(
            device_power_data, save_path, obstacles, device_index, crs
        )
        device_power_data.power_plot_data.fig.savefig(
            os.path.join(save_path, "Device_Power.png"), dpi=150
        )
        # plt.close(fig)
