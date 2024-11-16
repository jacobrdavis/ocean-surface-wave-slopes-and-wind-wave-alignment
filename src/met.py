"""
Functions for reading meteorological datasets.  This includes
flight-level meteorological datasets from NOAA and the United States Air
Force Reserve (USAFR) Weather Reconnaissance Squadron as well as
National Data Buoy Center (NDBC) standard meteorological data.
"""

__all__ = [
    "read_usafr_met_directory",
    "read_usafr_met_file",
    "read_noaa_met_directory",
    "read_noaa_met_file",
    "read_ndbc_met_file",
]

import glob
import os
import re
from typing import List, Literal, Callable, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.variable import MissingDimensionsError

# Retrieved Oct 2024 from: https://www.aoml.noaa.gov/hrd/format/usaf.html
USAFR_VAR_ATTRS = {
    "GMT_Time": {"Description": "Time of Day (UTC)", "Units": "Hours:Minutes:Seconds"},
    "ADR": {"Description": "Air Density Ratio", "Units": "none"},
    "AOA": {"Description": "Angle of Attack", "Units": "degrees"},
    "BCA": {"Description": "Baro-Corrected Altitude", "Units": "millibars"},
    "BSP": {"Description": "Baroset Pressure", "Units": "millibars"},
    "CAS": {"Description": "Calibrated Air Speed", "Units": "knots"},
    "CSP": {"Description": "Corrected Static Pressure", "Units": "millibars"},
    "DPR": {"Description": "Dynamic Pressure", "Units": "millibars"},
    "GPSA": {"Description": "GPS Altimeter", "Units": "meters"},
    "GS": {"Description": "Ground Speed", "Units": "knots"},
    "IA": {"Description": "Inertial Altimeter", "Units": "meters"},
    "ISP": {"Description": "Inertial Static Pressure", "Units": "millibars"},
    "LAT": {"Description": "Latitude", "Units": "degrees"},
    "LON": {"Description": "Longitude", "Units": "degrees"},
    "PA": {"Description": "Pressure Altitude", "Units": "meters"},
    "PITCH": {"Description": "Pitch angle", "Units": "degrees"},
    "PR": {"Description": "Pressure Ratio", "Units": "none"},
    "PT": {"Description": "Total Pressure", "Units": "millibars"},
    "RA": {"Description": "Radar Altitude", "Units": "meters"},
    "ROLL": {"Description": "Roll angle", "Units": "degrees"},
    "SS": {"Description": "Side Slip", "Units": "degrees"},
    "TA": {"Description": "Corrected Static Air Temperature", "Units": "°C"},
    "TAS": {"Description": "True Air Speed", "Units": "knots"},
    "THD": {"Description": "True Heading", "Units": "degrees"},
    "TRK": {"Description": "Track", "Units": "degrees"},
    "TT": {"Description": "Total Temperature", "Units": "°C"},
    "VE": {"Description": "Velocity East", "Units": "knots"},
    "VN": {"Description": "Velocity North", "Units": "knots"},
    "VV": {"Description": "Vertical Velocity", "Units": "knots"},
    "WDIR": {"Description": "Wind Direction", "Units": "degrees"},
    "WSPD": {"Description": "Wind Speed", "Units": "knots"},
    "TD": {"Description": "Dew Point Temperature - Digital", "Units": "°C"},
    "RR": {"Description": "Rain Rate (SFMR)", "Units": "mm/hr"},
    "SWS": {"Description": "Surface Wind Speed (SFMR)", "Units": "knots"},
    "CC": {"Description": "Course Correction", "Units": "degrees"},
    "DVAL": {"Description": "Deviation Value", "Units": "meters"},
    "GA": {"Description": "Geopotential Altitude", "Units": "meters"},
    "HSS": {"Description": "Height of Standard Surface", "Units": "meters"},
    "SLP": {"Description": "Sea Level Pressure", "Units": "millibars"},
    "WD": {"Description": "Wind Direction (calculated)", "Units": "degrees"},
    "WS": {"Description": "Wind Speed (calculated)", "Units": "knots"},
    "Valid Flags": {"Description": "Parameter validity flags", "Units": "binary"},
    "Validity": {"Description": "Parameter validity flags", "Units": "binary"},
    "Source Tags": {"Description": "Data Source Tags", "Units": "hexidecimal"},
    "Source": {"Description": "Data Source Tags", "Units": "hexidecimal"},
    "SATCOM": {"Description": "Satellite Communications status", "Units": "binary"},
    "ARC210": {"Description": "Motorola ARC-210 Radio status", "Units": "binary"},
    "AD": {"Description": "Analog/Digital Card status", "Units": "binary"},
    "DDPH": {"Description": "Digital Dew Point Hygrometer status", "Units": "binary"},
    "1553": {"Description": "Main aircraft data bus", "Units": "digital"},
    "SFMR": {"Description": "Stepped Frequency Microwave Radiometer output", "Units": "hexidecimal"},
}


def read_usafr_met_directory(
    directory: str,
    file_type: str = '.txt',
    data_vars: Union[str, List] = 'all',
    data_type: Literal['pandas', 'xarray'] = 'pandas',
    **concat_kwargs,
) -> Union[pd.DataFrame, xr.Dataset]:
    """ Read a directory of USAFR met data and concatenate.

    USAFR Weather Reconnaissance Squadron flight-level met data are at:
    https://www.aoml.noaa.gov/{year}-hurricane-field-program-data/
    where {year} is the year corresponding to the season of data collection.
    The met data file date should have the format {YYYYMMDD}U{#}.txt where
    Y, M, D are the year, month, and day of the flight, respectively, U is the
    aircraft identifier (U for C-130), and # is the flight number.
    For example, the file 20230829U1.01.txt was collected by a USAFR C-130
    during the first mission of August 29, 2023.

    Args:
        directory (str): Directory containing the met data files.
        file_type (str, optional): Met data file type. Defaults to '.txt'.
        data_vars (Union[str, List]): Variables to load into memory. Acceptable
            values include a list of variables or 'all'.  Defaults to 'all'.
        concat_kwargs: Additional keyword arguments passed to `pd.concat` or
            to `xr.concat`.

    Returns:
        Union[pd.DataFrame, xr.Dataset]: USAFR met data.
    """
    met_files = glob.glob(os.path.join(directory, '*' + file_type))
    met_files.sort()

    if not met_files:
        raise FileNotFoundError(
            f'No files of type "{file_type}" found in "{directory}". '
            'Please double check the directory and file_type.')

    # Read met files into a dictionary keyed by filename.
    met = {}
    for file in met_files:
        met_data = read_usafr_met_file(file, data_vars, data_type)
        base_file = os.path.basename(file)
        key = base_file.split('.')[0]
        met[key] = met_data

    # Concatenate the DataFrames (or Datasets)
    if data_type == 'xarray':
        concat_dim = 'GMT_Time'
        met_ds_concat = xr.concat(list(met.values()),
                                  dim=concat_dim,
                                  combine_attrs=_combine_usafr_attrs,
                                  **concat_kwargs)
        return met_ds_concat
    elif data_type == 'pandas':
        met_df_concat = pd.concat(met.values(), **concat_kwargs)
        return met_df_concat
    else:
        raise ValueError(f'{data_type} not supported.')


def read_usafr_met_file(
    filepath: str,
    data_vars: Union[str, List] = 'all',
    data_type: Literal['pandas', 'xarray'] = 'pandas',
) -> Union[pd.DataFrame, xr.Dataset]:
    """ Read USAFR C-130 'J' met data as a DataFrame or Dataset.

    Args:
        filepath (str): Path to met data file.
        data_vars (str or List): Variables to return. Acceptable
            values include a list of variables or 'all'.  Defaults to 'all'.
        data_type (Literal['pandas', 'xarray']): Return type for the data.

    Returns:
        Union[pd.DataFrame, xr.Dataset]: USAFR met data.
    """
    # Parse header information and column names from the first six lines.
    column_names, attrs = _parse_usafr_headers(filepath)

    # Get the file basename, which also contains mission date information.
    file_basename = os.path.basename(filepath)
    attrs['filename'] = file_basename

    # Remaining data is space-delimited.
    met_df = pd.read_csv(filepath,
                         sep=r'\s+',
                         names=column_names,
                         skiprows=6)

    # Convert time column (HH:MM:SS) to a datetime index using the start date.
    mission_start_date = _parse_usafr_filename_date(file_basename)
    met_df = _usafr_met_time_to_datetime(met_df, mission_start_date)

    # Subset the DataFrame columns. Return a Dataset if data_type == 'xarray'.
    if data_vars != 'all':
        met_df = met_df.loc[:, data_vars]
    else:
        pass
    if data_type == 'xarray':
        met_ds = met_df.to_xarray()
        met_ds.attrs.update(attrs)
        met_ds = _assign_usafr_met_variable_attrs(met_ds)
        return met_ds
    elif data_type == 'pandas':
        return met_df
    else:
        raise ValueError(f'{data_type} not supported.')


def read_noaa_met_directory(
    directory: str,
    file_type: str = 'nc',
    data_vars: Union[str, List] = 'all',
    **concat_kwargs,
) -> xr.Dataset:
    """
    Read a directory of NOAA P-3 met data files and concatenate into a Dataset.

    NOAA flight-level met data are available at:
    https://www.aoml.noaa.gov/{year}-hurricane-field-program-data/
    where {year} is the year corresponding to the season of data collection.
    The met data file date should have the format {YYYYMMDD}{ID}{#}.nc where
    Y, M, D are the year, month, and day of the flight, respectively, ID is the
    aircraft identifier (H for N42 and I for N43), and # is the flight number.
    For example, the file 20230829I1_A.nc was collected by the NOAA P-3 N43
    during the first mission of August 29, 2023.

    Note: P-3 met datasets contain many variables (>600).  Providing a subset
    of desired variables to `data_vars` can speed up the read-in process,
    since these variables are extracted prior to passing the dataset operations
    which require reading in the dataset into memory.

    Args:
        directory (str): Directory containing the met data files.
        file_type (str, optional): Met data file type. Defaults to '.nc'.
        data_vars (str or List): Variables to load into memory. Acceptable
            values include a list of variables or 'all'.  Defaults to 'all'.
        concat_kwargs (optional): Additional keyword arguments are passed to
            the xarray.concat.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found
            inside of `directory`.

    Returns:
        xr.Dataset: all met data files concatenated into a single Dataset.
    """
    met_files = glob.glob(os.path.join(directory, '*' + file_type))
    met_files.sort()

    met = {}
    if not met_files:
        raise FileNotFoundError(
            f'No files of type "{file_type}" found in "{directory}". '
            'Please double check the directory and file_type.')

    for file in met_files:
        met_ds = read_noaa_met_file(file, data_vars)
        base_file = os.path.basename(file)
        key = base_file.split('.')[0]
        met[key] = met_ds

    concat_dim = 'Time'

    met_ds_concat = xr.concat(list(met.values()),
                              dim=concat_dim,
                              combine_attrs=_combine_noaa_attrs,  #  TODO: implement this!
                              **concat_kwargs)

    return met_ds_concat


def read_noaa_met_file(
    filepath: str,
    data_vars: Union[str, List] = 'all',
) -> xr.Dataset:
    """
    Read NOAA P-3 met data as an xarray Dataset.

    Note: P-3 met datasets contain many variables (>600).  Providing a subset
    of desired variables to `data_vars` can speed up the read-in process,
    since these variables are extracted prior to passing the dataset operations
    which require reading in the dataset into memory.

    Args:
        filepath (str): Path to met data file.
        data_vars (str or List): Variables to load into memory. Acceptable
            values include a list of variables or 'all'.  Defaults to 'all'.

    Returns:
        xr.Dataset: P-3 met dataset.
    """
    met_ds = xr.open_dataset(filepath, decode_times=False)
    if data_vars != 'all':
        met_ds = met_ds[data_vars]
    else:
        pass
    met_ds = _noaa_met_time_to_datetime(met_ds)
    return met_ds


def read_ndbc_file(
    filepath: str,
    data_vars: Union[str, List] = 'all',
) -> xr.Dataset:
    """ Read archival NDBC netCDF file as an xarray dataset.

    Archival data are available at:
    https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:NDBC-CMANWx

    Args:
        filepath (str): Path to NDBC .nc file.

    Returns:
        xr.Dataset: NDBC data.
    """
    # Opening some NDBC datasets may raise an exception due to a
    # conflict with `wave_frequency_bounds`.
    try:
        ndbc_ds = xr.open_dataset(filepath)
    except MissingDimensionsError as error:
        ndbc_ds = xr.open_dataset(filepath, drop_variables='wave_frequency_bounds')
        print(error)

    if data_vars != 'all':
        ndbc_ds = ndbc_ds[data_vars]
    return ndbc_ds


def read_ndbc_met_file(filepath: str) -> pd.DataFrame:
    """ Read NDBC standard meteorological data into a DataFrame.

    Args:
        filepath (str): Path to NDBC standard meteorological data files.

    Returns:
        DataFrame: NDBC meteorological data.
    """
    datetime_name_mapping = {
        'YY': 'year',
        'MM': 'month',
        'DD': 'day',
        'hh': 'hour',
        'mm': 'minute',
    }
    # Read column names from the first row and strip comment characters.
    column_names = pd.read_csv(filepath, sep=r'\s+', nrows=0).columns.tolist()
    column_names = [name.strip('#') for name in column_names]

    # Read data rows into a DataFrame.
    ndbc_df = pd.read_csv(filepath,
                          sep=r'\s+',
                          names=column_names,
                          skiprows=2)

    # Rename and assign datetime columns as a datetime index.
    ndbc_df_renamed = ndbc_df.rename(columns=datetime_name_mapping)
    ndbc_df_renamed['datetime'] = pd.to_datetime(
        ndbc_df_renamed[datetime_name_mapping.values()],
        utc=True,
    )
    return (ndbc_df_renamed
            .drop(columns=datetime_name_mapping.values())
            .set_index('datetime'))


# TODO: update:
def _resample_met_vars(
    met_ds: xr.Dataset,
    data_vars: List,
    resample_times: np.ndarray[np.datetime64],
    resample_method: Callable
) -> xr.Dataset:

    # Resample each variable onto `resample_times`.  Aggregate observations
    # in a 50 s window centered on each time using `resample_method`.
    var_dict = {var: [] for var in data_vars}
    for t in resample_times:
        t_start = t - pd.Timedelta(25, 'S')
        t_end = t + pd.Timedelta(25, 'S')
        met_in_window = met_ds.sel(Time=slice(t_start, t_end))
        for var, values in var_dict.items():
            values.append(resample_method(met_in_window[var].values))

    # Construct a new Dataset using the resampled variables and new times
    met_resampled_ds = xr.Dataset(
        data_vars={
            var: (['time'], values) for var, values in var_dict.items()
        },
        coords=dict(
            time=resample_times,
        ),
    )

    # Copy attributes from the original DataArray(s)
    for var in var_dict.keys():
        met_resampled_ds[var].attrs = met_ds[var].attrs

    return met_resampled_ds


def _parse_usafr_headers(filepath: str) -> Tuple[List, dict]:
    """ Parse USAFR met file headers for column names and attributes."""
    file = open(filepath)
    column_names = []
    attrs = {}
    for i, line in enumerate(file):
        if i == 0:  # 1st line
            attrs['information'] = line
        elif i == 1:
            attrs['ARWO Version'] = line.split(':')[-1].strip()
        elif i == 2:
            attrs['File Version'] = line.split(':')[-1].strip()
        elif i == 3:
            attrs['Tail Number'] = line.split(':')[-1].strip()
        elif i == 4:
            pass
        elif i == 5:
            column_names = _parse_usafr_columns(line)
        elif i > 5:
            break
        else:
            raise ValueError('Unexpected number of lines in file header.')

    file.close()
    return column_names, attrs


def _parse_usafr_columns(header_line: str) -> List:
    """ Parse USAFR met file header line for column names. """
    # Some column names have a space, which should be replaced.
    if 'GMT Time' in header_line:
        header_line = header_line.replace('GMT Time', 'GMT_Time')
    if 'V V' in header_line:
        header_line = header_line.replace('V V', 'VV')
    return header_line.split()


def _parse_usafr_filename_date(file_basename) -> str:
    """ Return the date, in YYYYMMDD format, from a USAFR met filename. """
    # Use a simple regex to search for the yyyymmdd date pattern.
    match = re.search(r'\d{8}', file_basename)
    if match:
        date_str = match.group()
    else:
        raise ValueError(f'Could not parse date from filename: {file_basename}')
    return date_str


def _usafr_met_time_to_datetime(
        met_df: pd.DataFrame,
        mission_start_date: str,
    ) -> pd.DataFrame:
    """Convert USAFR met data time column to datetimes.

    Convert the USAFR met data time column 'GMT_Time', provided as hour of the
    day (with no day specified), to a datetime array.

    Args:
        met_df (pd.DataFrame): USAFR met DataFrame with original time column.
        mission_start_date (str): Mission start date in YYYYMMDD format.

    Returns:
        pd.DataFrame: met DataFrame with datetime index.
    """
    # Add the start date to the times in the 'GMT_Time' column.
    datetime_str_series = mission_start_date + ' ' + met_df['GMT_Time']
    datetime_series = pd.to_datetime(datetime_str_series, format='%Y%m%d %H:%M:%S', errors='raise')

    # Since the `GMT_Time` column is just the hour of the day, it is possible
    # the times will wrap around to the next day. Assume times are sorted.
    met_df['GMT_Time'] = _unwrap_datetimes(datetime_series)

    return met_df.set_index('GMT_Time')


def _noaa_met_time_to_datetime(met_ds: xr.Dataset) -> xr.Dataset:
    """Convert NOAA met data time coordinate to datetimes.

    Convert the P-3 met data time coordinate, which are provided as
    seconds since the start of the flight, to a datetime array.

    Args:
        met_ds (xr.Dataset): P-3 met dataset with original time coordinate.

    Returns:
        xr.Dataset: P-3 met dataset with datetime coordinate.
    """
    # Drop NaNs and sort the Dataset by time (seconds from start of flight).
    met_ds = (met_ds
              .dropna(dim='Time', how='all', subset=['Time'])
              .sortby('Time'))

    # Convert seconds from start of flight to datetimes using the POSIX
    # timestamp stored in attributes.  Assign it as the new coordinate.
    start_datetime_posix = met_ds.attrs['StartTime']
    datetime_posix = start_datetime_posix + met_ds['Time']
    datetime = pd.to_datetime(datetime_posix, unit='s', origin='unix')
    met_ds['Time'] = datetime
    return met_ds


def _unwrap_datetimes(datetimes: pd.Series, period='1d'):
    """ Unwrap datetime series wrapped by `period`. Assumes sorted input. """
    current_datetimes = datetimes.copy()

    # Compare current datetime to the next datetime.  Wraps occur when the
    # difference is positive.
    next_datetimes = current_datetimes.shift(-1)
    is_time_wrap = (current_datetimes - next_datetimes) > pd.Timedelta(0)
    n_wraps = is_time_wrap.sum()

    # Remove wraps by recursively adding `period` to the first instance.
    if n_wraps > 0:
        first_wrap = is_time_wrap.idxmax()
        is_wrapped = current_datetimes.index > first_wrap
        current_datetimes.loc[is_wrapped] += pd.Timedelta(period)
        return _unwrap_datetimes(current_datetimes)
    else:
        return current_datetimes


def _assign_usafr_met_variable_attrs(met_ds: xr.Dataset) -> xr.Dataset:
    """ Assign attributes to USAFR met Dataset variables."""
    # Case varies amongst USAFR met datasets.
    var_attrs = {k.lower(): v for k, v in USAFR_VAR_ATTRS.items()}

    # Assign attributes to each variable.
    for name, var in met_ds.items():
        if name.lower() in var_attrs.keys():
            var.attrs = var_attrs[name.lower()]
    return met_ds


def _combine_usafr_attrs(variable_attrs: List, context=None) -> dict:
    """ Concatenate USAFR attributes.

    Handle attributes during concatenation of USAFR met Datasets. Passed to
    xr.concat as the `combine_attrs` argument.

    Args:
        variable_attrs (List): Attribute dictionaries to combine.
        TODO: context (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: Combined attributes.
    """
    attr_keys = _get_unique_keys(variable_attrs)
    attrs = {k: [] for k in attr_keys}
    for key in attr_keys:
        if key == 'information':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'ARWO Version':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'File Version':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'Tail Number':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'filename':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        else:
            # TODO: could reduce list to single value with .pop?
            attrs[key] = _get_unique_attrs(variable_attrs, key)
    return attrs


def _combine_noaa_attrs(variable_attrs: List, context=None) -> dict:
    """Concatenate NOAA metadata attributes.

    Handle attributes during concatenation of met Datasets. Passed to
    xr.concat as the `combine_attrs` argument.

    Args:
        variable_attrs (List): Attribute dictionaries to combine.
        TODO: context (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: Combined attributes.
    """
    attr_keys = _get_unique_keys(variable_attrs)
    attrs = {k: [] for k in attr_keys}
    for key in attr_keys:
        if key == 'StartTime':
            attrs[key] = _attrs_to_datetime(variable_attrs, key, unit='s', origin='unix')#[0].isoformat()
        elif key == 'FlightDate':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'TimeInterval':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        else:
            attrs[key] = _get_unique_attrs(variable_attrs, key)
    return attrs


def _get_unique_keys(variable_attrs):
    """ Return unique keys from a set of attributes """
    # return [key for key in {key:None for attrs in variable_attrs for key in attrs}]
    return list({key: None for attrs in variable_attrs for key in attrs})


def _get_unique_attrs(variable_attrs, key) -> List:
    """ Return unique values from a set of attributes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    return list(np.unique(all_attrs))  # TODO: try replacing with built-in set


def _aggregate_attrs(variable_attrs, key) -> List:
    """ Aggregate all attributes into a list """
    return [attrs[key] for attrs in variable_attrs if key in attrs.keys()]


def _attrs_to_datetime(variable_attrs, key, **kwargs) -> List:
    """ Convert date-like attributes to datetimes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    attrs_as_datetimes = pd.to_datetime(all_attrs, **kwargs)
    return list(attrs_as_datetimes.sort_values())

