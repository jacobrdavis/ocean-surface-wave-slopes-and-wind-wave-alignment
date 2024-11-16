"""
Utilities for handling time, unit conversions, and basic I/O.
"""

import pickle
from typing import Any

import pandas as pd


def create_time_slice(start_time, time_delta=pd.Timedelta('1h')) -> slice:
    """ Create a slice from `start_time` to `start_time + time_delta`. """
    end_time = start_time + time_delta
    time_slice = slice(start_time, end_time)
    return time_slice


def strip_time_slice_tz(time_slice):
    """ Return `time_slice` with timezone information removed. """
    start_slice = time_slice.start.tz_localize(None)
    stop_slice = time_slice.stop.tz_localize(None)
    return slice(start_slice, stop_slice)


def knots_to_mps(knots):
    """ Convert knots to meters per second. """
    return knots / 1.944


def nmi_to_m(nmi):
    """ Convert nautical miles to meters. """
    return nmi * 1852


def nmi_to_km(nmi):
    """ Convert nautical miles to kilometers. """
    return nmi * 1.852


def km_to_m(km):
    """ Convert kilometers to meters. """
    return km * 1000


def read_pickle(path: str) -> Any:
    """ Read a pickle file at `path`. """
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(variable: Any, path: str) -> None:
    """ Write a `variable` to a pickle file at `path`. """
    with open(path, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)
