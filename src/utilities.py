"""
Utilities for handling namespace files.
"""


import types

import pandas as pd
import toml


def get_namespace():
    with open('./src/namespace.toml', 'r') as f:
        config = toml.load(f)

    return config


def get_var_namespace(key='buoy', subset=None) -> types.SimpleNamespace:
    if subset is None:
        subset = 'vars'
    vars = get_namespace()[key][subset]
    var_namespace = types.SimpleNamespace(**vars)

    return var_namespace


#  TODO: Move to a more appropriate module?
def create_time_slice(start_time, time_delta = pd.Timedelta('1h')):
    end_time = start_time + time_delta
    time_slice = slice(start_time, end_time)
    return time_slice


#  TODO: Move to a more appropriate module?
def strip_time_slice_tz(time_slice):
    start_slice = time_slice.start.tz_localize(None)
    stop_slice = time_slice.stop.tz_localize(None)
    return slice(start_slice, stop_slice)
