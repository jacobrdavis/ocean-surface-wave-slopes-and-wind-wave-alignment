"""
COAMPS-TC functions.
"""


import numpy as np
import scipy

from src.array_helpers import is_sorted


def wind_direction(u_component: np.ndarray, v_component: np.ndarray):
    """ Calculate wind direction from components.

    Args:
        u_component (np.ndarray): E-W wind component (+E)
        v_component (np.ndarray): N-S wind component (+N)

    Returns:
        np.ndarray: wind direction in meterological convention (+ CW, 0 deg N)
    """
    direction_from_north_going_to = np.arctan2(u_component, v_component)
    return np.mod(180 + np.rad2deg(direction_from_north_going_to), 360)


def match_model_and_buoy_by_interpolation(
    buoy: dict,
    model: dict,
    temporal_tolerance: np.timedelta64 = np.timedelta64(30, 'm'),
    **interpn_kwargs,
):
    """
    Match model and buoy observations using linear interpolation in time
    and bilinear interpolation in space.

    Note: the `time` arrays in both `buoy` and `model` must be sorted.

    Args:
        buoy (dict): dictionary containing buoy coordinates 'time',
        'latitude', and 'longitude' where
            'time': np.array[datetime64]
            'latitude': np.array[float]
            'longitude': np.array[float]

        model (dict): dictionary containing models coordinates 'time',
        'latitude', and 'longitude' plus an additional key-value pair
        containing 'field' (np.array), the field variable to be matched
        onto the buoy coordinates; key names much match exactly.

        temporal_tolerance (np.timedelta64, optional): maximum allowable
        time difference between a model and observation point. Defaults
        to np.timedelta64(30, 'm').

        **interpn_kwargs: Remaining keyword arguments passed to scipy.interpn.

    Returns:
        np.ndarray: interpolated field values for each time in `buoy`.
    """
    if not is_sorted(model['time']) and is_sorted(buoy['time']):
        raise ValueError('Input `time` arrays must be sorted.')

    #TODO: replace with default_kwargs
    if 'method' not in interpn_kwargs:
        interpn_kwargs['method'] = 'linear'
    if 'bounds_error' not in interpn_kwargs:
        interpn_kwargs['bounds_error'] = True

    t_sort_indices = np.searchsorted(model['time'], buoy['time'])

    # Adjust the sort indices so that the final index is not greater
    # than the length of the array.  If so, replace with the last index.
    n = model['time'].size
    t_sort_indices[t_sort_indices >= n] = n - 1

    field_matches = []

    points = (model['latitude'], model['longitude'])
    for i, j in enumerate(t_sort_indices):

        time_difference = np.abs(buoy['time'][i] - model['time'][j])

        if time_difference > temporal_tolerance:
            value = np.nan
        else:
            x_i = (buoy['latitude'][i], buoy['longitude'][i])

            field_values_jm1 = model['field'][j-1]  # left
            field_values_j = model['field'][j]  # right

            bilinear_value_jm1 = scipy.interpolate.interpn(points,
                                                           field_values_jm1,
                                                           x_i,
                                                           **interpn_kwargs)

            bilinear_value_j = scipy.interpolate.interpn(points,
                                                         field_values_j,
                                                         x_i,
                                                         **interpn_kwargs)

            value = np.interp(buoy['time'][i].astype("float"),
                              np.array([model['time'][j-1],
                                        model['time'][j]]).astype("float"),
                              np.concatenate([bilinear_value_jm1,
                                              bilinear_value_j]))

        field_matches.append(value)

    return np.array(field_matches)

