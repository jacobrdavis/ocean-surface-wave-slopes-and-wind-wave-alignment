"""
Buoy kinematics functions.
"""


import warnings
from typing import Tuple

import pandas as pd
import numpy as np

from src import array_helpers, geodesy


ACCEL_GRAVITY = 9.81  # m/s^2
KMPH_TO_MPS = 0.277778  # m/s / km/hr


def drift_speed_and_direction(
    longitude: np.ndarray,
    latitude: np.ndarray,
    time: pd.DatetimeIndex,
    append: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate the drift speed and direction from adjacent positions.

    Note: this function assumes that `longitude` and `latitude` are subsequent
    positions sorted by `time`.

    Args:
        longitude (np.ndarray): Longitudes in decimal degrees with shape (t,).
        latitude (np.ndarray): Latitudes in decimal degrees with shape (t,).
        time (pd.DatetimeIndex): Sorted time index with shape (t,).
        append (bool, optional): If True, repeat the last value such that the
            output and input arrays are the same length. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: drift speed in m/s and drift direction
            in degrees with shapes (t-1,) or (t,) if append == True.
    """
    if not array_helpers.is_sorted(time):
        warnings.warn(
            'Input `time` is not sorted. Trailing values may be incorrect.'
        )

    # Difference the input times to obtain time deltas (in hours).
    time_difference_sec = time[1:] - time[0:-1]
    time_difference_hr = time_difference_sec.seconds / 3600
    time_difference_hr = time_difference_hr.to_numpy()

    # Compute the great circle distance and true bearing between each position.
    dist_km, drift_dir_deg = geodesy.great_circle_pathwise(longitude=longitude,
                                                            latitude=latitude)

    # Calculate drift magnitude; convert from km/hr to m/s.
    drift_speed_kmph = dist_km/time_difference_hr
    drift_speed_mps = drift_speed_kmph * KMPH_TO_MPS

    # If `append` is truthy, append the last value to maintain input size (n,).
    if append:
        drift_speed_mps = array_helpers.append_last(drift_speed_mps)
        drift_dir_deg = array_helpers.append_last(drift_dir_deg)

    return drift_speed_mps, drift_dir_deg


def drift_speed_components(drift_speed, drift_dir_deg):
    #TODO: need to define orientation
    #TODO: need to validate
    east_drift_speed = drift_speed * np.sin(np.deg2rad(drift_dir_deg))
    north_drift_speed = drift_speed * np.cos(np.deg2rad(drift_dir_deg))
    return east_drift_speed, north_drift_speed


def doppler_adjust(
    energy_density_obs: np.ndarray,
    frequency_obs: np.ndarray,
    drift_speed: float,
    drift_direction_going: float,
    wave_direction_coming: np.ndarray,
    frequency_cutoff: float,
    interpolate: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Doppler adjust a 1-D spectrum observed on a moving platform to the
    intrinsic reference frame using the omnidirectional solutions described in
    Collins et al. (2017) and Colosi et al. (2023).

    Adapted from the map_omni_dir_spectrum.m source code for Colosi et al.
    (2023) available at: https://github.com/lcolosi/WaveSpectrum/tree/main.

    References:
    Collins, C. O. et al. (2017) “Doppler Correction of Wave Frequency Spectra
        Measured by Underway Vessels” Journal of Atmospheric and Oceanic
        Technology https://doi.org/10.1175/JTECH-D-16-0138
    Colosi, Luke et al. (2023) “Observations of Surface Gravity Wave Spectra
        from Moving Platforms.” Journal of Atmospheric and Oceanic Technology
        https://doi.org/10.1175/JTECH-D-23-0022.1.

    Args:
        energy_density_obs (np.ndarray): 1-D energy density frequency spectrum
            with shape (f,).
        frequency_obs (np.ndarray): 1-D frequencies in the observed reference
            frame with shape (f,).
        drift_speed (float): platform drift speed.
        drift_direction_going (float): platform drift direction in degrees.
        wave_direction_coming (np.ndarray): wave direction in degrees at each
            frequency with shape (f,).
        frequency_cutoff (float): maximum frequency measurable by the platform.
        interpolate (bool, optional): If True, interpolate the intrinsic energy
            densities onto the observed frequency bins. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: intrinsic energy density and frequency
            arrays with shapes (f,) and (f,).
    """
    # Compute drift-wave misalignment.
    wave_direction_going = coming_to_going(wave_direction_coming, modulus=360)
    misalignment_deg = wave_drift_alignment(wave_direction_going,
                                            np.array([drift_direction_going]))
    misalignment_rad = np.deg2rad(misalignment_deg).squeeze()

    # Compute the Doppler shift speed projection.
    cos_misalignment = np.cos(misalignment_rad)
    projected_speed = drift_speed * cos_misalignment

    # Compute intrinsic frequency handling each frequency by branch.
    frequency_int = np.full(frequency_obs.shape, np.nan)

    # Branch 1: moving against waves
    is_branch_1 = (cos_misalignment < 0) & (drift_speed > 0)
    frequency_int[is_branch_1] = _branch_1(frequency_obs[is_branch_1],
                                           projected_speed[is_branch_1])
    # Branch 2: moving normal to waves
    is_branch_2 = (cos_misalignment == 0) | (drift_speed == 0)
    frequency_int[is_branch_2] = _branch_2(frequency_obs[is_branch_2])

    # Branch 3: moving with waves
    is_branch_3 = (cos_misalignment > 0) & (drift_speed > 0)
    frequency_int[is_branch_3] = _branch_3(frequency_obs[is_branch_3],
                                           projected_speed[is_branch_3],
                                           frequency_cutoff)

    # Approximate Jacobian using central finite differencing and map observed
    # energy density to intrinsic energy density.
    jacobian = np.gradient(frequency_obs, frequency_int)
    energy_density_int = energy_density_obs * jacobian

    # If True, interpolate the intrinsic energy density onto the observed
    # frequency array.
    if interpolate:
        energy_density_int = np.interp(x=frequency_obs,
                                       xp=frequency_int,
                                       fp=energy_density_int)
        frequency_int = frequency_obs.copy()
    else:
        pass

    return energy_density_int, frequency_int  #, misalignment_deg, jacobian, projected_speed


def _frequency_int_solution(frequency_obs, projected_speed):
    """
    Doppler adjust observed frequencies to intrinsic frequencies for a
    platform moving in waves. This is equation (7) in Colosi et al. (2023).
    """
    g = ACCEL_GRAVITY
    discriminant = g**2 - 8*np.pi * g * projected_speed * frequency_obs
    denominator = 4*np.pi * projected_speed
    return (g - np.sqrt(discriminant)) / denominator


def _branch_1(frequency_obs, projected_speed):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    against waves.
    """
    return _frequency_int_solution(frequency_obs, projected_speed)


def _branch_2(frequency_obs):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    normal to waves (or not at all).
    """
    return frequency_obs


def _branch_3(frequency_obs, projected_speed, frequency_cutoff):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves.
    """
    # Compute intrinsic frequency at the observed frequency value where
    # df_int(frequency_obs)/df_obs tends towards infinity.
    frequency_int_max = ACCEL_GRAVITY / (4*np.pi * projected_speed)

    # Compute intrinsic frequency when observed frequency = 0.
    frequency_int_zero = ACCEL_GRAVITY / (2*np.pi * projected_speed)

    # Compute intrinsic frequency handling each frequency by case.
    frequency_int = np.full(frequency_obs.shape, np.nan)

    # Case 1: Platform moving slower than energy and crests
    is_case_1 = frequency_int_max > frequency_cutoff
    frequency_int[is_case_1] = _branch_3_case_1(frequency_obs[is_case_1],
                                                projected_speed[is_case_1],
                                                frequency_cutoff)

    # Case 2: Platform moving faster than energy but slower than crests
    is_case_2 = ((frequency_int_max > 0) &
                 (frequency_int_max < frequency_cutoff) &
                 (frequency_int_zero > frequency_cutoff))
    frequency_int[is_case_2] = _branch_3_case_2(frequency_obs[is_case_2],
                                                projected_speed[is_case_2])

    # Case 3: Platform moving faster than energy and crests
    is_case_3 = frequency_int_zero < frequency_cutoff
    frequency_int[is_case_3] = _branch_3_case_3(frequency_obs[is_case_3],
                                                projected_speed[is_case_3])

    return frequency_int


def _branch_3_case_1(frequency_obs, projected_speed, frequency_cutoff):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves (branch 3) when it is moving slower than the energy and crests.
    """
    # Compute the intrnisic frequency value where it will exceed the cutoff.
    frequency_obs_cutoff = -(2*np.pi * projected_speed * frequency_cutoff**2
                             / ACCEL_GRAVITY) + frequency_cutoff
    above_cutoff = frequency_obs > frequency_obs_cutoff

    # Compute intrinsic frequency and replace frequencies above the cutoff.
    frequency_int = _frequency_int_solution(frequency_obs, projected_speed)
    frequency_int[above_cutoff] = np.NaN
    return frequency_int


def _branch_3_case_2(frequency_obs, projected_speed):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves (branch 3) when it is moving faster than the energy but slower
    than the crests.
    """
    # Compute observed frequency value where df_int(frequency_obs)/df_obs tends
    # towards infinity.
    frequency_obs_max = ACCEL_GRAVITY / (8*np.pi * projected_speed)
    above_cutoff = frequency_obs > frequency_obs_max

    # Compute intrinsic frequency and replace frequencies above the cutoff.
    frequency_int = _frequency_int_solution(frequency_obs, projected_speed)
    frequency_int[above_cutoff] = np.NaN
    return frequency_int


def _branch_3_case_3(frequency_obs, projected_speed):
    """
    Map observed frequencies to intrinsic frequencies for a platform moving
    with waves (branch 3) when it is moving faster than the energy and crests.
    """
    # Compute observed frequency value where df_int(frequency_obs)/df_obs tends
    # towards infinity.
    frequency_obs_max = ACCEL_GRAVITY / (8*np.pi * projected_speed)
    above_cutoff = frequency_obs > frequency_obs_max

    # Compute intrinsic frequency and replace frequencies above the cutoff.
    frequency_int = _frequency_int_solution(frequency_obs, projected_speed)
    frequency_int[above_cutoff] = np.NaN
    return frequency_int


def wave_drift_alignment(
    wave_direction_going: np.ndarray,
    drift_direction_going: np.ndarray,
) -> np.ndarray:
    """ Calculate the misalignment between wave and drift directions.

    TODO: update shapes
    Args:
        wave_direction_going (np.ndarray): Wave direction in "going to"
            convention with shape (...,f).
        drift_direction_going (np.ndarray): Drift direction in "going to"
            convention with shape (...,d).

    Returns:
        np.ndarray: misalignment_deg
    """
    misalignment_full_deg = drift_direction_going[:, None] - wave_direction_going
    misalignment_deg = (misalignment_full_deg + 180) % 360 - 180
    return misalignment_deg


def coming_to_going(coming_from: np.ndarray, modulus=360):
    """Helper function to convert "coming from" convention to "going to"."""
    going_to = (coming_from + 180) % modulus
    return going_to


def going_to_coming(going_to: np.ndarray, modulus=360):
    """Helper function to convert "going to" convention to "coming from"."""
    coming_from = (going_to - 180) % modulus
    return coming_from


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency


def angular_frequency_to_frequency(angular_frequency):
    """Helper function to convert angular frequency (omega) to frequency (f)"""
    return angular_frequency / (2 * np.pi)
