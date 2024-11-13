"""
Water wave functions.
"""

import warnings
from typing import Tuple, Optional, Union

import numpy as np
from scipy.optimize import newton

from src import array_helpers


GRAVITY = 9.81  # (m/s^2)
TWO_PI = 2 * np.pi


def mean_square_slope(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate spectral mean square slope as the fourth moment of the one-
    dimensional frequency spectrum.

    Note:
        This function requires that the frequency dimension is along the last
        axis of `energy_density`.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (..., f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        min_frequency (float, optional): lower frequency bound.
        max_frequency (float, optional): upper frequency bound.

    Returns:
    Mean square slope as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (..., f).
    """
    energy_density = np.asarray(energy_density)
    frequency = np.asarray(frequency)

    if min_frequency is None:
        min_frequency = frequency.min()

    if max_frequency is None:
        max_frequency = frequency.max()

    # Mask frequencies outside of the specified range.
    frequency_mask = np.logical_and(frequency >= min_frequency,
                                    frequency <= max_frequency)
    frequency = frequency[frequency_mask]
    energy_density = energy_density[..., frequency_mask]

    # Calculate the fourth moment of the energy density spectrum.
    fourth_moment = spectral_moment(energy_density=energy_density,
                                    frequency=frequency,
                                    n=4,
                                    axis=-1)
    return (TWO_PI**4 * fourth_moment) / (GRAVITY**2)


def wavenumber_mean_square_slope(
    energy_density_wn: np.ndarray,
    wavenumber: np.ndarray,
    min_wavenumber: Optional[float] = None,
    max_wavenumber: Optional[float] = None,
) -> Union[float, np.ndarray]:

    """
    Calculate mean square slope as the second moment of the one-dimensional
    wavenumber spectrum.

    Note:
        This function requires that the wavenumber dimension is along the last
        axis of `energy_density_wn`.

    Args:
        energy_density_wn (np.ndarray): 1-D energy density wavenumber spectrum
            with shape (k,) or (..., k).
        wavenumber (np.ndarray): 1-D wavenumbers with shape (k,).
        min_wavenumber (float, optional): lower wavenumber bound.
        max_wavenumber (float, optional): upper wavenumber bound.

    Returns:
    Mean square slope as a
        float: if the shape of `energy_density` is (k,).
        np.ndarray: if the shape of `energy_density` is (..., k).
    """
    energy_density_wn = np.asarray(energy_density_wn)
    wavenumber = np.asarray(wavenumber)

    if min_wavenumber is None:
        min_wavenumber = wavenumber.min()

    if max_wavenumber is None:
        max_wavenumber = wavenumber.max()

    # Mask wavenumbers outside of the specified range.
    wavenumber_mask = np.logical_and(wavenumber >= min_wavenumber,
                                     wavenumber <= max_wavenumber)
    wavenumber = wavenumber[wavenumber_mask]
    energy_density_wn = energy_density_wn[..., wavenumber_mask]

    # Calculate the second moment of the energy density wavenumber spectrum.
    return np.trapz(y=energy_density_wn * wavenumber**2, x=wavenumber, axis=-1)


def energy_period(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    return_as_frequency: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate energy-weighted frequency as the ratio of the first and zeroth
    moments of the one-dimensional frequency spectrum.

    Note:
        This function requires that the frequency dimension is along the last
        axis of `energy_density`.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (..., f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        return_as_frequency (bool): if True, return frequency in Hz.

    Returns:
    Energy-weighted period as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (..., f).

    """
    energy_density = np.asarray(energy_density)
    frequency = np.asarray(frequency)

    # Ratio of the 1st and 0th moments is equilvaent to 0th moment-
    # weighted frequency.
    energy_frequency = moment_weighted_mean(arr=frequency,
                                            energy_density=energy_density,
                                            frequency=frequency,
                                            n=0,
                                            axis=-1)
    if return_as_frequency:
        return energy_frequency
    else:
        return energy_frequency**(-1)


def spectral_moment(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    n: float,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    """ Compute the 'nth' spectral moment.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum.
        frequency (np.ndarray): 1-D frequencies.
        n (float): Moment order (e.g., `n=1` is returns the first moment).
        axis (int, optional): Axis to calculate the moment along. Defaults
            to -1.

    Returns:
    The nth moment as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if `energy_density` has more than one dimension.  The shape
            of the returned array is reduced along `axis`.
    """
    frequency_n = frequency ** n
    moment_n = np.trapz(energy_density * frequency_n, x=frequency, axis=axis)
    return moment_n


def moment_weighted_mean(
    arr: np.ndarray,
    energy_density: np.ndarray,
    frequency: np.ndarray,
    n: float,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    """ Compute the 'nth' moment-weighted mean of an array.

    Note:
        The `energy_density` and `arr` arrays must have the same shape.

    Args:
        arr (np.ndarray): Array to calculate the moment-weighted mean of.
        energy_density (np.ndarray): 1-D energy density frequency spectrum.
        frequency (np.ndarray): 1-D frequencies.
        n (float): Moment order (e.g., `n=1` is returns the first moment).
        axis (int, optional): Axis to calculate the moment along. Defaults
            to -1.

    Returns:
    The result of weighting `arr` by the nth moment as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if `energy_density` has more than one dimension.  The shape
            of the returned array is reduced along `axis`.
    """
    moment_n = spectral_moment(energy_density=energy_density,
                               frequency=frequency,
                               n=n,
                               axis=axis)

    weighted_moment_n = spectral_moment(energy_density=energy_density * arr,
                                        frequency=frequency,
                                        n=n,
                                        axis=axis)
    return weighted_moment_n / moment_n


def significant_wave_height(
    energy_density: np.ndarray,
    frequency: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Calculate significant wave height as four times the square root of the
    spectral variance.

    Note:
        This function requires that the frequency dimension is along the last
        axis of `energy_density`.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (..., f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).

    Returns:
    Significant wave height as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (..., f).

    """
    energy_density = np.asarray(energy_density)
    frequency = np.asarray(frequency)

    zeroth_moment = spectral_moment(energy_density=energy_density,
                                    frequency=frequency,
                                    n=0,
                                    axis=-1)

    return 4 * np.sqrt(zeroth_moment)


def direction(a1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """ Return the frequency-dependent direction from the directional moments.

    Calculate the direction at each frequency from the first two Fourier
    coefficients of the directional spectrum (see Sofar and Kuik et al.).

    References:
        Sofar (n.d.) Spotter Technical Reference Manual

        A J Kuik, G P Van Vledder, and L H Holthuijsen (1988) "A method for the
        routine analysis of pitch-and-roll buoy wave data" JPO, 18(7), 1020-
        1034, 1988.

    Args:
        a1 (np.ndarray): Normalized spectral directional moment (+E).
        b1 (np.ndarray): Normalized spectral directional moment (+N).

    Returns:
        np.ndarray: Direction at each spectral frequency in the metereological
            convention (degrees clockwise from North).
    """
    return (270 - np.rad2deg(np.arctan2(b1, a1))) % 360


def directional_spread(a1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """ Return the frequency-dependent directional spread from the moments.

    Calculate the direction at each frequency from the first two Fourier
    coefficients of the directional spectrum (see Sofar and Kuik et al.).

    References:
        Sofar (n.d.) Spotter Technical Reference Manual

        A J Kuik, G P Van Vledder, and L H Holthuijsen (1988) "A method for the
        routine analysis of pitch-and-roll buoy wave data" JPO, 18(7), 1020-
        1034, 1988.

    Args:
        a1 (np.ndarray): Normalized spectral directional moment (+E).
        b1 (np.ndarray): Normalized spectral directional moment (+N).

    Returns:
        np.ndarray: Directional spread at each spectral frequency in degrees.
    """
    directional_spread_rad = np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2)))
    return np.rad2deg(directional_spread_rad)


def merge_frequencies(
    spectrum: np.ndarray,
    n_merge: int,
) -> np.ndarray:
    """ Merge neighboring frequencies in a spectrum.

    Note:
        This function requires that the frequency dimension is along the last
        axis of `spectrum`.

    Args:
        spectrum (np.ndarray): 1-D frequency spectrum with shape (f,)
            or shape (..., f).
        n_merge (int): number of adjacent frequencies to merge.

    Returns:
        np.ndarray: merged spectrum with shape (f//n_merge,) or with
            shape (..., f//n_merge).

    Example:
    ```
    >>> frequency = np.arange(0, 9, 1)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    >>> energy_density = frequency * 3
    array([ 0,  3,  6,  9, 12, 15, 18, 21, 24])

    >>> merge_frequencies(frequency, n_merge=3)
    array([1., 4., 7.])

    >>> merge_frequencies(energy_density, n_merge=3)
    array([ 3., 12., 21.])
    ```
    """
    n_frequencies = spectrum.shape[-1]
    n_groups = n_frequencies // n_merge
    spectrum_merged = np.apply_along_axis(array_helpers.average_n_groups,
                                          axis=-1,
                                          arr=spectrum,
                                          n_groups=n_groups)
    return spectrum_merged


def sliding_energy_weighted_direction(
    energy_density: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Return a smoothed, frequency-dependent direction by computing the
    energy-weighted direction in sliding windows.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (..., f).
        a1 (np.ndarray): normalized spectral directional moment (+E) with
            shape (..., f).
        b1 (np.ndarray): normalized spectral directional moment (+N) with
            shape (..., f).
        window_size (int): size of the sliding window.

    Returns:
        np.ndarray: smoothed direction in degrees with shape (..., f).
    """
    window_kwargs = {'window_size': window_size, 'constant_value': np.NaN}
    energy_windows = array_helpers.padded_sliding_window(energy_density, **window_kwargs)
    a1_windows = array_helpers.padded_sliding_window(a1, **window_kwargs)
    b1_windows = array_helpers.padded_sliding_window(b1, **window_kwargs)
    a1_weighted = _energy_weighted_nanmean(a1_windows, energy_windows, axis=-1)
    b1_weighted = _energy_weighted_nanmean(b1_windows, energy_windows, axis=-1)
    return direction(a1_weighted, b1_weighted)


def _energy_weighted_nanmean(arr, energy_density, **kwargs):
    """ Return the energy weighted mean ignoring NaNs. """
    weighted = np.nansum(arr * energy_density, **kwargs)
    moment_0 = np.nansum(energy_density, **kwargs)
    return weighted / moment_0


def fq_energy_to_wn_energy(
    energy_density_fq: np.ndarray,
    frequency: np.ndarray,
    depth: Union[float, np.ndarray] = np.inf,
    var_rtol: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Transform energy density from frequency to wavenumber space.

    Transform energy density, defined in the frequency domain, to energy
    density on a wavenumber domain using the appropriate Jacobians:

    E(k) = E(w) dw/dk

    and

    E(w) = E(f) df/dw

    where dw/dk is equivalent to the group velocity and df/dw = 1/(2pi) [1].
    This conversion relies on the (inverse) linear dispersion relationship to
    calculate wavenumbers from the provided frequencies.

    References:
        1. L. H. Holthuijsen (2007) Waves in Oceanic and Coastal Waters,
        Cambridge University Press

    Args:
        energy_density_fq (np.ndarray): 1-D energy density frequency spectrum
            with shape (f,) or (n, f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        depth (float, optional): water depth (positive down) with shape (n,).
            Defaults to np.inf.
        var_rtol (float, optional): relative tolerance passed to np.isclose
            to check variance equality. Defaults to 0.02.

    Returns:
        Tuple[np.ndarray, np.ndarray]: energy density in wavenumber domain and
            the corresponding wavenumbers.
    """
    if np.isscalar(depth):
        depth = np.array([depth])

    wavenumber = inverse_dispersion(frequency, depth).squeeze()
    dw_dk = intrinsic_group_velocity(wavenumber, frequency, depth)
    df_dw = 1 / (2*np.pi)
    energy_density_wn = energy_density_fq * df_dw * dw_dk
    var_match = check_spectral_variance(energy_density_wn, wavenumber,
                                        energy_density_fq, frequency,
                                        rtol=var_rtol)

    not_var_match = np.logical_not(var_match)
    if np.any(not_var_match):
        energy_density_wn[not_var_match, :] = np.full_like(wavenumber, np.nan)
        n_mismatches = not_var_match.sum()
        warnings.warn(
            f'There are {n_mismatches} entries with a variance mismatch. '
            f'Filling mismatched entries with NaN.',
            category=RuntimeWarning,
        )
    return energy_density_wn, wavenumber


def check_spectral_variance(
    energy_density_wn: np.ndarray,
    wavenumber: np.ndarray,
    energy_density_fq: np.ndarray,
    frequency: np.ndarray,
    **kwargs
) -> Union[bool, np.ndarray]:
    """Check for variance equality between wavenumber and frequency spectra.

    Note: Tolerances are specified using the absolute (atol) and relative
    (rtol) tolerance arguments passed to np.isclose via **kwargs.

    Args:
        energy_density_wn (np.ndarray): energy density in wavenumber domain.
        wavenumber (np.ndarray): wavenumber array.
        energy_density_fq (np.ndarray): energy density in frequency domain.
        frequency (np.ndarray): frequency array.
        **kwargs: Additional kwarg arguments are passed to np.isclose.

    Returns:
        bool: True if variance matches within tolerance.
    """
    var_wn = np.trapz(energy_density_wn, x=wavenumber)
    var_fq = np.trapz(energy_density_fq, x=frequency)
    return np.isclose(var_wn, var_fq, **kwargs)


def intrinsic_group_velocity(
    wavenumber: np.ndarray,
    frequency: Optional[np.ndarray] = None,
    depth: float = np.inf,
) -> np.ndarray:
    """ Compute the intrinsic group velocity.

    Note:
        Wavenumber and frequency are used in the phase velocity calculation.
        If `frequency` is not provided, it is calculated from `wavenumber`
        using the intrinsic dispersion relationship.

    Args:
        wavenumber (np.ndarray): 1-D wavenumbers with shape (k,).
        frequency (Optional[np.ndarray], optional): 1-D frequencies with shape
            (k,). Defaults to None.
        depth (float, optional): positive water depth. Defaults to np.inf.

    Returns:
        np.ndarray: Group velocities with shape (k,).
    """
    ratio = group_to_phase_ratio(wavenumber, depth)
    return ratio * phase_velocity(wavenumber, frequency, depth)


def intrinsic_dispersion(
    wavenumber: np.ndarray,
    depth: float = np.inf,
) -> np.ndarray:
    """ Return angular frequency from wavenumber using intrinsic dispersion.

    Note:
        This implementation of the dispersion relationship is valid in the
        intrinsic reference frame only. See `doppler_adjust` in kinematics.py.

    Args:
        wavenumber (np.ndarray): 1-D wavenumbers with shape (k,).
        depth (float, optional): positive water depth. Defaults to np.inf.

    Returns:
        np.ndarray: Angular frequencies in rad/s with shape (k,).
    """
    GRAVITY = 9.81
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return np.sqrt(gk * np.tanh(kh))  # angular frequency


def phase_velocity(
    wavenumber: np.ndarray,
    frequency: Optional[np.ndarray] = None,
    depth: float = np.inf,
) -> np.ndarray:
    """ Return the wave phase velocity.

    Note:
        If `frequency` is not provided, it is calculated from `wavenumber`
        using the intrinsic dispersion relationship.

    Args:
        wavenumber (np.ndarray): 1-D wavenumbers with shape (k,).
        frequency (Optional[np.ndarray], optional): 1-D frequencies with shape
            (k,). Defaults to None.
        depth (float, optional): positive water depth. Defaults to np.inf.

    Returns:
        np.ndarray: Phase velocities with shape (k,).
    """
    if frequency is None:
        angular_frequency = intrinsic_dispersion(wavenumber, depth)
    else:
        angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency / wavenumber


def inverse_dispersion(
    frequency: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
    use_limits: bool = False
) -> Union[float, np.ndarray]:
    """Solve the linear dispersion relationship for wavenumber.

    Given frequencies (in Hz) and water depths, solve the linear
    dispersion relationship for the corresponding wavenumbers, k. Uses a
    Newton-Rhapson root-finding implementation.

    Note:
        Expects inputs of matching shapes. The input `frequency` is the
        frequency in Hz and NOT the angular frequency, omega or w.

        `use_limits` might provide speed-up for very large f*d.

    Args:
        frequency (float or np.ndarray): containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.
        depth (float or np.ndarray): water depths with shape matching
            that of `frequency`.
        use_limits (bool, optional): solve the dispersion relation only where
            kh is outside of the deep and shallow water limits.

    Raises:
        ValueError: if `frequency` and `depth` are not of size (d,f) or of size
            (f,) and (d,), respectively.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """
    if use_limits:
        wavenumber = _dispersion_with_limits(frequency, depth)
    else:
        wavenumber = _dispersion_solver(frequency, depth)
    return wavenumber


def inverse_dispersion_array(
    frequency: np.ndarray,
    depth: np.ndarray,
    use_limits: bool = False
) -> np.ndarray:
    """Solve the linear dispersion relationship for wavenumber based on
    an array of frequencies.

    Given frequencies (in Hz) and water depths, solve the linear dispersion
    relationship for the corresponding wavenumbers, k. Uses a Newton-Rhapson
    root-finding implementation.

    Note:
        Expects input as numpy.ndarrays of shape (d,f) where f is the number
        of frequencies and d is the number of depths, or a `frequency` of shape
        (f,) and `depth` of shape (d,). If the latter, the inputs will be
        meshed to (d,f) ndarrays, assuming a uniform frequency vector for every
        depth provided. The input `frequency` is the frequency in Hz and NOT
        the angular frequency, omega or w.

        `use_limits` might provide speed-up for very large f*d.

    Args:
        frequency (np.ndarray): of shape (f,) or (d,f) containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.
        depth (np.ndarray): of shape (d,) or (d,f) containing water depths.
        use_limits (bool, optional): solve the dispersion relation only where
            kh is outside of the deep and shallow water limits.

    Raises:
        ValueError: if `frequency` and `depth` are not of size (d,f) or of size
            (f,) and (d,), respectively.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """
    frequency = np.asarray(frequency)
    depth = np.asarray(depth)

    # Check incoming shape; if 1-dimensional, map to an (f, d) mesh. Otherwise
    # the shape should already be (f, d). Raise exception for mixed shapes.
    if frequency.ndim == 1 and depth.ndim == 1:
        f = len(frequency)
        d = len(depth)
        frequency = np.tile(frequency, (d, 1))
        depth = np.tile(depth, (f, 1)).T

    elif frequency.ndim == 2 and depth.ndim == 1:
        d, f = frequency.shape
        depth = np.tile(depth, (f, 1)).T

    elif frequency.ndim == 1 and depth.ndim == 2:
        d, f = depth.shape
        frequency = np.tile(frequency, (d, 1))

    elif frequency.shape == depth.shape:
        pass

    else:  # if frequency.shape != depth.shape:
        raise ValueError(
            '`frequency` and `depth` must be either arrays of size '
            '(f,) and (d,) \n or ndarrays of the same shape. Given:'
            f' frequency.shape={frequency.shape}'
            f' and depth.shape={depth.shape}.')

    if use_limits:
        wavenumber = _dispersion_with_limits(frequency, depth)
    else:
        wavenumber = _dispersion_solver(frequency, depth)

    return wavenumber


def _dispersion_with_limits(
    frequency: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """ Solve the dispersion relation only where parameters are outside of the
    deep and shallow water limits.

    Approximates the wavenumber using both the deep and shallow water linear
    dispersion limits and checks against the `kh` limits:

        shallow:  kh < np.pi/10 (h < L/20)
           deep:  kh > np.pi    (h > L/2)

    Frequencies and depths outside of these limits are solved using
    a standard root-finding algorithm. This might provide speed-up for cases
    where the combined size of the number of depths and frequencies is very
    large, e.g., O(10^6) and above, since an iterative approach is not needed
    for `kh` at the tanh(kh) limits. Values close to the limits will be
    approximate.

    Args:
        frequency (float or np.ndarray): frequencies in [Hz]
        depth (float or np.ndarray): water depths

    Returns:
       float or np.ndarray: wavenumbers with shape matching input
    """
    # Initialize wavenumber array and assign values according to limits.
    wavenumber = np.empty(frequency.shape)
    wavenumber_shallow = shallow_water_inverse_dispersion(frequency, depth)
    wavenumber_deep = deep_water_inverse_dispersion(frequency)

    # These are commonly used limits, but they are still approximations.
    in_deep = wavenumber_deep * depth > np.pi
    in_shallow = wavenumber_shallow * depth < np.pi/10
    in_intermd = np.logical_and(~in_deep, ~in_shallow)

    if np.any(in_intermd):
        wavenumber_intermd = _dispersion_solver(frequency[in_intermd],
                                                depth[in_intermd])
    else:
        wavenumber_intermd = np.empty(frequency[in_intermd].shape)

    wavenumber[in_deep] = wavenumber_deep[in_deep]
    wavenumber[in_shallow] = wavenumber_shallow[in_shallow]
    wavenumber[in_intermd] = wavenumber_intermd

    return wavenumber


def deep_water_inverse_dispersion(frequency):
    """Computes wavenumber from the deep water linear dispersion relationship.

    Given frequencies (in Hz) solve the linear dispersion relationship in the
    deep water limit for the corresponding wavenumbers, k. The linear
    dispersion relationship in the deep water limit, tanh(kh) -> 1, has the
    closed form solution k = omega^2 / g and is (approximately) valid for
    kh > np.pi (h > L/2).

    Args:
        frequency (float or np.ndarray): of any shape containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.

    Returns:
       float or np.ndarray: wavenumbers with shape matching input
    """
    angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency**2 / GRAVITY


def shallow_water_inverse_dispersion(frequency, depth):
    """Computes wavenumber from shallow water linear dispersion.

    Given frequencies (in Hz) solve the linear dispersion relationship in the
    shallow water limit for the corresponding wavenumbers, k. The linear
    dispersion relationship in the shallow water limit, kh -> kh, has the
    closed form solution k = omega / sqrt(gh) and is (approximately) valid for
    kh < np.pi/10 (h < L/20).

    Args:
        frequency (float or np.ndarray): containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.
        depth (float or np.ndarray): water depths with shape matching
            that of `frequency`.

    Returns:
       float or np.ndarray: wavenumbers with shape matching input
    """
    angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency / np.sqrt(GRAVITY * depth)


def _dispersion_solver(
    frequency: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Solve the linear dispersion relationship.

    Solves the linear dispersion relationship w^2 = gk tanh(kh) using a
    Scipy Newton-Raphson root-finding implementation.

    Note:
        Expects input as a floats or ndarrays with matching shapes. The
        input `frequency` is the frequency in Hz and NOT the angular
        frequency, omega or w.

    Args:
        frequency (float or np.ndarray): containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.
        depth (float or np.ndarray): water depths with shape matching
            that of `frequency`.

    Returns:
       float or np.ndarray: wavenumbers with shape matching input
    """

    angular_frequency = frequency_to_angular_frequency(frequency)

    wavenumber_deep = deep_water_inverse_dispersion(frequency)

    wavenumber = newton(func=_dispersion_root,
                        x0=wavenumber_deep,
                        args=(angular_frequency, depth),
                        fprime=_dispersion_derivative)
    return wavenumber


def _dispersion_root(wavenumber, angular_frequency, depth):
    """ Dispersion relation rearranged for root finding. """
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return gk * np.tanh(kh) - angular_frequency**2


def _dispersion_derivative(wavenumber, angular_frequency, depth):
    """ First derivative of the dispersion relation w.r.t to wavenumber. """
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return GRAVITY * np.tanh(kh) + gk * depth * (1 - np.tanh(kh)**2)


def group_to_phase_ratio(
    wavenumber: np.ndarray,
    depth: float = np.inf,
) -> np.ndarray:
    """ Compute the ratio of group velocity to phase velocity.

    Note: to prevent overflows in `np.sinh`, the product of wavenumber and
    depth (relative depth) are used to assign ratios at deep or shallow limits:

        shallow:  Cg/Cp = 1.0 if kh < np.pi/10 (h < L/20)
           deep:  Cg/Cp = 0.5 if kh > np.pi    (h > L/2)

    Args:
        wavenumber (np.ndarray): 1-D wavenumbers with shape (k,).
        depth (float, optional): positive water depth. Defaults to np.inf.

    Returns:
        np.ndarray: of shape (k,) containing ratio at each wavenumber.
    """
    kh = wavenumber * depth
    in_deep, in_shallow, in_intermd = depth_regime(kh)
    ratio = np.empty(kh.shape)
    ratio[in_deep] = 0.5
    ratio[in_shallow] = 1.0
    ratio[in_intermd] = 0.5 + kh[in_intermd] / np.sinh(2 * kh[in_intermd])
    return ratio


def depth_regime(kh: np.ndarray) -> Tuple:
    """ Classify depth regime based on relative depth.

    Classify depth regime based on relative depth (product of wavenumber
    and depth) using the shallow and deep limits:

        shallow:  kh < np.pi/10 (h < L/20)
           deep:  kh > np.pi    (h > L/2)

    The depth regime is classified as intermediate if not at the deep or
    shallow limits.

    Args:
        kh (np.ndarray): relative depth of shape (k, )

    Returns:
        np.ndarray[bool]: true where kh is deep, false otherwise
        np.ndarray[bool]: true where kh is shallow, false otherwise
        np.ndarray[bool]: true where kh is intermediate, false otherwise
    """
    in_deep = kh > np.pi
    in_shallow = kh < np.pi/10
    in_intermd = np.logical_and(~in_deep, ~in_shallow)
    return in_deep, in_shallow, in_intermd


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency
