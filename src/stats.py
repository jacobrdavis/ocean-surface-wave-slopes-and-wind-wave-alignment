"""
Statistics functions.
"""


from typing import List, Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic, distributions


def calculate_residuals(y: np.ndarray, x: np.ndarray, fit: Callable, *param):
    #TODO: Add docstring
    return y - fit(x, *param)


def root_mean_square_error(y, y_hat):
     """ Return the root mean square error. """
     bias = y_hat - y
     sq_err = bias**2
     return np.sqrt(sq_err.mean())


def percent_difference(a, b):
    #TODO:
    abs_difference = np.abs(a - b)
    mean = (a + b) / 2
    return 100 * abs_difference / mean


def parameter_confidence_intervals(popt, pcov, n, alpha=0.05):
    """ Approximate parameter confidence intervals. """
    # Calculate one standard deviation errors on the parameter.
    perr = np.sqrt(np.diag(pcov))

    # Student's t value
    p = len(popt)  # number of parameters
    dof = max(0, n - p) #  degrees of freedom
    t_value = distributions.t.ppf(1.0 - alpha / 2.0, dof)

    # Approximate confidence intervals
    one_sided_err = t_value * perr
    popt_upper = popt + one_sided_err
    popt_lower = popt - one_sided_err
    return perr, t_value, popt_upper, popt_lower


def print_fit_parameters(fit_dict, parameter_names=None, print_format='.5f'):
    #TODO: Add docstring
    for i, p in enumerate(fit_dict['popt']):
        if parameter_names is None:
            name = f'p{i}'
        else:
            name = parameter_names[i]
        print(f'{name}: {p:{print_format}} +/- {fit_dict['t_value']*fit_dict["perr"][i]:{print_format}}')


def binned_statistic_df(df, x, y, stat='mean', bin_width=5, min_bin=None, max_bin=None, bins=None):
    x_values = df[x].values
    y_values = df[y].values

    if bins is None:
        if min_bin is None:
            min_bin = np.round(x_values.min() + bin_width/2)
        if max_bin is None:
            max_bin = np.round(x_values.max() - bin_width/2)
        bins = np.arange(min_bin, max_bin, bin_width)

    bin_means, bin_edges = binned_statistic(x_values,
                                            y_values,
                                            statistic=stat,
                                            bins=bins)[0:2]
    bin_std = binned_statistic(x_values, y_values, statistic='std', bins=bins)[0]

    # bin_width = (bin_edges[1] - bin_edges[0])
    # bin_centers = bin_edges[1:] - bin_width/2
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

    return bins_to_series(bin_means, bin_std, bin_centers, bin_edges)


def bins_to_series(bin_means, bin_stds, bin_centers, bin_edges):
    bins = {'bin_stat': bin_means,
            'bin_stds': bin_stds,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges}
    return pd.Series(bins, index=bins.keys())
