"""
Array operation helper functions.
"""


from typing import Tuple, Optional, Union

import numpy as np


def average_n_groups(arr: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Split an array into `n_groups` groups and return the average of each group.

    Adapted from Divakar via https://stackoverflow.com/questions/53178018/
    average-of-elements-in-a-subarray.

    Functionally equivalent to (but faster than):
    arr_split = np.array_split(arr, n_groups)
    arr_merged = np.array([group.mean() for group in arr_split])
    """
    n = len(arr)
    m = n // n_groups
    w = np.full(n_groups, m)
    w[:n - m*n_groups] += 1
    sums = np.add.reduceat(arr, np.r_[0, w.cumsum()[:-1]])
    return sums / w


def padded_sliding_window(
    arr: np.ndarray,
    window_size: int,
    constant_value: float = np.NaN
) -> np.ndarray:
    """ Return a sliding window view of an array with padding.

    The array is padded with `constant_value` distributed evenly on either side
    but preferentially appending to the left side if the number of pad elements
    is not divisible by 2. After padding, the sliding window view has as many
    windows as the length of the input array.

    Args:
        arr (np.ndarray): The array to be padded and windowed.
        window_size (int): The size of the sliding window.
        constant_value (float, optional): A constant value to use for padding.
            Defaults to np.NaN.

    Returns:
        np.ndarray: _description_
    """
    # Preferentially append to the left side if n_pad is not divisible by 2.
    n_pad = window_size - 1
    q, m = divmod(n_pad, 2)
    r_pad = q
    l_pad = q + m
    arr_padded = np.pad(arr, (l_pad, r_pad), constant_values=constant_value)
    return np.lib.stride_tricks.sliding_window_view(arr_padded, window_size)


def append_last(arr: np.ndarray) -> np.ndarray:
    """ Append the last value of an array to itself. """
    return np.append(arr, arr[-1])


def is_sorted(arr: np.ndarray) -> bool:
    """ Return True if array is sorted. """
    return bool(np.all(arr[:-1] <= arr[1:]))
