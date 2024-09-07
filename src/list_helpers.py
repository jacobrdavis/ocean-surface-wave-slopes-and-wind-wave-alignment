"""
List operation helper functions.
"""


from typing import List


def list_difference(list_a: List, list_b: List) -> List:
    """
    Return the difference between elements of `list_a` and `list_b` while
    maintaining order.
    """
    return [x for x in list_a if x not in list_b]


def list_intersection(list_a: List, list_b: List) -> List:
    """
    Return the intersection of`list_a` with `list_b` while preserving order.
    """
    set_2 = frozenset(list_b)
    return [x for x in list_a if x in set_2]