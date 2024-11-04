"""
Utilities for handling namespace files.  A SimpleNamespace is used to map
user-defined variable names to the static variable names used in the src code.
"""

import types

import toml


def get_namespace():
    """ Load the namespace file. """
    with open('./src/namespace.toml', 'r') as f:
        config = toml.load(f)

    return config


def get_var_namespace(key='buoy', subset=None) -> types.SimpleNamespace:
    """ Return SimpleNamespace from the `key` table of the namespace file. """
    if subset is None:
        subset = 'vars'
    var_key_value_pairs = get_namespace()[key][subset]
    return types.SimpleNamespace(**var_key_value_pairs)
