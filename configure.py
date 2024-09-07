"""
Notebook configuration functions.
"""


import toml
from IPython import get_ipython


def get_config():
    with open('config.toml', 'r') as f:
        config = toml.load(f)

    return config


def read_stored_variable(variable_name: str):
    """ Read a variable stored by another ipynb instance.

    Read a variable from the current iPython InteractiveShell instance
    using the iPython `store -r`magic command.  The variable is evaluated
    upon return so that the variable is recognized in the development
    environment (e.g., by Pylance in VS code).
    """
    get_ipython().run_line_magic('store', '-r ' + variable_name)
    return eval(variable_name)
