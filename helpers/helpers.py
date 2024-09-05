import numpy as np
import argparse

#from .fitting import show_wave

def draw_uniform(range):
    """
    Draw a random number from a given range using a uniform distribution.

    Args:
        range (tuple): A tuple of two numbers (min, max) defining the range.

    Returns:
        float: A random number uniformly distributed within the given range.

    Note:
        This function uses numpy's random number generator to ensure
        consistent random number generation across different platforms.
    """
    rng = np.random.default_rng()
    return range[0]+(range[1]-range[0])*rng.random()

def is_notebook() -> bool:
    """
    Check if the current environment is a Jupyter notebook.

    This function attempts to determine whether the code is being executed
    within a Jupyter notebook environment. It does this by checking for the
    presence and type of the IPython shell.

    Returns:
        bool: True if the code is running in a Jupyter notebook, False otherwise.

    Note:
        This function relies on the availability of IPython-specific attributes.
        If IPython is not available, it assumes the code is not running in a notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def argparser(parameters):
    """
    Parse command-line arguments to update parameters.

    This function creates an argument parser for each key in the provided parameters
    dictionary. It allows users to input parameters directly from the command line
    when running scripts. If executed in a Jupyter notebook environment, the function
    returns the original parameters without modification.

    Args:
        parameters (dict): A dictionary of default parameter values.

    Returns:
        dict: Updated parameters from command-line arguments if not in a notebook,
              otherwise returns the original parameters.

    Note:
        The function uses the `is_notebook()` function to determine the execution
        environment. Command-line arguments are ignored in notebook environments.
    """
    if not is_notebook():
        parser = argparse.ArgumentParser()
        for key in parameters:
            parser.add_argument(f'--{key}', type=type(parameters[key]), default=parameters[key])
        args = parser.parse_args()
        print("Updating parameters from arguments")
        return vars(args)
    else:
        return parameters