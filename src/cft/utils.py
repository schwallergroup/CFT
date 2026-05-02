"""
cft.utils
~~~~~~~~~
General-purpose utility helpers.
"""
import ast

import numpy as np


def parse_vec(s):
    """Parse a string representation of a numeric vector into a numpy array.

    Accepts both comma-separated ``"[1, 2, 3]"`` and space-separated
    ``"[1 2 3]"`` formats.

    Parameters
    ----------
    s : str
        String to parse.

    Returns
    -------
    numpy.ndarray, shape (N,), dtype float
    """
    s = s.strip()
    if "," in s:
        return np.array(ast.literal_eval(s), dtype=float)
    return np.fromstring(s.strip("[]"), sep=" ")
