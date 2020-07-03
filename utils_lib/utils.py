"""
@author: rz
@email: r.zemblys@tf.su.lt
"""

import numpy as np


def round_up_to_odd(f, min_val=3):
    """Rounds input value up to nearest odd number.
    Parameters:
        f       --  input value
        min_val --  minimum value to retun
    Returns:
        Rounded value
    """
    w = np.int32(np.ceil(f) // 2 * 2 + 1)
    w = min_val if w < min_val else w
    return w


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
