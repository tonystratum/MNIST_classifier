import numpy as np

from modules.im2col import im2col_sliding_strided


def max_pool_greedy(input):
    """
    Returns a maximum value for each 2x2 FOV with a stride of 2.
    (Looped implementation)
    :param input: numpy.ndarray of shape (n, n)
    :return: pooled numpy.ndarray of shape (n // 2, n // 2)
    """
    orig_shape = input.shape
    pad_shape = [s + 1 for s in input.shape]
    new_shape = [s // 2 for s in pad_shape]

    input_padded = np.zeros(pad_shape)
    input_padded[:input.shape[0], :input.shape[1]] = input
    input_str = im2col_sliding_strided(input_padded, (2, 2), stepsize=1)

    aaa = np.hsplit(input_str, orig_shape[0])
    skipped = [array[:, ::2] for array in aaa[::2]]
    proc_conc = np.concatenate(skipped, axis=1)
    pooled = np.max(proc_conc, axis=0)
    final = np.reshape(pooled, new_shape)
    return final


def max_pool_not_greedy(input):
    """
    Returns a maximum value for each 2x2 FOV with a stride of 2.
    (Unlooped implementation)
    :param input: numpy.ndarray of shape (n, n)
    :return: pooled numpy.ndarray of shape (n // 2, n // 2)
    """
    orig_shape = input.shape
    pad_shape = (input.shape[0] + 1, input.shape[1] + 1)
    new_shape = (pad_shape[0] // 2, pad_shape[1] // 2)

    input_padded = np.zeros(pad_shape)
    input_padded[:input.shape[0], :input.shape[1]] = input
    input_str = im2col_sliding_strided(input_padded, (2, 2), stepsize=1)

    aaa = np.hsplit(input_str, orig_shape[0])
    skipped = np.stack(aaa, axis=0)[::2, :, ::2]
    proc_conc = np.concatenate(skipped[:], axis=1)
    pooled = np.max(proc_conc, axis=0)
    final = np.reshape(pooled, new_shape)
    return final
