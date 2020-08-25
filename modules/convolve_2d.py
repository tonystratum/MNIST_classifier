import numpy as np

from modules.im2col import im2col_sliding_strided


def convolve_2d(input, W, stride=1):
    output_shape = (input.shape[-1] - W[0].shape[0]) // stride + 1

    output = []

    for filtered in np.split(input, input.shape[0], axis=0):
        img = filtered.reshape(
            filtered.shape[-2],
            filtered.shape[-1]
        )

        img_ = im2col_sliding_strided(img, W[0].shape, stride)  # is the one

        W_flattened = np.concatenate([np.reshape(filter, (1, -1)) for filter in W], axis=0)

        a_ = np.dot(W_flattened, img_)
        a = a_.copy()
        a.resize((len(W), output_shape, output_shape))
        output.append(a)

    return np.concatenate(output, axis=0)
