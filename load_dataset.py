# TODO: refactor

from os import listdir
from PIL import Image

from random import shuffle

import numpy as np

# form a dataset

def load_dataset(path):
    test_path = path + '/testing'
    train_path = path + '/training'

    dataset = {'test': [],
               'train': []}

    for set_name in dataset.keys():
        set_path = '{}/{}{}'.format(path, set_name, 'ing')
        for digit in listdir(set_path):
            digit_path = '{}/{}'.format(set_path, digit)
            for im_name in listdir(digit_path):
                dataset[set_name].append(
                    (digit_path + '/' + im_name, digit)
                )

    ds = dataset.copy()

    shuffle(ds['train'])
    shuffle(ds['test'])

    new_shape = np.asarray(Image.open(ds['train'][0][0])).shape
    new_shape = new_shape[0] * new_shape[0]

    dataset = {'test': None,
               'train': None}

    for set_name in ds.keys():

        X = [np.asarray(
            Image.open(image_path)
        ).reshape(new_shape)[:, np.newaxis] / 255
             for image_path, Y_value in ds[set_name]
             ]
        X = np.concatenate(X, axis=1)

        Y_l = []
        for image_path, Y_value in ds[set_name]:
            Y = np.zeros((10, 1))
            Y[int(Y_value)][0] = 1.
            Y_l.append(Y)
        Y = np.concatenate(Y_l, axis=1)

        dataset[set_name] = {'X': X,
                             'Y': Y}

    return dataset



