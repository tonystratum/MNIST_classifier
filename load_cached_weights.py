import pandas as pd
from os import listdir


def load_weights(cache_path):
    W_path = cache_path + 'W/'
    b_path = cache_path + 'b/'

    weights = {}
    for path in (W_path, b_path):
        weights[path[-2]] = {}
        for weight in listdir(path):
            weights[path[-2]][weight[:2].upper()] = pd.read_csv(path + weight).to_numpy()

    return weights
