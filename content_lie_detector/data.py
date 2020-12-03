import os
import pandas as pd


def get_data(file="full", nb_samples=5_000):

    path = os.path.dirname(os.path.dirname(__file__)) + '/raw_data/'

    file = file + ".csv"

    data = pd.read_csv(path + file)

    if nb_samples < data.shape[0]:

        tmp = data.sample(n=nb_samples)
        n_0, n_1 = tmp.target.value_counts()

        while n_0 / n_1 < 0.8 or n_0 / n_1 > 1.2:
            tmp = data.sample(n=nb_samples)

        data = tmp.dropna()
        del tmp

    X = data["title"] + " " + data["content"]
#     X = data["title"]
    y = data["target"]

    print("Loaded file -- > " + file)

    return X, y
