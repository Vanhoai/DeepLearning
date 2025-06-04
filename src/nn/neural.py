import numpy as np
import matplotlib.pyplot as plt

from src.nn.model import NeuralNetwork
from src.nn.layer import Dense
from src.nn.activation import ReLU, Sigmoid
from src.nn.loss import BinaryCrossEntropy
from src.nn.optimizer import SGD

mean = np.array([(1, 1), (10, 10)])
cov = np.array([[1, 0], [0, 1]])


def prepare(num: int = 500, groups=2):
    np.random.seed(5)
    X = np.zeros((num * groups, 2))
    y = np.zeros((num * groups))

    for idx, center in enumerate(mean):
        points = np.random.multivariate_normal(mean=center, cov=cov, size=num)
        X[idx * num:(idx + 1) * num, :] = points
        y[idx * num:(idx + 1) * num] = idx

    datas = X.T
    labels = y.reshape(1, -1)

    # shuffle the data
    indices = np.arange(datas.shape[1])
    np.random.shuffle(indices)

    return datas[:, indices], labels[:, indices]

#
# n = 1000
# g = 2
# d = 2
#
# X_train, y_train = prepare(num=int(n / g), groups=g)
# assert X_train.shape == (d, n)
# assert y_train.shape == (1, n)
#
# nn = NeuralNetwork(
#     layers=[
#         (Dense(units=3), ReLU()),
#         (Dense(units=3), ReLU()),
#         (Dense(units=1), Sigmoid())
#     ],
#     loss=BinaryCrossEntropy(),
#     optimizer=SGD(lr=0.01)
# )
#
# epoch = 10
# nn.fit(X_train, y_train, epoch)
