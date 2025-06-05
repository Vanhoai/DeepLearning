import numpy as np

mean = np.array([(1, 1), (1, 6), (6, 1), (6, 6)])
cov = np.array([[1, 0], [0, 1]])

def prepare(num, groups):
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


def datasets(N, classes, d):
    X, y = prepare(num=int(N / classes), groups=classes)

    X = X.T
    y = y.reshape(-1)

    # One-hot encoding
    Y = np.zeros((N, classes))
    for i in range(N):
        Y[i, int(y[i])] = 1

    assert X.shape == (N, d)  # (N, dimensions)
    assert Y.shape == (N, classes)  # (N, classes)

    return X, Y