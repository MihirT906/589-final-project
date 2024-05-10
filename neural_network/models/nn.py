import numpy as np
import itertools as it
import pandas as pd
from eval import metrics
import matplotlib.pyplot as plt


def pad(x):  # (batch, n, m) -> (batch, n+1, m)
    return np.pad(x, [(0, 0), (1, 0), (0, 0)], constant_values=1)


def trunc(x):  # (batch, n, m) -> (batch, n - 1, m)
    return x[:, 1:, :]


def T(x):  # (batch, n, m) -> (batch, m, n)
    return np.transpose(x, (0, 2, 1))


def zero(M):
    M[:, 0] = 0
    return M


def sigmoid(x):
    return np.piecewise(x, [x > 0], [
        lambda i: 1 / (1 + np.exp(-i)),
        lambda i: np.exp(i) / (1 + np.exp(i))
    ])


def cross_entropy(a, y):
    return -1/y.shape[1] * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))


class NN:
    def __init__(self, l, decay=0, act=sigmoid):
        self.l = l
        self.w = [np.random.normal(size=(y, x + 1))
                  for x, y in it.pairwise(self.l)]
        self.decay = decay
        self.act = act

    def cost(self, P, y):
        j = np.mean([cross_entropy(p, y_i) for p, y_i in zip(P, y)])
        return j + self.decay * sum(np.sum(w[:, 1:] ** 2) for w in self.w) / (2 * y.shape[0])

    def forward(self, x):
        a = []
        for w in self.w:
            a.append(x := pad(x))
            x = self.act(w @ x)
        a.append(x)
        return a

    def backward(self, a, y):
        d = [a[-1] - y]
        for w_i, a_i in zip(reversed(self.w[1:]), reversed(a[1:-1])):
            d.append(trunc(w_i.T @ d[-1] * a_i * (1 - a_i)))
        return reversed(d)

    def backprop(self, X, y):
        a = self.forward(X)
        d = self.backward(a, y)
        return [d_i @ T(a_i) for d_i, a_i in zip(d, a[:-1])]

    def update(self, grad, lr):
        self.w = [w - lr * g for w, g in zip(self.w, grad)]

    def step(self, X, y, lr):
        grad = self.backprop(X, y)
        reg = map(zero, [self.decay * w for w in self.w])

        grad = [(1 / X.shape[0]) * (np.sum(g, axis=0) +
                p) for g, p in zip(grad, reg)]

        self.update(grad, lr)
        return grad

    def train(self, X, y, lr, epochs, num_batches=1):
        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)

        for _ in range(epochs):
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]
            Xb, yb = map(lambda x: np.array_split(x, num_batches), (X, y))
            for Xb_i, yb_i in zip(Xb, yb):
                self.step(Xb_i, yb_i, lr)

    def predict(self, X):
        return self.forward(X)[-1]

    def evaluate(self, X, y, type='multiclass'):
        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)

        y_pred = self.predict(X)

        y_c, p_c = map(lambda x: np.squeeze(np.argmax(x, axis=1)), (y, y_pred))
        acc = np.mean(y_c == p_c)

        if type == 'multiclass':
            f1 = metrics.multiclass_eval(y_c, p_c)[2]
        else:
            f1 = metrics.binary_eval(y_c, p_c)[2]

        return [self.cost(y_pred, y), acc, f1]


def one_hot(y, columns=None):
    return pd.get_dummies(y, columns=columns, dtype=int).to_numpy()


def minmax(X, x_max, x_min):
    return (X - x_min) / (x_max - x_min)


def stratified_k_fold_eval(X, y, k, layers, decay, eval_type='multiclass', normalize=False, *args, **kwargs):
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]

    y_groups = [np.array_split(idx[y == i], k) for i in np.unique(y)]
    [np.random.shuffle(l) for l in y_groups]
    folds = [np.concatenate(k) for k in zip(*y_groups)]

    y = one_hot(y)

    metrics = []
    for fold in folds:
        nn = NN(layers, decay=decay)
        mask = np.in1d(idx, fold)
        X_te, y_te = X[mask], y[mask]
        X_tr, y_tr = X[~mask], y[~mask]
        
        if normalize:
            x_max, x_min = np.max(X_tr, axis=0), np.min(X_tr, axis=0)
            X_tr, X_te = minmax(X_tr, x_max, x_min), minmax(X_te, x_max, x_min)

        nn.train(X_tr, y_tr, *args, **kwargs)
        metrics.append(nn.evaluate(X_te, y_te, eval_type))

    return np.mean(metrics, axis=0)


def make_cost_graph(fname, nn, X, y, lr, epochs, step):
    y = one_hot(y)
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    X_tr, y_tr = X[:int(0.7 * X.shape[0])], y[:int(0.7 * X.shape[0])]
    X_te, y_te = X[int(0.7 * X.shape[0]):], y[int(0.7 * X.shape[0]):]

    X_tr = np.expand_dims(X_tr, axis=-1)
    y_tr = np.expand_dims(y_tr, axis=-1)

    costs = [nn.evaluate(X_te, y_te)[0]]
    for _ in range(epochs):
        idx = np.random.permutation(X_tr.shape[0])
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        for i in range(0, X_tr.shape[0]-step, step):
            nn.step(X_tr[i:i+step], y_tr[i:i+step], lr)
            costs.append(nn.evaluate(X_te, y_te)[0])
    fig, ax = plt.subplots()
    ax.plot(step * np.arange(len(costs)), costs)

    ax.set_xlabel("Number of Examples")
    ax.set_ylabel("Cost (J)")
    fig.savefig(fname)
