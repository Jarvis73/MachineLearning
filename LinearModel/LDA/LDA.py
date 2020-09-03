#####################################################
#
#   Linear Discriminant Analysis
#
#   Machine Learning , Zhou Zhiwei
#
#   Chapter 3. Linear model, exercise 3.5
#
#   Target: Implement LDA, and test on watermelon dataset 3.0 alpha
#
#   Writen by Jarvis (zjw.math@qq.com)
#
#   Date: 2017.05.03
#

import numpy as np
import matplotlib.pylab as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(4325)


def fix_data(number=50):
    mean = np.array([[-2.96308878,  2.94349617],
                     [-2.03249244, -2.0312322 ]])
    cov = np.array([[[2.52722827, 1.39863059],
                     [1.39863059, 3.14037231]],
                    [[1.3235537,  0.52337228],
                     [0.52337228, 1.23186163]]])
    data = {}
    for i in range(2):
        data[i] = np.random.multivariate_normal(mean[i], cov[i], size=number * (i + 1))
    return data


def toy_data(classes=2, number=50):
    mean = np.random.uniform(-10, 10, size=(classes, 2))
    print("mean:", mean)
    # Positive-definite
    t = np.random.rand(classes, 2, 2)
    t = t @ t.transpose(0, 2, 1)
    cov = t + t.transpose(0, 2, 1)
    cov[:, [1, 0], [0, 1]] /= 2
    print("cov:", cov)

    data = {}
    for i in range(classes):
        data[i] = np.random.multivariate_normal(mean[i], cov[i], size=number)
    return data


def LDA(X1, X2):
    """
    Parameters
    ----------
    X1: np.ndarray
        sample dataset, positive
    X2: np.ndarray
        sample dataset, negative

    Returns
    -------
    w: np.ndarray
        coefficient
    """
    # np.cov treats rows as variables and columns as observations
    Sw = np.cov(X1, rowvar=False) * (X1.shape[0] - 1) + np.cov(X2, rowvar=False) * (X2.shape[0] - 1)
    # compute inverse by svd decomposition
    u, s, vh = np.linalg.svd(Sw)
    Sw_inv = vh.T @ np.diag(1. / s) @ u.T

    w = Sw_inv.dot(X1.mean(axis=0) - X2.mean(axis=0))
    return w


def plot_wm(X1, X2, w, w2):
    plt.scatter(X1[:, 1], X1[:, 0], s=100, marker=(4, 0), facecolors="b")
    plt.scatter(X2[:, 1], X2[:, 0], s=100, marker=(3, 0), facecolors="r")
    x = np.linspace(-8, 8, 2)
    y = w[1] / w[0] * x    # w[0]*y + w[1]*x = 0  <==  W^T.X = 0
    plt.plot(x, y)
    y = w2[1] / w2[0] * x    # w[0]*y + w[1]*x = 0  <==  W^T.X = 0
    plt.plot(x, y)

    plt.title('LDA')
    # plt.savefig("LDA.png")
    plt.axis("square")
    plt.show()


def main():
    data = fix_data(number=10)
    # data = toy_data(classes=2, number=100)
    w2 = LDA(data[0], data[1])
    lda = LinearDiscriminantAnalysis()
    lda.fit(np.r_[data[0], data[1]], np.r_[np.array([0] * len(data[0])), np.array([1] * len(data[1]))])
    w = lda.coef_[0]
    print(lda.coef_[0], w2)
    plot_wm(data[0], data[1], w, w2)
    # for i, v in data.items():
    #     plt.scatter(v[:, 1], v[:, 0])
    # plt.show()


if __name__ == "__main__":
    main()
