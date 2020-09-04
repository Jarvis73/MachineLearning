# Copyright 2019-2020 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================
"""
References: https://github.com/nlml/tsne_raw
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common import mnist_kits

SEED = 1
state = np.random.RandomState(SEED)


def square_euc_dist(X):
    """ (a - b)^2 = a^2 + b^2 - 2ab

    Parameters
    ----------
    X: np.ndarray
        with shape [N, D]

    Returns
    -------
    dist: np.ndarray
        with shape [N, D]
    """
    sum_X = np.sum(X ** 2, axis=1, keepdims=True)
    sq_dist = sum_X + sum_X.T - 2 * (X @ X.T)
    return sq_dist


def softmax(X, diag_zero=True, eps=1e-8, zero_index=None):
    # Subtract max for numerical stability (necessary!!!)
    exp = np.exp(X - np.max(X, axis=1, keepdims=True))
    if np.any(np.isinf(exp)):
        raise ValueError

    if zero_index is None:
        if diag_zero:
            np.fill_diagonal(exp, 0)
    else:
        exp[:, zero_index] = 0

    exp += eps
    return exp / np.sum(exp, axis=1, keepdims=True)


def prob_matrix(dist, sigma=None, zero_index=None):
    if sigma is None:
        return softmax(-dist, zero_index=zero_index)
    else:
        return softmax(-dist / (2 * sigma.reshape(-1, 1) ** 2), zero_index=zero_index)


def perplexity(dist, sigma, zero_index):
    pjci = prob_matrix(dist, sigma, zero_index)
    entropy = -np.sum(pjci * np.log2(pjci), axis=1)
    return 2 ** entropy


def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000):
    guess = val = 0
    for i in range(max_iter):
        guess = (lower + upper) / 2
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess


def get_optimal_sigma(dist, target_perplexity):
    all_sigmas = []
    # One sigma for each point
    print("Search sigma ", end="")
    for i in range(dist.shape[0]):
        sigma = binary_search(lambda sigma_: perplexity(dist[i:i + 1], np.array(sigma_), i), target_perplexity)
        all_sigmas.append(sigma)
        if i % (dist.shape[0] // 10) == 0:
            print(".", end="")
    print()
    return np.array(all_sigmas)


def p_joint(X, target_perplexity):
    dist = square_euc_dist(X)
    sigmas = get_optimal_sigma(dist, target_perplexity)
    prob = prob_matrix(dist, sigmas)
    return (prob + prob.T) / (2 * prob.shape[0])


def q_joint(Y):
    dist = np.exp(-square_euc_dist(Y))
    np.fill_diagonal(dist, 0)
    # Sum over all the entries
    return dist / np.sum(dist), None


def symmetric_SNE_grad(P, Q, Y, _):
    PQ_diff = np.expand_dims(P - Q, axis=-1)        # [N, N, 1]
    Y_diff = Y[:, None] - Y[None]                   # [N, N, 2]
    grad = 4 * np.sum(PQ_diff * Y_diff, axis=1)     # [N, 2]
    return grad


def q_tSNE(Y):
    dist = square_euc_dist(Y)
    inv_dist = 1. / (1 + dist)
    np.fill_diagonal(inv_dist, 0.)
    # Sum over all the entries
    return inv_dist / np.sum(inv_dist), inv_dist


def tSNE_grad(P, Q, Y, inv_dist):
    PQ_diff = np.expand_dims(P - Q, axis=-1)        # [N, N, 1]
    Y_diff = Y[:, None] - Y[None]                   # [N, N, 2]
    inv_dist = inv_dist[:, :, None]                 # [N, N, 1]
    grad = 4 * np.sum(PQ_diff * Y_diff * inv_dist, axis=1)     # [N, 2]
    return grad


#########################################################################################


def scatter_plot(X, y, figsize=(12, 6), show=True, save=None):
    if not show and not save:
        return
    fig, ax = plt.subplots(figsize=figsize)
    classes = list(np.unique(y))
    markers = 'os' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

    for i, c in enumerate(classes):
        ax.scatter(*X[y == c].T, marker=markers[i], c=[colors[i]], label=str(c), alpha=0.6)
    ax.legend()
    ax.axis("off")
    fig.set_facecolor((0., 0., 0.))

    if save:
        plt.tight_layout()
        plt.savefig(save, facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.show()


def fit_tSNE(X, y, P, iters, q_fn, grad_fn, lr, momentum, plot, save=False, prefix=""):
    # Initlize low-dimension data
    Y = state.normal(0, 1e-4, size=[X.shape[0], 2])

    if momentum:
        Ym1 = Y.copy()  # Y minus 1
        Ym2 = Y.copy()  # Y minus 2

    # Start loop
    for i in range(iters):
        # Get Q
        Q, dist = q_fn(Y)
        grads = grad_fn(P, Q, Y, dist)

        # Update Y
        Y -= lr * grads
        if momentum:
            Y += momentum * (Ym1 - Ym2)
            Ym2 = Ym1
            Ym1 = Y.copy()

        if plot and i % (iters // plot) == 0:
            if not save:
                scatter_plot(Y, y)
            else:
                scatter_plot(Y, y, save=prefix + str(i))

        if i % 20 == 0:
            print(i)

    return Y


def main():
    data_dir = Path(__file__).parents[1] / "data"
    X, y = mnist_kits.load_mnist(data_dir, digits=list(range(10)), n=1000)
    P = p_joint(X, target_perplexity=20)
    # Y = fit_tSNE(X, y, P, iters=501, q_fn=q_joint, grad_fn=symmetric_SNE_grad, lr=10., momentum=0.9, plot=5, save=True, prefix="SNE-")
    Y = fit_tSNE(X, y, P, iters=501, q_fn=q_tSNE, grad_fn=tSNE_grad, lr=10., momentum=0.9, plot=5, save=True, prefix="tSNE-")


if __name__ == "__main__":
    main()
