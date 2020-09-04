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

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

from common import mnist_kits

SEED = 1
state = np.random.RandomState(SEED)


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
    fig.set_facecolor('k')

    if save:
        plt.tight_layout()
        plt.savefig(save, facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.show()


def main():
    data_dir = Path(__file__).parents[1] / "data"
    X, y = mnist_kits.load_mnist(data_dir, digits=list(range(10)), n=1000)
    X_tSNE = TSNE(n_components=2, perplexity=20, learning_rate=10., n_iter=501, random_state=state, verbose=1)
    Y = X_tSNE.fit_transform(X, y)
    scatter_plot(Y, y, save="lib-tSNE")


if __name__ == "__main__":
    main()
