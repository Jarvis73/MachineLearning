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

import gzip
import pickle
import numpy as np
from urllib import request
from pathlib import Path

MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'


def _maybe_download(filename, directory, source_url):
    direc = Path(directory)
    direc.mkdir(parents=True, exist_ok=True)
    filepath = direc / filename
    if not filepath.exists():
        print('Downloading', filename, '...')
        filepath, _ = request.urlretrieve(source_url, filepath)
        print('Successfully downloaded', filename)
    return filepath


def load_mnist(datasets_path, digits, n):
    path = _maybe_download('mnist.pkl.gz', datasets_path, MNIST_URL)

    with gzip.open(path, "rb") as f:
        (images, labels), _, _ = pickle.load(f, encoding='latin1')

    includes_matrix = [(labels==i) for i in digits]
    keep_indices = np.sum(includes_matrix, 0).astype(np.bool)

    images, labels = [images[keep_indices], labels[keep_indices]]

    n = min(n, images.shape[0])
    return images[:n], labels[:n]
