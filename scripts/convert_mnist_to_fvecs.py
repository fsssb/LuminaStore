#!/usr/bin/env python3
"""Convert the tf-keras MNIST npz into fvecs files for ann_bench.

Usage: convert_mnist_to_fvecs.py <mnist.npz> <out_dir>
Writes base.fvecs (train, 60k x 784) and query.fvecs (test, 10k x 784),
normalized to [0,1] (L2 distance on pixel intensities).
"""

import os
import struct
import sys

import numpy as np


def write_fvecs(path, data):
    n, dim = data.shape
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", dim))
            f.write(data[i].astype("<f4").tobytes())
    print(f"wrote {path}: {n} x {dim}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    npz = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(npz)
    x_train = d["x_train"].astype(np.float32).reshape(d["x_train"].shape[0], -1) / 255.0
    x_test = d["x_test"].astype(np.float32).reshape(d["x_test"].shape[0], -1) / 255.0
    write_fvecs(os.path.join(out_dir, "base.fvecs"), x_train)
    write_fvecs(os.path.join(out_dir, "query.fvecs"), x_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
