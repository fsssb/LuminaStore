#!/usr/bin/env python3
"""Convert an ann-benchmarks HDF5 dataset to fvecs files.

Usage: convert_hdf5_to_fvecs.py <dataset.hdf5> <out_dir>
Writes <out_dir>/base.fvecs (train) and <out_dir>/query.fvecs (test).
"""

import os
import struct
import sys

import h5py
import numpy as np


def write_fvecs(path, data):
    """data: (n, dim) float32 array -> fvecs binary."""
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
    h5_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)

    write_fvecs(os.path.join(out_dir, "base.fvecs"), train)
    write_fvecs(os.path.join(out_dir, "query.fvecs"), test)

    # Ground truth for recall verification (top-100 per query).
    gt_path = os.path.join(out_dir, "truth.npy")
    np.save(gt_path, neighbors)
    print(f"wrote {gt_path}: {neighbors.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
