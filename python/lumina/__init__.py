# LuminaStore Python binding (ctypes wrapper over the C API shared library).
#
# Zero third-party runtime deps except numpy (optional for batch paths).
# Load path resolution order:
#   1. $LUMINA_LIB
#   2. sibling build/lib (project-local)
#   3. system library path
#
# Example:
#   import lumina
#   col = lumina.open_collection("/tmp/db", dim=128, metric=0)
#   col.add(ids=np.arange(1000), vectors=vecs, payloads=["..." for _ in range(1000)])
#   hits = col.search(queries, top_k=10)
#   col.snapshot()
#   col.close()

import ctypes
import json
import os

import numpy as np

_DIM = 64  # reserved


def _load_library():
    candidates = []
    env = os.environ.get("LUMINA_LIB")
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    # project build dir (works from python/lumina when repo is built in build/)
    repo_root = os.path.dirname(os.path.dirname(here))
    candidates.append(os.path.join(repo_root, "build", "libluminastore_shared.dylib"))
    candidates.append(os.path.join(repo_root, "build", "libluminastore_shared.so"))
    candidates.append("libluminastore_shared.dylib")
    candidates.append("libluminastore_shared.so")

    last_err = None
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            last_err = e
    raise RuntimeError(f"cannot load luminastore_shared: {last_err}")


_lib = _load_library()

# ---- declare C signatures ----
_lib.lumina_open.restype = ctypes.c_void_p
_lib.lumina_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

_lib.lumina_close.restype = ctypes.c_int
_lib.lumina_close.argtypes = [ctypes.c_void_p]

_lib.lumina_add.restype = ctypes.c_int
_lib.lumina_add.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float),
                            ctypes.c_int, ctypes.c_char_p]

_lib.lumina_add_batch.restype = ctypes.c_int
_lib.lumina_add_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
]

_lib.lumina_remove.restype = ctypes.c_int
_lib.lumina_remove.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

_lib.lumina_search.restype = ctypes.c_void_p
_lib.lumina_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]

_lib.lumina_search_batch.restype = ctypes.c_void_p
_lib.lumina_search_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.lumina_get.restype = ctypes.c_int
_lib.lumina_get.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int)]

_lib.lumina_snapshot.restype = ctypes.c_int
_lib.lumina_snapshot.argtypes = [ctypes.c_void_p]

_lib.lumina_stats.restype = ctypes.c_void_p
_lib.lumina_stats.argtypes = [ctypes.c_void_p]

_lib.lumina_free_string.restype = None
_lib.lumina_free_string.argtypes = [ctypes.c_void_p]


def _as_float_array(vectors):
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class Collection:
    """Handle-based wrapper around a LuminaStore collection."""

    def __init__(self, path, dim, metric=0, auto_open=True):
        self.path = path
        self.dim = int(dim)
        self.metric = int(metric)
        self._h = None
        if auto_open:
            self.open()

    def open(self):
        self._h = _lib.lumina_open(self.path.encode(), self.dim, self.metric)
        if not self._h:
            raise RuntimeError(f"lumina_open failed for {self.path!r}")
        return self

    def close(self):
        if self._h:
            _lib.lumina_close(self._h)
            self._h = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _check(self):
        if not self._h:
            raise RuntimeError("collection is not open")

    # ---- writes ----

    def add(self, ids, vectors, payloads=None):
        """Add vectors (n, dim) with ids (n,) and optional payloads (list of str)."""
        self._check()
        ids = np.asarray(ids, dtype=np.uint64)
        arr, ptr = _as_float_array(vectors)
        if arr.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {arr.shape[1]}")
        n = len(ids)
        if n == 1:
            payload = payloads[0] if payloads else ""
            rc = _lib.lumina_add(self._h, int(ids[0]), ptr, self.dim, payload.encode())
        else:
            if payloads is None:
                payloads = [""] * n
            cpayloads = (ctypes.c_char_p * n)(*[p.encode() for p in payloads])
            rc = _lib.lumina_add_batch(self._h, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                                       ptr, n, self.dim, cpayloads)
        if rc != 0:
            raise RuntimeError(f"lumina_add failed (rc={rc})")

    def remove(self, ids):
        self._check()
        if isinstance(ids, (int, np.integer)):
            ids = [ids]
        for i in ids:
            _lib.lumina_remove(self._h, int(i))

    # ---- reads ----

    def search(self, queries, top_k=10, ef=200):
        """Search. Returns a list of list of {id, distance, payload}."""
        self._check()
        arr, ptr = _as_float_array(queries)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            single = True
        else:
            single = False
        if arr.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {arr.shape[1]}")
        n = arr.shape[0]
        raw = _lib.lumina_search_batch(self._h, ptr, n, self.dim, int(top_k))
        text = ctypes.string_at(raw).decode()
        _lib.lumina_free_string(raw)
        data = json.loads(text)
        results = data.get("results", [])
        return results[0] if single else results

    def get(self, id):
        self._check()
        p = ctypes.c_char_p()
        length = ctypes.c_int()
        rc = _lib.lumina_get(self._h, int(id), ctypes.byref(p), ctypes.byref(length))
        if rc != 0:
            return None
        try:
            return p.value.decode()
        finally:
            _lib.lumina_free_string(p)

    # ---- maintenance ----

    def snapshot(self):
        self._check()
        rc = _lib.lumina_snapshot(self._h)
        if rc != 0:
            raise RuntimeError(f"snapshot failed (rc={rc})")

    def stats(self):
        self._check()
        raw = _lib.lumina_stats(self._h)
        try:
            return json.loads(ctypes.string_at(raw).decode())
        finally:
            _lib.lumina_free_string(raw)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def open_collection(path, dim, metric=0, **kwargs):
    return Collection(path, dim, metric, **kwargs)
