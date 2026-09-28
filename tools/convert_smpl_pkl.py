import argparse
import pickle
import sys

import numpy as np

KEYS = ["v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f"]


class _Stub:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})

    def __array__(self, dtype=None):
        return np.asarray(self.__dict__.get("x", self.__dict__.get("r")), dtype=dtype)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def to_array(v):
    if hasattr(v, "toarray"):
        v = v.toarray()
    return np.asarray(v)


def convert(src, dst):
    with open(src, "rb") as f:
        data = _Unpickler(f, encoding="latin1").load()
    out = {}
    for k in KEYS:
        arr = to_array(data[k])
        out[k] = arr.astype(np.int64) if k in ("kintree_table", "f") else arr.astype(np.float32)
    assert out["v_template"].shape == (6890, 3) and out["J_regressor"].shape == (24, 6890), {k: v.shape for k, v in out.items()}
    np.savez(dst, **out)
    print(f"{src} -> {dst}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="convert the official SMPL_NEUTRAL.pkl (chumpy objects) into a plain npz")
    p.add_argument("src")
    p.add_argument("dst")
    a = p.parse_args()
    convert(a.src, a.dst)
