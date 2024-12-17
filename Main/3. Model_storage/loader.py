import os
import dill

def _load_pyfunc(path):
    if os.path.isdir(path):
        path = os.path.join(path, "model.dill")

    with open(path, "rb") as f:
        return dill.load(f)
