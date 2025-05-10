""" Make `experiments` folder import-safe, i.e. compatible with `python -m experiments.run ...``` """
# for error code F401: module imported but unused
from importlib import import_module  # noqa: F401
