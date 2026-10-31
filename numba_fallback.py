"""Fallback compatibility module for environments where Numba native DLLs are blocked.

Provides pure-Python passthrough decorators and types so Librosa can utilize
standard NumPy, SciPy, and SoundFile routines without requiring native Numba DLLs.
"""

import sys
import types
import numpy as np

try:
    import numba  # noqa: F401
except (ImportError, Exception):
    fake_numba = types.ModuleType("numba")

    def _passthrough_decorator(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def _wrapper(func):
            return func

        return _wrapper

    fake_numba.jit = _passthrough_decorator
    fake_numba.njit = _passthrough_decorator
    fake_numba.guvectorize = _passthrough_decorator
    fake_numba.vectorize = _passthrough_decorator
    fake_numba.stencil = _passthrough_decorator
    fake_numba.prange = range

    # Map standard scalar types
    fake_numba.boolean = np.bool_
    fake_numba.uint8 = np.uint8
    fake_numba.uint16 = np.uint16
    fake_numba.uint32 = np.uint32
    fake_numba.uint64 = np.uint64
    fake_numba.int8 = np.int8
    fake_numba.int16 = np.int16
    fake_numba.int32 = np.int32
    fake_numba.int64 = np.int64
    fake_numba.float32 = np.float32
    fake_numba.float64 = np.float64

    fake_types = types.ModuleType("numba.types")
    fake_numba.types = fake_types
    sys.modules["numba.types"] = fake_types
    sys.modules["numba"] = fake_numba
