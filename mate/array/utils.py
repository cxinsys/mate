import time
import os

from mate.array.numpymodule import NumpyModule
from mate.array.cupymodule import CuPyModule
from mate.array.jaxmodule import JaxModule
from mate.array.pytorchmodule import TorchModule
from mate.array.tensorflowmodule import TFModule

def parse_device(backend):
    if backend is None:
        return "cpu", 0

    backend = backend.lower()
    _device = backend
    _device_id = 0

    if ":" in backend:
        _device, _device_id = backend.split(":")
        _device_id = int(_device_id)

    if _device not in ["cpu", "gpu", "cuda", "cupy", "jax", "torch", "tensorflow", "tf"]:
        raise ValueError("backend should be one of 'cpu', 'gpu', 'cuda'," \
                         "'cupy', 'jax', 'torch', and 'tensorflow' not %s" % (backend))

    return _device, _device_id


def get_array_module(backend):
    _device, _device_id = parse_device(backend)

    if "gpu" in _device or "cuda" in _device or "torch" in _device:
        return TorchModule(_device, _device_id)
    elif "jax" in _device:
        return JaxModule(_device, _device_id)
    elif "cupy" in _device:
        return CuPyModule(_device, _device_id)
    elif "tensorflow" in _device or "tf" in _device:
        return TFModule(_device, _device_id)
    else:
        return NumpyModule(_device, _device_id)