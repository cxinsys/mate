import sys

try:
    import cupy
except (ModuleNotFoundError, ImportError) as err:
    pass

try:
    import torch
except (ModuleNotFoundError, ImportError) as err:
    pass

def get_device_list():
    if 'cupy' in sys.modules:
        n_gpus = cupy.cuda.runtime.getDeviceCount()
    elif 'torch' in sys.modules:
        n_gpus = torch.cuda.device_count()
    else:
        raise ImportError("GPU module (CuPy or PyTorch) not found. Please install it before proceeding.")

    return [i for i in range(n_gpus)]