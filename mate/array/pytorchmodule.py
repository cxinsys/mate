import numpy as np

try:
    import torch

    TORCH_DTYPES = {
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        "numpy.int16": torch.int16,
        "numpy.int32": torch.int32,
        "numpy.float16": torch.float16,
        "numpy.float32": torch.float32,
        "numpy.float64": torch.float64,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        'torch.int16': torch.int16,
        'torch.int32': torch.int32,
        'torch.int64': torch.int64,
        'torch.float16': torch.float16,
        'torch.float32': torch.float32,
        'torch.float64': torch.float64,
        'complex64': torch.complex64,
        'complex128': torch.complex128,
        'torch.complex64': torch.complex64,
        'torch.complex128': torch.complex128,
    }
except (ModuleNotFoundError, ImportError) as err:
    pass

from mate.array.numpymodule import NumpyModule

class TorchModule(NumpyModule):
    def __init__(self, backend=None, device_id=None):
        super().__init__(backend, device_id)

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def array(self, *args, **kwargs):
        if len(args) == 2:
            return torch.tensor(args[0], dtype=TORCH_DTYPES[args[1]], device='cuda:' + str(self.device_id))
        elif len(args) == 1 and len(kwargs) == 1:
            dtype = kwargs.pop('dtype')
            return torch.tensor(args[0], dtype=TORCH_DTYPES[dtype], device='cuda:' + str(self.device_id))
        else:
            if type(args[0]) == list:
                dtype = str(np.array(args[0]).dtype)
            return torch.tensor(args[0], dtype=TORCH_DTYPES[dtype], device='cuda:' + str(self.device_id))

    def take(self, *args, **kwargs):
        if len(args)+len(kwargs) == 3:
            return torch.index_select(args[0], kwargs['axis'], args[1])
        else:
            return torch.take(args[0], args[1])

    def take_along_axis(self, *args, **kwargs):
        val_dim = kwargs.pop('axis')
        return torch.take_along_dim(args[0], args[1].to(torch.int64), dim=val_dim)

    def repeat(self, *args, **kwargs):
        return torch.repeat_interleave(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        val_dim = kwargs.pop('axis')
        return torch.concatenate(*args, **kwargs, dim=val_dim)

    def stack(self, *args, **kwargs):
        val_dim = kwargs.pop('axis')
        return torch.stack(*args, **kwargs, dim=val_dim)

    def unique(self, *args, **kwargs):
        if len(kwargs) == 2:
            val_dim = kwargs.pop('axis')
            return torch.unique(*args, **kwargs, dim=val_dim)
        else:
            return torch.unique(*args, **kwargs)

    def lexsort(self, keys, dim=-1):
        if keys.ndim < 2:
            raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
        if len(keys) == 0:
            raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

        idx = keys[0].argsort(dim=dim, stable=True)
        for k in keys[1:]:
            idx = idx.index_select(dim, k.index_select(dim, idx).argsort(dim=dim, stable=True))

        return idx

    def arange(self, *args, **kwargs):
        return torch.arange(*args, **kwargs, device='cuda:' + str(self.device_id))

    def multiply(self, *args, **kwargs):
        return torch.multiply(*args, **kwargs)

    def subtract(self, *args, **kwargs):
        return torch.subtract(*args, **kwargs)

    def divide(self, *args, **kwargs):
        return torch.divide(*args, **kwargs)

    def log2(self, *args, **kwargs):
        return torch.log2(*args, **kwargs)

    def bincount(self, *args, **kwargs):
        return torch.bincount(*args, **kwargs)

    def asnumpy(self, *args, **kwargs):
        return args[0].detach().cpu().numpy()

    def argsort(self, *args, **kwargs):
        return torch.argsort(*args, **kwargs)

    def astype(self, x, dtype):
        return x.to(TORCH_DTYPES[dtype])

    def tile(self, *args, **kwargs):
        return torch.tile(args[0], (args[1],))

    def where(self, *args, **kwargs):
        return torch.where(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        val_dim = kwargs.pop('axes')
        if not val_dim:
            return torch.t(*args, **kwargs)
        return torch.permute(*args, **kwargs, dims=val_dim)

    def reshape(self, *args, **kwargs):
        return torch.reshape(*args, **kwargs)

    def greater(self, *args, **kwargs):
        return torch.greater(*args, **kwargs)

    def greater_equal(self, *args, **kwargs):
        return torch.greater_equal(*args, **kwargs)

    def less(self, *args, **kwargs):
        return torch.less(*args, **kwargs)

    def less_equal(self, *args, **kwargs):
        return torch.less_equal(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        return torch.logical_and(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        return torch.broadcast_to(*args, **kwargs)

    def minimum(self, *args, **kwargs):
        return torch.minimum(*args, **kwargs)

    def max(self, *args, **kwargs):
        val_dim = kwargs.pop('axis')
        maxes, inds = torch.max(args[0], dim=val_dim)
        return maxes

    def min(self, *args, **kwargs):
        val_dim = kwargs.pop('axis')
        return torch.min(args[0], dim=val_dim)

    def exp(self, *args, **kwargs):
        return torch.exp(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)

    def seed(self, seed):
        torch.manual_seed(seed)

    def random_uniform(self, *args, **kwargs):
        size = kwargs.pop('size')
        low = kwargs.pop('low')
        high = kwargs.pop('high')

        return torch.empty(size, device='cuda:' + str(self.device_id)).uniform_(low, high)

    def linalg_solve(self, *args, **kwargs):
        return torch.linalg.solve(*args, **kwargs)

    def pinv(self, *args, **kwargs):
        return torch.linalg.pinv(*args, **kwargs)

    def sum(self, *args, **kwargs):
        val_dim = kwargs.pop('axis')
        return torch.sum(*args, dim=val_dim)

    def diag(self, *args, **kwargs):
        return torch.diag(*args, **kwargs)

    def nonzero(self, *args, **kwargs):
        return torch.nonzero(*args, **kwargs, as_tuple=True)

    def eig(self, *args, **kwargs):
        return torch.linalg.eig(*args, **kwargs)

    def inv(self, *args, **kwargs):
        return torch.linalg.inv(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return torch.linspace(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        if len(args) == 2:
            return torch.zeros(args[0], dtype=TORCH_DTYPES[args[1]], device='cuda:' + str(self.device_id))
        elif len(args) == 1 and len(kwargs) == 1:
            dtype = kwargs.pop('dtype')
            return torch.zeros(args[0], dtype=TORCH_DTYPES[dtype], device='cuda:' + str(self.device_id))
        else:
            if type(args[0]) == list:
                dtype = str(np.array(args[0]).dtype)
            return torch.zeros(args[0], dtype=TORCH_DTYPES[dtype], device='cuda:' + str(self.device_id))

    def real(self, *args, **kwargs):
        return torch.real(*args, **kwargs)

    def matmul(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)