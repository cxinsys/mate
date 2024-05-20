import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as err:
    pass

try:
    import jax
    from jax import device_put
    import jax.numpy as jnp
except (ModuleNotFoundError, ImportError) as err:
    pass

try:
    import torch
except (ModuleNotFoundError, ImportError) as err:
    pass

TORCH_DTYPES = {
    'int16' : torch.int16,
    'int32' : torch.int32,
    'float16' : torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    np.int16 : torch.int16,
    np.int32 : torch.int32,
    np.float16 : torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64
}

def parse_device(device):
    if device is None:
        return "cpu", 0

    device = device.lower()
    _device = device
    _device_id = 0

    if ":" in device:
        _device, _device_id = device.split(":")
        _device_id = int(_device_id)

    if _device not in ["cpu", "gpu", "cuda", "cupy", "jax", "torch"]:
        raise ValueError("device should be one of 'cpu', " \
                         "'gpu', or 'cuda', 'cupy', 'jax' and 'torch' not %s" % (device))

    return _device, _device_id


def get_array_module(device):
    _device, _device_id = parse_device(device)

    if "gpu" in _device or "cuda" in _device or "torch" in _device:
        return TorchModule(_device, _device_id)
    elif "jax" in _device:
        return JaxModule(_device, _device_id)
    elif "cupy" in _device:
        return CuPyModule(_device, _device_id)
    else:
        return NumpyModule(_device, _device_id)


class ArrayModule:
    def __init__(self, device, device_id):
        self._device = device
        self._device_id = device_id

    def __enter__(self):
        return

    def __exit__(self, *args, **kwargs):
        return

    @property
    def device(self):
        return self._device

    @property
    def device_id(self):
        return self._device_id


class NumpyModule(ArrayModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

    def array(self, *args, **kwargs):
        return np.array(*args, **kwargs)

    def take(self, *args, **kwargs):
        return np.take(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return np.repeat(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return np.concatenate(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return np.stack(*args, **kwargs)

    def unique(self, *args, **kwargs):
        return np.unique(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs)

    def lexsort(self, *args, **kwargs):
        return np.lexsort(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)

    def subtract(self, *args, **kwargs):
        return np.subtract(*args, **kwargs)

    def multiply(self, *args, **kwargs):
        return np.multiply(*args, **kwargs)

    def divide(self, *args, **kwargs):
        return np.divide(*args, **kwargs)

    def log2(self, *args, **kwargs):
        return np.log2(*args, **kwargs)

    def bincount(self, *args, **kwargs):
        return np.bincount(*args, **kwargs)

    def asnumpy(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        return np.argsort(*args, **kwargs)

    def astype(self, x, *args, **kwargs):
        return x.astype(*args, **kwargs)

    def tile(self, *args, **kwargs):
        return np.tile(*args, **kwargs)

    def where(self, *args, **kwargs):
        return np.where(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        return np.transpose(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return np.reshape(*args, **kwargs)

    def greater(self, *args, **kwargs):
        return np.greater(*args, **kwargs)

    def greater_equal(self, *args, **kwargs):
        return np.greater_equal(*args, **kwargs)

    def less(self, *args, **kwargs):
        return np.less(*args, **kwargs)

    def less_equal(self, *args, **kwargs):
        return np.less_equal(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        return np.logical_and(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        return np.broadcast_to(*args, **kwargs)


class CuPyModule(NumpyModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

        self._device = cp.cuda.Device()
        self._device.id = self._device_id
        self._device.use()

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def array(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.array(*args, **kwargs)

    def take(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.take(*args, **kwargs)

    def repeat(self, array, repeats):
        with cp.cuda.Device(self.device_id):
            repeats = cp.asnumpy(repeats).tolist()
            return cp.repeat(array, repeats)

    def concatenate(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.concatenate(*args, **kwargs)

    def stack(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.stack(*args, **kwargs)

    def unique(self, array, return_counts=False, axis=None):
        with cp.cuda.Device(self.device_id):
            if axis is None:
                return cp.unique(array, return_counts=return_counts)
            else:
                if len(array.shape) != 2:
                    raise ValueError("Input array must be 2D")
                sortarr = array[cp.lexsort(array.T[::-1])]
                mask = cp.empty(array.shape[0], dtype=cp.bool_)
                mask[0] = True
                mask[1:] = cp.any(sortarr[1:] != sortarr[:-1], axis=1)

                ret = sortarr[mask]

                if not return_counts:
                    return ret

                ret = ret,
                if return_counts:
                    nonzero = cp.nonzero(mask)[0]
                    idx = cp.empty((nonzero.size + 1,), nonzero.dtype)
                    idx[:-1] = nonzero
                    idx[-1] = mask.size
                    ret += idx[1:] - idx[:-1],

                return ret

    def zeros(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.zeros(*args, **kwargs)

    def lexsort(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.lexsort(*args, **kwargs)

    def arange(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.arange(*args, **kwargs)

    def multiply(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.multiply(*args, **kwargs)

    def subtract(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.subtract(*args, **kwargs)

    def divide(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.divide(*args, **kwargs)

    def log2(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.log2(*args, **kwargs)

    def bincount(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.bincount(*args, **kwargs)

    def asnumpy(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.asnumpy(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.argsort(*args, **kwargs)

    def astype(self, x, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return x.astype(*args, **kwargs)

    def tile(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.tile(*args, **kwargs)

    def where(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.where(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.transpose(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.reshape(*args, **kwargs)
    def greater(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.greater(*args, **kwargs)

    def greater_equal(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.greater_equal(*args, **kwargs)

    def less(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.less(*args, **kwargs)

    def less_equal(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.less_equal(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.logical_and(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.broadcast_to(*args, **kwargs)


class JaxModule(NumpyModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def array(self, *args, **kwargs):
        return device_put(jnp.array(*args, **kwargs), jax.devices()[self.device_id])

    def take(self, *args, **kwargs):
        return jnp.take(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return jnp.repeat(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return jnp.concatenate(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return jnp.stack(*args, **kwargs)

    def unique(self, *args, **kwargs):
        return jnp.unique(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return jnp.zeros(*args, **kwargs)

    def lexsort(self, *args, **kwargs):
        return jnp.lexsort(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return device_put(jnp.arange(*args, **kwargs), jax.devices()[self.device_id])

    def multiply(self, *args, **kwargs):
        return jnp.multiply(*args, **kwargs)

    def subtract(self, *args, **kwargs):
        return jnp.subtract(*args, **kwargs)

    def divide(self, *args, **kwargs):
        return jnp.divide(*args, **kwargs)

    def log2(self, *args, **kwargs):
        return jnp.log2(*args, **kwargs)

    def bincount(self, *args, **kwargs):
        return jnp.bincount(*args, **kwargs)

    def asnumpy(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        return jnp.argsort(*args, **kwargs)

    def astype(self, x, *args, **kwargs):
        return x.astype(*args, **kwargs)

    def tile(self, *args, **kwargs):
        return jnp.tile(*args, **kwargs)

    def where(self, *args, **kwargs):
        return jnp.where(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        return jnp.transpose(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return jnp.reshape(*args, **kwargs)

    def greater(self, *args, **kwargs):
        return jnp.greater(*args, **kwargs)

    def greater_equal(self, *args, **kwargs):
        return jnp.greater_equal(*args, **kwargs)

    def less(self, *args, **kwargs):
        return jnp.less(*args, **kwargs)

    def less_equal(self, *args, **kwargs):
        return jnp.less_equal(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        return jnp.logical_and(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        return jnp.broadcast_to(*args, **kwargs)

class TorchModule(NumpyModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def array(self, *args, **kwargs):
        if len(args) == 2:
            return torch.tensor(args[0], dtype=TORCH_DTYPES[str(args[1])], device='cuda:' + str(self.device_id))
        else:
            return torch.tensor(args[0], device='cuda:' + str(self.device_id))
    def take(self, *args, **kwargs):
        if len(args)+len(kwargs) == 3:
            return torch.index_select(args[0], kwargs['axis'], args[1])
        else:
            return torch.take(args[0], args[1])

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

    # def zeros(self, *args, **kwargs):
    #     return jnp.zeros(*args, **kwargs)

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