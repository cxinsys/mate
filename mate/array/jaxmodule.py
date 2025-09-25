import os
from datetime import datetime

import numpy as np

try:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    from jax import device_put
    import jax.numpy as jnp
    import jax.lax
except (ModuleNotFoundError, ImportError) as err:
    pass

from mate.array.numpymodule import NumpyModule

class JaxModule(NumpyModule):
    def __init__(self, backend=None, device_id=None):
        super().__init__(backend, device_id)
        self.key = None

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        devices = jax.devices()
        self.selected_device = next((device for device in devices if device.id == self.device_id), None)

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def array(self, *args, **kwargs):
        # return device_put(jnp.array(*args, **kwargs), self.selected_device)
        with jax.default_device(self.selected_device):
            return jnp.array(*args, **kwargs)

    def take(self, *args, **kwargs):
        return jnp.take(*args, **kwargs)

    def take_along_axis(self, *args, **kwargs):
        return jnp.take_along_axis(*args, **kwargs)

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
        # devices = jax.devices()
        # selected_device = next((device for device in devices if device.id == self.device_id), None)
        # return device_put(jnp.arange(*args, **kwargs), self.selected_device)
        with jax.default_device(self.selected_device):
            return jnp.arange(*args, **kwargs)

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

    def minimum(self, *args, **kwargs):
        return jnp.minimum(*args, **kwargs)

    def max(self, *args, **kwargs):
        return jnp.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        return jnp.min(*args, **kwargs)

    def exp(self, *args, **kwargs):
        return jnp.exp(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return jnp.dot(*args, **kwargs)

    def seed(self, seed):
        self.key = jax.random.PRNGKey(seed)

    def random_uniform(self, *args, **kwargs):
        size = kwargs.pop('size')
        minval = kwargs.pop('low')
        maxval = kwargs.pop('high')

        self.seed(int(datetime.now().timestamp()))

        # devices = jax.devices()
        # selected_device = next((device for device in devices if device.id == self.device_id), None)
        # return device_put(jax.random.uniform(key=self.key, shape=size, minval=minval, maxval=maxval),
        #                   self.selected_device)
        with jax.default_device(self.selected_device):
            return jax.random.uniform(key=self.key, shape=size, minval=minval, maxval=maxval)

    def linalg_solve(self, *args, **kwargs):
        return jax.scipy.linalg.solve(*args, **kwargs)

    def pinv(self, *args, **kwargs):
        return jnp.linalg.pinv(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return jnp.sum(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return jnp.diag(*args, **kwargs)

    def nonzero(self, *args, **kwargs):
        # devices = jax.devices()
        # selected_device = next((device for device in devices if device.id == self.device_id), None)
        # return device_put(jnp.nonzero(*args, **kwargs), self.selected_device)
        with jax.default_device(self.selected_device):
            return jnp.nonzero(*args, **kwargs)

    def eig(self, *args, **kwargs):
        return jnp.linalg.eig(*args, **kwargs)

    def inv(self, *args, **kwargs):
        return jnp.linalg.inv(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        # devices = jax.devices()
        # selected_device = next((device for device in devices if device.id == self.device_id), None)
        # return device_put(jnp.linspace(*args, **kwargs), self.selected_device)
        with jax.default_device(self.selected_device):
                return jnp.linspace(*args, **kwargs)

    def real(self, *args, **kwargs):
        return jnp.real(*args, **kwargs)

    def matmul(self, *args, **kwargs):
        return jnp.matmul(*args, **kwargs)