import numpy as np

class ArrayModule:
    def __init__(self, backend, device_id):
        self._device = backend
        self._device_id = device_id

    def __enter__(self):
        return

    def __exit__(self, *args, **kwargs):
        return

    @property
    def backend(self):
        return self._device

    @property
    def device_id(self):
        return self._device_id

class NumpyModule(ArrayModule):
    def __init__(self, backend=None, device_id=None):
        super().__init__(backend, device_id)

    def array(self, *args, **kwargs):
        return np.array(*args, **kwargs)

    def take(self, *args, **kwargs):
        return np.take(*args, **kwargs)

    def take_along_axis(self, *args, **kwargs):
        return np.take_along_axis(*args, **kwargs)

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

    def minimum(self, *args, **kwargs):
        return np.minimum(*args, **kwargs)

    def min(self, *args, **kwargs):
        return np.min(*args, **kwargs)

    def max(self, *args, **kwargs):
        return np.max(*args, **kwargs)

    def exp(self, *args, **kwargs):
        return np.exp(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return np.dot(*args, **kwargs)

    def seed(self, seed):
        np.random.seed(seed)

    def random_uniform(self, *args, **kwargs):
        return np.random.uniform(*args, **kwargs)

    def linalg_solve(self, *args, **kwargs):
        return np.linalg.solve(*args, **kwargs)

    def pinv(self, *args, **kwargs):
        return np.linalg.pinv(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return np.sum(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return np.diag(*args, **kwargs)

    def nonzero(self, *args, **kwargs):
        return np.nonzero(*args, **kwargs)

    def eig(self, *args, **kwargs):
        return np.linalg.eig(*args, **kwargs)

    def inv(self, *args, **kwargs):
        return np.linalg.inv(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return np.linspace(*args, **kwargs)

    def real(self, *args, **kwargs):
        return np.real(*args, **kwargs)

    def matmul(self, *args, **kwargs):
        return np.matmul(*args, **kwargs)