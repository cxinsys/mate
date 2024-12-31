try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as err:
    pass

from mate.array.numpymodule import NumpyModule

class CuPyModule(NumpyModule):
    def __init__(self, backend=None, device_id=None):
        super().__init__(backend, device_id)

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

    def take_along_axis(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.take_along_axis(*args, **kwargs)

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

    def minimum(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.minimum(*args, **kwargs)

    def min(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.amin(*args, **kwargs)

    def max(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.amax(*args, **kwargs)

    def exp(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.exp(*args, **kwargs)

    def dot(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.dot(*args, **kwargs)

    def seed(self, seed):
        cp.random.seed(seed)

    def random_uniform(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.random.uniform(*args, **kwargs)

    def linalg_solve(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.linalg.solve(*args, **kwargs)

    def pinv(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.linalg.pinv(*args, **kwargs)

    def sum(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.sum(*args, **kwargs)

    def diag(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.diag(*args, **kwargs)

    def nonzero(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.nonzero(*args, **kwargs)

    def eig(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.linalg.eig(*args, **kwargs)

    def inv(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.linalg.inv(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.linspace(*args, **kwargs)

    def real(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.real(*args, **kwargs)

    def matmul(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.matmul(*args, **kwargs)