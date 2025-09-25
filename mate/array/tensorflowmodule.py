import os

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    TF_DTYPES = {
        'int16': tf.int16,
        'int32': tf.int32,
        'int64': tf.int64,
        'float16': tf.float16,
        'float32': tf.float32,
        'float64': tf.float64,
        "numpy.int16": tf.int16,
        "numpy.int32": tf.int32,
        "numpy.float16": tf.float16,
        "numpy.float32": tf.float32,
        "numpy.float64": tf.float64,
        np.int16: tf.int16,
        np.int32: tf.int32,
        np.float16: tf.float16,
        np.float32: tf.float32,
        np.float64: tf.float64,
        'torch.int16': tf.int16,
        'torch.int32': tf.int32,
        'torch.int64': tf.int64,
        'torch.float16': tf.float16,
        'torch.float32': tf.float32,
        'torch.float64': tf.float64,
        'complex64': tf.complex64,
        'complex128': tf.complex128,
        'torch.complex64': tf.complex64,
        'torch.complex128': tf.complex128,
    }
except (ModuleNotFoundError, ImportError) as err:
    pass

from mate.array.numpymodule import NumpyModule


class TFModule(NumpyModule):
    def __init__(self, backend=None, device_id=None):
        super().__init__(backend, device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(devices[0], True)

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def array(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.constant(*args, **kwargs)

    def take(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.gather(*args, **kwargs)

    def take_along_axis(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.experimental.numpy.take_along_axis(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.repeat(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.concat(*args, **kwargs)

    def stack(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.stack(*args, **kwargs)

    def unique(self, array, return_counts=False, axis=None):
        with tf.device('/GPU:0'):
            if axis is None:
                if return_counts == False:
                    y, idx = tf.unique(array)
                    return y
                else:
                    y, idx, count = tf.unique_with_counts(array)
                    return y, count
            else:
                if len(array.shape) != 2:
                    raise ValueError("Input array must be 2D")
                idx = self.lexsort(tf.transpose(array)[::-1])
                sortarr = tf.gather(array, idx)
                mask = tf.Variable(tf.zeros(array.shape[0], dtype=bool))
                mask[0].assign(True)
                mask[1:].assign(tf.math.reduce_any(sortarr[1:] != sortarr[:-1], axis=1))

                ret = sortarr[tf.constant(mask)]

                if not return_counts:
                    return ret

                ret = ret,
                if return_counts:
                    nonzero = tf.experimental.numpy.nonzero(mask)[0]
                    idx = tf.Variable(tf.zeros((tf.size(nonzero) + 1,), nonzero.dtype))
                    idx[:-1].assign(nonzero)
                    idx[-1].assign(tf.size(mask, out_type=idx.dtype))
                    ret += idx[1:] - idx[:-1],

                return ret

    def lexsort(self, keys, axis=-1):
        with tf.device('/GPU:0'):
            if tf.rank(keys) < 2:
                raise ValueError(f"keys must be at least 2 dimensional, but {tf.rank(keys)=}.")
            if len(keys) == 0:
                raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

            idx = tf.argsort(keys[0], axis=axis)
            for k in keys[1:]:
                idx = tf.gather(idx, tf.argsort(tf.gather(k, idx, axis=axis), axis=axis), axis=axis)

            return idx

    def arange(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.range(*args, **kwargs)

    def multiply(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.multiply(*args, **kwargs)

    def subtract(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.subtract(*args, **kwargs)

    def divide(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.divide(*args, **kwargs)

    def log2(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            numerator = tf.math.log(*args, **kwargs)
            denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
            return numerator / denominator

    def bincount(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.bincount(*args, **kwargs)

    def asnumpy(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return args[0].numpy()

    def argsort(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.argsort(*args, **kwargs)

    def astype(self, x, dtype):
        with tf.device('/GPU:0'):
            return tf.cast(x, dtype=TF_DTYPES[dtype])

    def tile(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.experimental.numpy.tile(*args, **kwargs)

    def where(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.where(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            val_dims = kwargs.pop('axes')
            return tf.transpose(*args, **kwargs, perm=val_dims)

    def reshape(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.reshape(*args, **kwargs)

    def greater(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.greater(*args, **kwargs)

    def greater_equal(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.greater_equal(*args, **kwargs)

    def less(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.less(*args, **kwargs)

    def less_equal(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.less_equal(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.logical_and(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.broadcast_to(*args, **kwargs)

    def minimum(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.minimum(*args, **kwargs)

    def max(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.reduce_max(*args, **kwargs)

    def min(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.reduce_min(*args, **kwargs)

    def exp(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.exp(*args, **kwargs)

    def dot(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.linalg.matmul(*args, **kwargs)

    def seed(self, seed):
        tf.random.set_seed(seed)

    def random_uniform(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            size = kwargs.pop('size')
            low = kwargs.pop('low')
            high = kwargs.pop('high')

            return tf.random.uniform(shape=size, minval=low, maxval=high)

    def linalg_solve(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.linalg.solve(*args, **kwargs)

    def pinv(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.linalg.pinv(*args, **kwargs)

    def sum(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.reduce_sum(*args, **kwargs)

    def diag(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.linalg.diag(*args, **kwargs)

    def nonzero(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.experimental.numpy.nonzero(*args, **kwargs)

    def eig(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.linalg.eig(*args, **kwargs)

    def inv(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.linalg.inv(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.linspace(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.zeros(*args, **kwargs)

    def real(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.math.real(*args, **kwargs)

    def matmul(self, *args, **kwargs):
        with tf.device('/GPU:0'):
            return tf.matmul(*args, *kwargs)