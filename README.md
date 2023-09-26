# :mate:MATE
- **MATE** represents **M**anycore-processor-**A**ccelerated **T**ransfer **E**ntropy computation.


## Installation
- :snake: [Anaconda](https://www.anaconda.com) is recommended to use and develop MATE.
- :penguin: Linux distros are tested and recommended to use and develop MATE.


MATE requires following backend-specific dependencies to be installed:


- CuPy: [Installing CuPy from Conda-Forge with cudatoolkit](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)

Install Cupy from Conda-Forge with cudatoolkit supported by your driver
```angular2html
conda install -c conda-forge cupy cuda-version=xx.x (check your CUDA version)
```
<br>

- JAX: [Installing JAX refer to the installation guide in the project README](https://github.com/google/jax#installation)

**You need to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [CuDNN](https://developer.nvidia.com/cudnn) before install JAX**

After install CUDA and CuDNN you can specify a particular CUDA and CuDNN version for jax explicitly
```angular2html
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.x and cudnn >= 8.6
pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

JAX preallocate 90% of the totla GPU memory when the first JAX operation is run \
Use 'XLA_PYTHON_CLIENT_PREALLOCATE=false' to disables the preallocation behavior\
(https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)

<br>

- :zap: PyTorch Lightning: [Installing PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html#install-pytorch-lightning)

MATE supports multi-GPU and multi-TPU Transfer Entropy calculations via PyTorch Lightning.<br>

If you're using PyTorch Lightning, use the [MATELightning class](#MATELightning-class). For more information, see MATELightning class below

### Install from GitHub repository
First, clone the recent version of this repository.

```
git clone https://github.com/cxinsys/mate
```


Now, we need to install MATE as a module.

```
cd mate
pip install -e .
```

## Tutorial

### MATE class
#### Create MATE instance

```angular2html
import mate

worker = mate.MATE()
```


#### Run MATE

#### parameters
- **arr**: numpy array for transfer entropy calculation, required
- **pair**: numpy array for calculation pairs, optional, default: compute possible pairs from all nodes in the arr
- **device**: optional, default: 'cpu'
- **device_ids**: optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **batch_size**: required
- **kp**: kernel percentile, optional, default: 0.5
- **percentile**: data crop percentile, optional, default: 0
- **win_length**: smoothe func window length parameter, optional, default: 10
- **polyorder**: smoothe func polyorder parameter, optional, default: 3

```angular2html
result_matrix = worker.run(arr=arr,
                           pairs=pairs,
                           device=device,
                           device_ids=device_ids,
                           batch_size=batch_size,
                           kp=kp,
                           percentile=percentile,
                           win_length=win_length,
                           polyorder=polyorder)
```

### MATELightning class
#### Create MATELightning instance

#### parameters
- **arr**: numpy array for transfer entropy calculation, required
- **pair**: numpy array for calculation pairs, optional, default: compute possible pairs from all nodes in the arr
- **kp**: kernel percentile, optional, default: 0.5
- **percentile**: data crop percentile, optional, default: 0
- **win_length**: smoothe func window length parameter, optional, default: 10
- **polyorder**: smoothe func polyorder parameter, optional, default: 3
- **len_time**: total length of expression array, optional, default: column length of array
- **dt**: history length of expression array, optional, default: 1

```angular2html
import mate

worker = mate.MATELightning(arr=arr,
                            pairs=pairs,
                            kp=kp,
                            percentile=percentile,
                            win_length=win_length,
                            polyorder=polyorder,
                            len_time=len_time,
                            dt=dt)
```

### Run MATELightning
#### parameters

- **device**: required, 'gpu', 'cpu' or 'tpu' etc
- **devices**: required, int or [list of device id]
- **batch_size**: required
- **num_workers**: optional, default: 0
```angular2html
result_matrix = worker.run(device=device,
                           devices=devices,
                           batch_size=batch_size,
                           num_worker=num_worker)
```

## TODO

- [x] add 'jax' backend module
- [x] implement 'pytorch lightning' backend module