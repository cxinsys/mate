# MATE
- **MATE** represents **M**anycore computing for **A**cceleration of **T**ensor **E**xecution.


## Installation
- :snake: [Anaconda](https://www.anaconda.com) is recommended to use and develop MATE.
- :penguin: Linux distros are tested and recommended to use and develop MATE.

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
<br>

- Default backend framework of the 'MATE' class is PyTorch.
- **[recommended]** To use PyTorch Lightning framework, you need to use a another class called 'MATELightning' (see [MATELightning class](#MATELightning-class))

<br>

### Install optional frameworks

MATE supports several optional backend frameworks such as CuPy and JAX. \
To use optional frameworks, you need to install the framework manually

<br>

- CuPy: [Installing CuPy from Conda-Forge with cudatoolkit](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)

Install Cupy from Conda-Forge with cudatoolkit supported by your driver
```angular2html
conda install -c conda-forge cupy cuda-version=xx.x (check your CUDA version)
```
<br>

- JAX: [Installing JAX refer to the installation guide in the project README](https://github.com/google/jax#installation)

[//]: # (**You must first install [CUDA]&#40;https://developer.nvidia.com/cuda-downloads&#41; and [CuDNN]&#40;https://developer.nvidia.com/cudnn&#41; before installing JAX**)

[//]: # ()
[//]: # (After install CUDA and CuDNN you can specify a particular CUDA and CuDNN version for jax explicitly)
Install JAX with CUDA > 12.x
```angular2html
pip install -U "jax[cuda12]"
```

<br>

- TensorFlow: [Installing TensorFlow refer to the installation guide](https://www.tensorflow.org/install/pip?hl=en#linux)

[//]: # (**You must first install [CUDA]&#40;https://developer.nvidia.com/cuda-downloads&#41; and [CuDNN]&#40;https://developer.nvidia.com/cudnn&#41; before installing JAX**)

[//]: # ()
[//]: # (After install CUDA and CuDNN you can specify a particular CUDA and CuDNN version for jax explicitly)
Install TensorFlow-GPU with CUDA
```angular2html
python3 -m pip install tensorflow[and-cuda]
```

<br>


## Tutorial

### MATE class
#### Create MATE instance

```angular2html
import mate

worker = mate.MATE()
```


#### Run MATE

#### parameters

[//]: # (MATE goes through a binning process, which is sensitive to noise. )

[//]: # (To work around this, you can use a smooth function like )

[//]: # (scipy's [savgol_filter]&#40;https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy-signal-savgol-filter&#41;.)

- **arr**: numpy array for transfer entropy calculation, required
- **pair**: numpy array for calculation pairs, optional, default: compute possible pairs from all nodes in the arr
- **backend**: optional, default: 'cpu'
- **device_ids**: optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **procs_per_device**: The number of processes to create per device when using non 'cpu' devices, optional, default: 1
- **batch_size**: required
- **kp**: kernel percentile, optional, default: 0.5
- **df**: history length, optional, default: 1


```angular2html
result_matrix = worker.run(arr=arr,
                           pairs=pairs,
                           backend=backend,
                           device_ids=device_ids,
                           procs_per_device=procs_per_device,
                           batch_size=batch_size,
                           kp=kp,
                           dt=dt,
                           )
```

### MATELightning class
#### Create MATELightning instance

#### parameters


- **arr**: numpy array for transfer entropy calculation, required
- **pair**: numpy array for calculation pairs, optional, default: compute possible pairs from all nodes in the arr
- **kp**: kernel percentile, optional, default: 0.5
- **len_time**: total length of expression array, optional, default: column length of array
- **dt**: history length of expression array, optional, default: 1

```angular2html
import mate

worker = mate.MATELightning(arr=arr,
                            pairs=pairs,
                            kp=kp,
                            len_time=len_time,
                            dt=dt)
```
<br>

#### Run MATELightning
#### parameters

MATELightning's run function parameters take the same values as [PyTorch's DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
and [PyTorch Lightning's Trainer](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#trainer). 
For additional options for parameters, see those documents

- **backend**: required, 'gpu', 'cpu' or 'tpu' etc
- **devices**: required, int or [list of device id]
- **batch_size**: required
- **num_workers**: optional, default: 0
```angular2html
result_matrix = worker.run(backend=backend,
                           devices=devices,
                           batch_size=batch_size,
                           num_worker=num_worker)
```

<br>

## TODO

- [x] add 'jax' backend module
- [x] implement 'pytorch lightning' backend module
- [x] add 'tensorflow' backend module
