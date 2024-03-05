import os
import os.path as osp

import numpy as np

import mate
from scipy.signal import savgol_filter



def load_exp_data(dpath):
    fpath_binary = dpath[:-4]+'.npy'
    if osp.isfile(fpath_binary):
        print('Binary file already exist, load data from binary file')
        exp_data = np.load(fpath_binary)
        node_name = np.load(fpath_binary[:-4]+'_node_name.npy')
    else:
        exp_data = np.loadtxt(dpath, delimiter=',', dtype=str)
        node_name = exp_data[0, 1:]
        exp_data = exp_data[1:, 1:].T.astype(np.float32)

    return node_name, exp_data

def load_time_data(dpath, dtype=np.float32):
    return np.loadtxt(dpath, dtype=dtype)

if __name__ == "__main__":
    fpath_data = "./" # osp.join(droot, f'nodes_{nodes}/edges_{edges}/times_{times}')

    dpath_exp_data = osp.join(fpath_data, f'synthetic_small.csv')
    dpath_trj_data = osp.join(fpath_data, f'pseudo_time.txt')
    dpath_branch_data = osp.join(fpath_data, f'cell_select.txt')

    node_name, exp_data = load_exp_data(dpath_exp_data)
    trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
    branch = load_time_data(dpath_branch_data, dtype=np.int32)

    selected_trj = trajectory[branch == 1]
    inds_sorted_trj = np.argsort(selected_trj)

    selected_exp_data = exp_data[:, branch == 1]
    refined_exp_data = selected_exp_data[:, inds_sorted_trj]

    # worker = mate.MATELightning(arr=refined_exp_data,
    #                             smooth_func=savgol_filter,
    #                             smooth_param={'window_length': 10,
    #                                           'polyorder': 3})
    # results = worker.run(device='gpu', devices=1, batch_size=256, num_workers=1)
    
    worker = mate.MATE()
    results = worker.run(arr=refined_exp_data,
                         device='gpu',
                         device_ids=1,
                         batch_size=2 ** 15,
                         num_kernels=1,
                         method='pushing',
                         kp=0.5
                         )
                         

    print(len(results))
            
    np.save('result_matrix.npy', results)
    
    