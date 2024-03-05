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
    # droot = '/home/rbsung/fasttenet/synex/data/iter_0'
    droot = r'D:\repos\fasttenet\synex\data\iter_0'
    # nodes = 32768
    # edges = 98304
    # times = 1024

    nodes = 128
    edges = 128
    times = 128

    fpath_data = osp.join(droot, f'nodes_{nodes}/edges_{edges}/times_{times}')

    print(f"Data file path: {fpath_data}")
    dpath_exp_data = osp.join(fpath_data, f'synthetic_expression_{nodes}_{edges}_{times}.csv')
    dpath_trj_data = osp.join(fpath_data, f'pseudo_time_{times}.txt')
    dpath_branch_data = osp.join(fpath_data, f'cell_sellect_{times}.txt')

    node_name, exp_data = load_exp_data(dpath_exp_data)
    trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
    branch = load_time_data(dpath_branch_data, dtype=np.int32)

    selected_trj = trajectory[branch == 1]
    inds_sorted_trj = np.argsort(selected_trj)

    selected_exp_data = exp_data[:, branch == 1]
    refined_exp_data = selected_exp_data[:, inds_sorted_trj]


    worker = mate.MATELightning(arr=refined_exp_data,
                                smooth_func=savgol_filter,
                                smooth_param={'window_length': 10,
                                              'polyorder': 3})
    results = worker.run(device='gpu', devices=1, batch_size=256, num_workers=1)
    
    # worker = mate.MATE()
    # results = worker.run(arr=refined_exp_data,
    #                      device='gpu',
    #                      device_ids=[0],
    #                      batch_size=128,
    #                      smooth_func=savgol_filter,
    #                      smooth_param={'window_length': 10,
    #                                    'polyorder': 3})
                         

    print(len(results))
            
    np.save('./result_matrix.npy', results)
    
    