import numpy as np
import os
import random

def getBatchTask(meta_dataset, batch_num=16):
    """
    Args:
        meta_dataset: meta_train or meta_test
        batch_num
    Returns:(type: torch.tensor)
        x_spt_tensor: shape(batch_num, spt_shot, seq_num, seq_len)
        y_spt_tensor: shape(batch_num, spt_shot, seq_num)
        x_qry_tensor: shape(batch_num, qry_shot, seq_num, seq_len)
        y_qry_tensor: shape(batch_num, qry_shot, seq_num)
    """
    ways = len(meta_dataset.keys())
    """
    这里构造batch的策略是：
    尽量从每个dataset中选取task来构成batch，如果无法整除，
    则会出现有的数据集多一个task的情况，这是合理的。
    """
    base_k = batch_num // ways
    add_k = batch_num % ways
    way_tasks_num = []
    for _ in range(ways):
        cur_tasks = base_k
        if add_k > 0:
            cur_tasks += 1
        add_k -= 1
        way_tasks_num.append(cur_tasks)
    
    x_spt_batch_list = []
    y_spt_batch_list = []
    x_qry_batch_list = []
    y_qry_batch_list = []
    
    for dataset_idx, key in enumerate(list(meta_dataset.keys())):
        x_spt_pool = meta_dataset[key]['x_spt']
        y_spt_pool = meta_dataset[key]['y_spt']
        x_qry_pool = meta_dataset[key]['x_qry']
        y_qry_pool = meta_dataset[key]['y_qry']

        sample_len = x_spt_pool.shape[0]
        sample_idx_list = random.sample(range(0, sample_len-1), way_tasks_num[dataset_idx])

        sample_x_spt = x_spt_pool[sample_idx_list]
        sample_y_spt = y_spt_pool[sample_idx_list]
        sample_x_qry = x_qry_pool[sample_idx_list]
        sample_y_qry = y_qry_pool[sample_idx_list]

        x_spt_batch_list.append(sample_x_spt)
        y_spt_batch_list.append(sample_y_spt)
        x_qry_batch_list.append(sample_x_qry)
        y_qry_batch_list.append(sample_y_qry)

    x_spt_batch = np.concatenate(x_spt_batch_list)
    y_spt_batch = np.concatenate(y_spt_batch_list)
    x_qry_batch = np.concatenate(x_qry_batch_list)
    y_qry_batch = np.concatenate(y_qry_batch_list)

    return x_spt_batch, y_spt_batch, x_qry_batch, y_qry_batch


def poolRead():
    # dataset name list
    dn_list = ['ai', 'hw', 'yahoo', 'aiops']
    x_spt_pool_f_list = [os.path.join('fsl_generator', 'fsl_pool', '{}_spt_x_pool.npy'.format(dn)) for dn in dn_list]
    y_spt_pool_f_list = [os.path.join('fsl_generator', 'fsl_pool', '{}_spt_y_pool.npy'.format(dn)) for dn in dn_list]
    x_qry_pool_f_list = [os.path.join('fsl_generator', 'fsl_pool', '{}_qry_x_pool.npy'.format(dn)) for dn in dn_list]
    y_qry_pool_f_list = [os.path.join('fsl_generator', 'fsl_pool', '{}_qry_y_pool.npy'.format(dn)) for dn in dn_list]

    x_spt_pool_list = [np.load(f) for f in x_spt_pool_f_list]
    y_spt_pool_list = [np.load(f) for f in y_spt_pool_f_list]
    x_qry_pool_list = [np.load(f) for f in x_qry_pool_f_list]
    y_qry_pool_list = [np.load(f) for f in y_qry_pool_f_list]

    meta_train = {
        "ai": {
            "x_spt": x_spt_pool_list[0],
            "y_spt": y_spt_pool_list[0], 
            "x_qry": x_qry_pool_list[0],
            "y_qry": y_qry_pool_list[0]
            }, 
        "hw": {
            "x_spt": x_spt_pool_list[1], 
            "y_spt": y_spt_pool_list[1], 
            "x_qry": x_qry_pool_list[1], 
            "y_qry": y_qry_pool_list[1]
            }, 
        "yahoo": {
            "x_spt": x_spt_pool_list[2], 
            "y_spt": y_spt_pool_list[2], 
            "x_qry": x_qry_pool_list[2], 
            "y_qry": y_qry_pool_list[2]
            }
    }

    meta_test = {
        "aiops": {
            "x_spt": x_spt_pool_list[3], 
            "y_spt": y_spt_pool_list[3], 
            "x_qry": x_qry_pool_list[3], 
            "y_qry": y_qry_pool_list[3]
            }
    }

    print("------meta_train------")
    for key in meta_train.keys():
        print("{}: x_spt_shape {}, y_spt_shape {}, x_qry_shape {}, y_qry_shape {}".format(
            key, meta_train[key]['x_spt'].shape, meta_train[key]['y_spt'].shape, 
            meta_train[key]['x_qry'].shape, meta_train[key]['y_qry'].shape
        ))
    
    print("------meta_test------")
    print("aiops: x_spt_shape {}, y_spt_shape {}, x_qry_shape {}, y_qry_shape {}".format(
        meta_test['aiops']['x_spt'].shape, meta_test['aiops']['y_spt'].shape,
        meta_test['aiops']['x_qry'].shape, meta_test['aiops']['y_qry'].shape
    ))

    return meta_train, meta_test


if __name__ == '__main__':
    meta_train, meta_test = poolRead()
    x_spt, y_spt, x_qry, y_qry = getBatchTask(meta_train)
    print('x_spt.shape = {}'.format(x_spt.shape))
    print('y_spt.shape = {}'.format(y_spt.shape))
    print('x_qry.shape = {}'.format(x_qry.shape))
    print('y_qry.shape = {}'.format(y_qry.shape))