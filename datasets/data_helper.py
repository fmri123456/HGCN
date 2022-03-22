import scipy.io as scio
import numpy as np


def load_ft(data_dir):
    # data = scio.loadmat(data_dir)
    # lbls = np.concatenate((np.zeros(233), np.ones(237))).astype(np.longlong)
    # idx = data['indices'].item()
    # x = data['feat'].shape[1]
    # y = data['feat'].shape[2]
    # data1 = data['feat'].reshape(lbls.shape[0], x*y)
    # fts = data1.astype(np.float32)
    #
    # idx_train = np.where(idx == 0)[0]
    # idx_test = np.where(idx == 1)[0]
    data = scio.loadmat(data_dir)
    lbls = np.concatenate((np.zeros(233), np.ones(237))).astype(np.longlong)

    shuffle_idx = np.array(range(0 , lbls.shape[0]))
    np.random.shuffle(shuffle_idx)
    # idx = data['indices'].item()
    data1 = data['feat']
    data1=data1.reshape(lbls.shape[0] , -1)
    fts = data1.astype(np.float32)
    n = int(lbls.shape[0]*0.8)
    idx_train = shuffle_idx[0:n]
    idx_test =  shuffle_idx[n:lbls.shape[0]]
    # idx_train = np.where(idx == 0)[0]
    # idx_test = np.where(idx == 1)[0]
    print("idx_train:", idx_train.shape)
    print("idx_test:", idx_test.shape)
    return fts, lbls, idx_train, idx_test

