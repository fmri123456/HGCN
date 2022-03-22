from config import get_config
import scipy.io as scio
import numpy as np

def set_weight(H):
    cfg1 = get_config('config/config1.yaml')
    cfg2 = get_config('config/config2.yaml')
    data_dir1 = cfg1['modelnet40_ft']
    data_dir2 = cfg2['modelnet40_ft']
    data1 = scio.loadmat(data_dir1)
    data2 = scio.loadmat(data_dir2)
    fts1 = data1['feat']
    fts2 = data2['feat']
    fts = np.concatenate((fts1, fts2), axis=1) #shape:(470,161,70)
    fts = fts.reshape(470, -1)
    col = fts.shape[1]
    # 将关联矩阵倒置，此时行表示超边
    H_ = H.T
    # 找到倒置矩阵中值为1的位置
    A = np.where(H_==1)
    # 存放权重值
    W_ = []
    for i in range(470):
        # 找到行坐标为i的元素在行坐标中的位置，这个位置与元素在列坐标中位置一致。
        x = np.where(A[0]==i)
        # 计算超边关联的顶点个数
        count = np.size(x)
        v = np.zeros((count, col))
        pccs = np.zeros(sum(range(1, count)))
        # 根据上面的坐标找到列坐标对应位置的值，这个值就是超边对应的顶点号
        y = A[1][x]
        # 找到每条超边所关联的count个顶点
        for j in range(count):
            v[j] = fts[y[j]]
        # 计算count个顶点两两皮尔逊相关系数
        for m in range(count):
            for n in range(m+1, count):
                pccs[int(m*(count-1)-(m*(m+1))/2+n-1)] = np.min(np.corrcoef(v[m], v[n]))
        mean = np.mean(pccs)
        W_.append(mean)
    W = np.diag(W_)
    return W



