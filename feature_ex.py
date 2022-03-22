import numpy as np
import torch

def feature_extraction(hgw1,hgw2):
    hgw1_temp = hgw1.reshape(161 , -1) #161*8960
    hgw1_fc = hgw1_temp.sum(1)
    # hgw1_fc = abs(hgw1_temp).sum(1)

    hgw1_temp2 = hgw1.reshape(161 , -1,128)
    w1mulw2 = hgw1_temp2.matmul(hgw2).reshape(161,-1)
    # w1mulw2_fc = w1mulw2.sum(1)
    w1mulw2_fc = abs(w1mulw2).sum(1)
    features =  hgw1_fc + w1mulw2_fc
    feature = (hgw1_fc + w1mulw2_fc).reshape(-1,1)

    feat = features/100
    B = np.argsort(feat)
    B = list(reversed(B))  # B中存储排序后的下标
    A = sorted(feat , reverse=True)  # A中存储排序后的结果
    AA = torch.tensor(A)
    BB = torch.tensor(B)
    return AA,BB