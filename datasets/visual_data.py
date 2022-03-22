from datasets import load_ft
from utils import hypergraph_utils as hgut
import numpy as np

def load_feature_construct_H(data_dir,
                             m_prob=1,
                             K_neigs=None,
                             is_probH=True,
                             split_diff_scale=False):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :return:
    """
    # init feature
    ft, lbls, idx_train, idx_test = load_ft(data_dir)
    # construct feature matrix
    fts = None
    fts = hgut.feature_concat(fts, ft)

    if fts is None:
        raise Exception(f'None feature used for model!')

    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    tmp = hgut.construct_H_with_KNN(ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
    H = hgut.hyperedge_concat(H, tmp)
    H[np.where(H!=0)] = 1
    print("fts:", fts.shape)
    print("lbls:", lbls.shape)
    print("H:", H.shape)


    return fts, lbls, idx_train, idx_test, H
