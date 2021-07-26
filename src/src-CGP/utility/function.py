import os
import sys
import shutil
import numpy as np
import scipy.sparse as sp
import pickle
import copy
from operator import itemgetter
from sklearn import metrics
import torch
from typing import List, Union
import torch.utils.data


### mkdir
def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


### sparse_to_tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


### dropout_sparse
def dropout_sparse(x, dropout, noise_shape):
    i = x._indices()
    v = x._values()

    random_tensor = 1 - dropout
    random_tensor = random_tensor + torch.rand(noise_shape)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape)
    out = out * (1. / (1 - dropout))

    return out


### scipy_sparse_to_torch_sparse
def scipy_sparse_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    values = sparse_mx.data
    indices = np.vstack((sparse_mx.row, sparse_mx.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_mx.shape
    x = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return x


### fixed_unigram_candidate_sampler
### https://github.com/YuutoOhnuki/decagon-pytorch/blob/master/decagon-pytorch/src/decagon_pytorch/sampling.py
def fixed_unigram_candidate_sampler(
    true_classes: Union[np.array, torch.Tensor],
    num_samples: int,
    unigrams: List[Union[int, float]],
    distortion: float = 1.):

    if isinstance(true_classes, torch.Tensor):
        true_classes = true_classes.numpy()
    if true_classes.shape[0] != num_samples:
        raise ValueError('true_classes must be a 2D matrix with shape (num_samples, num_true)')
    unigrams = np.array(unigrams)

    if distortion != 1.:
        unigrams = unigrams.astype(np.float64) ** distortion
    indices = np.arange(num_samples)
    result = np.zeros(num_samples, dtype=np.int64)
    tmp = indices
    while len(tmp) > 0:
        sampler = torch.utils.data.WeightedRandomSampler(unigrams, len(indices))
        candidates = np.array(list(sampler))
        candidates = np.reshape(candidates, (len(indices), 1))
        result[indices] = candidates.T
        mask = (candidates == true_classes[indices, :])
        mask = mask.sum(1).astype(np.bool)
        tmp = indices[mask]
    return result


# ###  hinge_loss
# def hinge_loss(pos, neg, margin):
#     """Maximum-margin optimization using the hinge loss."""
#     diff = torch.relu(torch.sub(neg, pos - margin))
#     loss = torch.sum(diff)
#     return loss


###  test
def test(prefix, epo_idx, net, dp, args, log):

    net.eval()
    dp.is_train = False
    dp.dropout = 0.  # disable dropout for val

    edges = pickle.load(open(args.edg_pkl_path + "{}_edges.pkl".format(prefix), "rb"))
    false_edges = pickle.load(open(args.edg_pkl_path + "{}_false_edges.pkl".format(prefix), "rb"))

    auroc_list = []
    auprc_list = []
    apk_list = []
    auroc_avg_list = []
    auprc_avg_list = []
    apk_avg_list = []
    rel_type_tmp = []
    rel_type_list = []
    for rel_type, edge_lists in edges.items():
        rel_type_tmp.append(rel_type)  # [(0, 1)] ; [(0, 1), (1, 0)]
        rel_type_list.append(copy.deepcopy(rel_type_tmp))  # [[(0, 1)], [(0, 1), (1, 0)]]

        i, j = rel_type
        for rt_k, edge_list in enumerate(edge_lists):
            ### pos & neg edges
            pos_edge_list = edge_list
            neg_edge_list = false_edges[rel_type][rt_k]

            ### preds result
            pos_preds_all = []
            neg_preds_all = []

            ### test per batch or in one time
            if args.te_batch:
                """
                test the data per batch
                """
                ### edges number
                n_edges = len(pos_edge_list)
                n_batchs = n_edges // args.te_batch_size
                bt_idx = 0
                while True:
                    ### start, end
                    bt_s = bt_idx * args.te_batch_size
                    bt_e = bt_s + args.te_batch_size
                    if bt_s < n_edges and bt_e >= n_edges:
                        bt_e = n_edges
                    ### pos edges
                    pos_edges_array = np.array(pos_edge_list[bt_s:bt_e])
                    pos_edges_tensor = torch.from_numpy(pos_edges_array)
                    pos_edges_tensor = pos_edges_tensor.to(args.device)
                    pos_inputs = ((i, j), rt_k, pos_edges_tensor)
                    pos_preds, _ = net(pos_inputs)
                    pos_preds = torch.sigmoid(pos_preds)
                    pos_preds = pos_preds.detach().cpu().numpy().tolist()
                    pos_preds_all += pos_preds
                    ### neg edges
                    neg_edges_array = np.array(neg_edge_list[bt_s:bt_e])
                    neg_edges_tensor = torch.from_numpy(neg_edges_array)
                    neg_edges_tensor = neg_edges_tensor.to(args.device)
                    neg_inputs = ((i, j), rt_k, neg_edges_tensor)
                    neg_preds, _ = net(neg_inputs)
                    neg_preds = torch.sigmoid(neg_preds)
                    neg_preds = neg_preds.detach().cpu().numpy().tolist()
                    neg_preds_all += neg_preds
                    ### bt_idx ++
                    bt_idx += 1
                    ### break
                    if bt_idx > n_batchs:
                        break
            else:
                """
                test all data in one time
                """
                ### pos edges
                pos_edges_array = np.array(pos_edge_list)
                pos_edges_tensor = torch.from_numpy(pos_edges_array)
                pos_edges_tensor = pos_edges_tensor.to(args.device)
                pos_inputs = ((i, j), rt_k, pos_edges_tensor)
                pos_preds, _ = net(pos_inputs)
                pos_preds = torch.sigmoid(pos_preds)
                pos_preds_all = pos_preds.detach().cpu().numpy().tolist()
                ### neg edges
                neg_edges_array = np.array(neg_edge_list)
                neg_edges_tensor = torch.from_numpy(neg_edges_array)
                neg_edges_tensor = neg_edges_tensor.to(args.device)
                neg_inputs = ((i, j), rt_k, neg_edges_tensor)
                neg_preds, _ = net(neg_inputs)
                neg_preds = torch.sigmoid(neg_preds)
                neg_preds_all = neg_preds.detach().cpu().numpy().tolist()

            ### all
            preds_all = np.hstack([pos_preds_all, neg_preds_all])
            preds_all = np.nan_to_num(preds_all)
            labels_all = np.hstack([np.ones(len(pos_preds_all)), np.zeros(len(neg_preds_all))])

            actual = list(range(len(pos_preds_all)))
            predicted = []
            for idx, score in enumerate(preds_all):
                predicted.append((score, idx))
            predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

            ### roc, auprc, apk
            auroc = metrics.roc_auc_score(labels_all, preds_all)
            auprc = metrics.average_precision_score(labels_all, preds_all)
            apk = compute_apk(actual, predicted, k=args.ap_k)
            log.info("[{}] | Epoch: [{}/{}] | Type: [{},{},{}] |"
                     " AUROC: {:.5f} | AUPRC: {:.5F} | AP@{}: {:.5f}".format(
                prefix, epo_idx + 1, args.epoch, i, j, rt_k,
                auroc, auprc, args.ap_k, apk))

            ### append
            auroc_list.append(auroc)
            auprc_list.append(auprc)
            apk_list.append(apk)

        ### avg
        auroc_avg = np.mean(auroc_list)
        auprc_avg = np.mean(auprc_list)
        apk_avg = np.mean(apk_list)
        auroc_avg_list.append(auroc_avg)
        auprc_avg_list.append(auprc_avg)
        apk_avg_list.append(apk_avg)

    for rel_type, auroc_avg, auprc_avg, apk_avg in zip(rel_type_list, auroc_avg_list, auprc_avg_list, apk_avg_list):
        log.info("[{}] | Epoch: [{}/{}] | all >>> type: {} |"
                 " avg AUROC: {:.5f} | avg AUPRC: {:.5F} | avg AP@{}: {:.5f}".format(
            prefix, epo_idx + 1, args.epoch, rel_type,
            auroc_avg, auprc_avg, args.ap_k, apk_avg))

    return auroc_avg_list[0], auprc_avg_list[0], apk_avg_list[0]


### apk
def compute_apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


### mapk
def compute_mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average precision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([compute_apk(a,p,k) for a, p in zip(actual, predicted)])

