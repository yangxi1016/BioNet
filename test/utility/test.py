# import pickle
# import torch
#
# edges = pickle.load(open("../../data/edg-pkl/tr_edges.pkl", "rb"))
# edges = torch.from_numpy(edges[(0, 1)][0])
# print(edges)
# indices = torch.tensor([1])
# a = torch.index_select(edges, 1, indices)
# a = torch.squeeze(a)
# print(a)
#
#
# a = None
# #
# # print(a is None)
#
#
# a = list(range(10))
# print(a)
# print(a[1:10:2])
# print(range(10))

# import pickle
# from torch.utils.data import TensorDataset
# import torch
# import numpy as np
# tr_edges = pickle.load(open("../../data/edg-pkl/tr_edges.pkl", "rb"))
#
# rc_idx = 0  # relation category index
# rc_lens_list = []
# tr_edges_list = []
# tr_labels_list = []
# for rel_type, edge_lists in tr_edges.items():
#     for edge_list in edge_lists:
#         rc_lens_list.append(len(edge_list))
#         for edge in edge_list:
#             tr_edges_list.append(edge)
#             tr_labels_list.append(rc_idx)
#         rc_idx += 1
# tr_edges_array = np.array(tr_edges_list)
# tr_labels_array = np.array(tr_labels_list)
# tr_edges = TensorDataset(torch.from_numpy(tr_edges_array), torch.from_numpy(tr_labels_array))
# rc_lens = torch.Tensor(rc_lens_list)
#
# print(len(rc_lens))


# print(tr_edges[1, 0][0])
#
# print(rc_len_list)
# rt = 3
# start = sum(rc_len_list[:rt])
# end = start + rc_len_list[rt]
# print(tr_edges_list[start:end])
# print(len(tr_edges_list[start:end]))

# from collections import defaultdict
# a = defaultdict(list)
# b = [1,2,3,4]
# a[0] += b
# print(a)


# import torch
# l = [1, 1, 1, 1]
# l = torch.Tensor(l)
# l = l[0]
# print(l)


# import torch
# from torch import nn
#
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         print('before register:\n', self._parameters, end='\n\n')
#         self.register_parameter('my_param1', nn.Parameter(torch.randn(3, 3)))
#         # print('after register and before nn.Parameter:\n', self._parameters, end='\n\n')
#
#         self.my_param2 = nn.Parameter(torch.randn(2, 2))
#         # print('after register and nn.Parameter:\n', self._parameters, end='\n\n')
#
#
#         # k = 1
#         # self.register_parameter("weight_{}".format(k), nn.Parameter(torch.randn(3, 3)))
#         # self.name = "weight_{}".format(k)
#         # print(self."weight_{}".format(k))
#
#         # self.register_parameter("b", nn.Parameter(torch.randn(3, 3)))
#         # print(self._parameters)
#
#     def forward(self, x):
#         return x
#
# mymodel1 = MyModel()
# mymodel2 = MyModel()
#
# for k, v in mymodel.named_parameters():
# #     print(k, v)
#
# from torch import nn
# import torch
# w = nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
#
# print(w)


import numpy as np
from scipy.sparse import csr_matrix
import torch



# a = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
# print('a', a)
#
# # Acoo = Acsr.tocoo()
# # # print('Acoo',Acoo)
# #
# # Apt = torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
# #                               torch.FloatTensor(Acoo.data.astype(np.float)))
# #
# # print('Apt',Apt)
# print(a.tocoo().row)

# a = []
# a.append(torch.tensor([1., 1., 1.]))
# a.append(torch.tensor([2., 2., 2.]))
# # b = torch.randn(3)
# r = torch.stack(a)
# print(r)
# r = torch.sum(r, dim=0, keepdim=True)
# print(r)
# r = torch.squeeze(r)
# print(r)

#
# #在数据处理中，
# num_samples= 4
# t = [[2],[4],[7],[8]]
# indices = np.arange(num_samples)
# from  torch.utils.data.sampler import   WeightedRandomSampler
# wights=[2,2,1,1,1,1,2,2]
# sampler=WeightedRandomSampler(wights,num_samples=num_samples,replacement=True)
# candidates = np.array(list(sampler))
# candidates = np.reshape(candidates, (num_samples, 1))
# result = candidates.T
# mask = (candidates == t)
# print(mask)
# mask = mask.sum(1).astype(np.bool)
# print(mask)
#
# print(indices)
# indices = indices[mask]
# print(indices)
#
# # print(candidates)

#
# import torch
# a = torch.randn(3, 3)
# a = torch.diag(a)
# print(a)
#
#
# b = torch.randn(3, 3)
# b = torch.diag(b)
# print(b)
# print(b.unsqueeze(0))
# print(b-0.1)
#
# diff = torch.relu(torch.sub(a, b.unsqueeze(0) - 0.1))
# print(diff)
# loss = torch.sum(diff)
# print(loss)
#
# diff1 = torch.relu(torch.sub(a, b - 0.1))
# print(diff1)
# loss1 = torch.sum(diff1)
# print(loss1)
#
# r = (1, 2)
# k = 3
# b = "{}_b_{}".format(r, k)
# print(b)
#
# import torch.nn as nn
# from torch.nn import functional
#
# class Example(nn.Module):
#     def __init__(self):
#         super(Example, self).__init__()
#         # # print('看看我们的模型有哪些parameter:\t', self._parameters, end='\n')
#         # self.W1_params = nn.Parameter(torch.rand(2, 3))
#         # # print('增加W1后看看：', self._parameters, end='\n')
#         #
#         # self.register_parameter('W2_params', nn.Parameter(torch.rand(2, 3)))
#         # # print('增加W2后看看：', self._parameters, end='\n')
#
#         self.w = {}
#         for k in range(3):
#             self.w["weight_%d" % k] = nn.Parameter(torch.randn(16, 8))
#             self.register_parameter("weight_{}".format(k), self.w["weight_%d" % k])
#
#     def forward(self, x, adj):
#
#         outs_list = []
#         for k in range(3):
#             xw = torch.mm(x, self.w["weight_%d" % k])
#             out = torch.mm(adj, xw)
#             outs_list.append(out)
#
#         outs = torch.stack(outs_list)
#         outputs = torch.squeeze(torch.sum(outs, dim=0, keepdim=True))
#         outputs = functional.normalize(outputs, dim=1)
#
#         return outputs
#
#
# x = torch.rand(100, 16)
# adj = torch.rand(200, 100)
#
# ex1 = Example()
# out1 = ex1(x, adj)
# out1 = torch.sum(out1)
#
#
# ex2 = Example()
# out2 = ex2(x, adj)
# out2 = torch.sum(out2)
#
# out = out1 + out2
#
# out.backward()
#
#
# for k,v in ex1.named_parameters():
#     print(k,v)
#
# ex2 = Example()
# for k,v in ex2.named_parameters():
#     print(k,v)
#
# from torch import nn
# import torch
# from torch.nn import functional
# torch.autograd.set_detect_anomaly(True)
#
# class test(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.w = {}
#         for k in range(3):
#             self.w["weight_%d" % k] = nn.Parameter(torch.randn(32, 16))
#
#         params = {}
#         params.update(self.w)
#         for k, v in params.items():
#             self.register_parameter(k, v)
#
#         for n, p in self.named_parameters():
#             print(n, p.size())
#
#
#     def forward(self, x):
#
#         o = []
#         for k in range(3):
#             y = torch.mm(x, self.w["weight_%d" % k])
#             self.w["weight_%d" % k] = 1
#
#             o.append(y)
#         outs = torch.stack(o)
#         outputs = torch.squeeze(torch.sum(outs, dim=0, keepdim=True))
#         return outputs
#
# t = test()
# x = torch.randn(100, 32)
# r = t(x)
# r = torch.sum(r)
# r = r.backward()
# print(r)





# o = []
# for i in range(2):
#     x = torch.FloatTensor([[1., 2.], [1., 2.]])
#
#     w[0] = torch.FloatTensor([[2., 4., 3.], [1., 3., 2.]])
#     w[0].requires_grad = True
#
#     w[1] = torch.FloatTensor([[2., 4., 5.], [1., 3., 2.]])
#     w[1].requires_grad = True
#     d = torch.mm(x, w[i])
#     o.append(d)
#
# t = torch.stack(o)
# t = torch.squeeze(torch.sum(t, dim=0, keepdim=True))
# f = torch.sum(t)
# f.backward()
# print(f)

#
# a = 5 % 2
# print(a)
# import scipy.sparse as sp
# chem_feat = sp.identity(9, dtype=float)
# print(chem_feat)
# print("====")

# mat = np.zeros((7, 7))
# mat[0, 0] = 1
# mat[2, 3] = 1
# mat[4, 5] = 1
# mat[6, 6] = 1
# # a = sp.csr_matrix(mat)
# print(mat)
# print(mat.sum(axis=0))
# import copy
# b = []
#
# a = [(0, 1)]
# b.append(copy.deepcopy(a))
# a.append((1, 0))
# b.append(copy.deepcopy(a))
#
# print(b)

# # print(list(range(3)))
# import numpy as np
# # import random
# np.random.seed(1)
# a = [1,2,3,4,5,6]
# np.random.shuffle(a)
# print(a)
from typing import List, Union
import sys
#
# def fixed_unigram_candidate_sampler(
#     true_classes: Union[np.array, torch.Tensor],
#     num_samples: int,
#     unigrams: List[Union[int, float]],
#     distortion: float = 1.):
#
#     if isinstance(true_classes, torch.Tensor):
#         true_classes = true_classes.numpy()
#     if true_classes.shape[0] != num_samples:
#         raise ValueError('true_classes must be a 2D matrix with shape (num_samples, num_true)')
#     unigrams = np.array(unigrams)
#
#     if distortion != 1.:
#         unigrams = unigrams.astype(np.float64) ** distortion
#     indices = np.arange(num_samples)
#     result = np.zeros(num_samples, dtype=np.int64)
#     tmp = indices
#     while len(tmp) > 0:
#         sampler = torch.utils.data.WeightedRandomSampler(unigrams, len(indices))
#         candidates = np.array(list(sampler))
#         candidates = np.reshape(candidates, (len(indices), 1))
#         result[indices] = candidates.T
#         mask = (candidates == true_classes[indices, :])
#         print(candidates)
#         print(true_classes[indices, :])
#         print(mask)
#         mask = mask.sum(1).astype(np.bool)
#         print(mask)
#         print(indices)
#         tmp = indices[mask]
#         print(tmp)
#         result[tmp] += 1
#         print(result)
#         print(len(tmp))
#         print("======")
#     return result
#
# edges_col = np.array([[1],[2],[4],[3],[5],[8],[3],[5],[10]])
# degree_list = [2,3,4,1,5,3,4,2,5,7,1,6,5,4,4]
#
# r = fixed_unigram_candidate_sampler(true_classes=edges_col,
#                                 num_samples=len(edges_col),
#                                 unigrams=degree_list)
# # print(r)

c = []
a = [0, 1, 2, 3]
b = [4, 5]
c.append(a)
c.append(b)
print(c)
