import sys
import torch
from torch import nn
from torch.nn import functional
from utility.function import dropout_sparse, scipy_sparse_to_torch_sparse
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


### graph convolutional layer
class GraphConvolutionLayer(nn.Module):
    def __init__(self,
                 rt_num,
                 adj,
                 input_dim,
                 output_dim,
                 nonzero_feat_num=None,
                 is_sparse=False,
                 is_bias=False,
                 act=torch.relu,
                 device=None):
        super().__init__()

        self.rt_num = rt_num
        self.adj = adj
        self.device = device
        self.nonzero_feat_num = nonzero_feat_num
        self.act = act
        self.is_sparse = is_sparse
        self.is_bias = is_bias

        ### weight & bias
        self.w = {}
        self.b = {}
        for k in range(rt_num):
            self.w["weight_%d" % k] = nn.Parameter(torch.randn(input_dim, output_dim, device=device))
            if is_bias:
                self.b["bias_%d" % k] = nn.Parameter(torch.zeros(output_dim, device=device))

        ### register parameters
        params = {}
        params.update(self.w)
        params.update(self.b)
        for k, v in params.items():
            self.register_parameter(k, v)

    ### forward
    def forward(self, inputs, dropout):
        # if isinstance(inputs, np.ndarray):
        #     inputs = torch.from_numpy(inputs).type(torch.FloatTensor)  # first layer for one-hot feature input
        # inputs = inputs.to(self.device)

        ### dropout
        if self.is_sparse:
            inputs = scipy_sparse_to_torch_sparse(inputs).to(self.device)
            x = dropout_sparse(inputs, dropout, self.nonzero_feat_num)
            # x = dropout_sparse(inputs.to_sparse(), self.dropout, self.nonzero_feat_num)
        else:
            x = functional.dropout(inputs, dropout)

        outs_list = []
        for k in range(self.rt_num):
            ### act((x * w) * adj + bias)
            if self.is_sparse:
                xw = torch.sparse.mm(x, self.w["weight_%d" % k])
            else:
                xw = torch.mm(x, self.w["weight_%d" % k])
            adj = scipy_sparse_to_torch_sparse(self.adj[k]).to(self.device)
            out = torch.sparse.mm(adj, xw)

            ### bias
            if self.is_bias:
                out = out + self.b["bias_%d" % k]

            outs_list.append(self.act(out))

        ### add
        outs = torch.stack(outs_list)
        outputs = torch.squeeze(torch.sum(outs, dim=0, keepdim=True))
        outputs = functional.normalize(outputs, dim=1)

        return outputs


### dedicom layer
class DEDICOMLayer(nn.Module):
    def __init__(self,
                 rt_num,
                 input_dim,
                 device,
                 ):
        super().__init__()

        self.rt_num = rt_num

        ### global_interaction & local_variation
        self.vars = {}
        self.vars["global_interaction"] = nn.Parameter(torch.randn(input_dim, input_dim, device=device))
        for k in range(self.rt_num):
            self.vars["local_variation_%d" % k] = nn.Parameter(torch.randn(input_dim, device=device))

        ### register parameters
        params = {}
        params.update(self.vars)
        for k, v in params.items():
            self.register_parameter(k, v)

    ### forward
    def forward(self, inputs):
        rt_k, edges_embeds_row, edges_embeds_col = inputs

        ### tensor decomposition
        loc = torch.diag(self.vars["local_variation_%d" % rt_k])
        glb = self.vars["global_interaction"]

        p1 = torch.mm(edges_embeds_row, loc)                    # z*D          z*E = z
        p2 = torch.mm(p1, glb)                                  # z*D*R        z*M
        p3 = torch.mm(p2, loc)                                  # z*D*R*D      z*M*E = z*M
        preds = torch.mm(p3, edges_embeds_col.transpose(0, 1))  # z*D*R*D*zt   z*M*zt
        preds = torch.sigmoid(preds)

        return preds

