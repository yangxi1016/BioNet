import sys
import torch
from torch import nn
import numpy as np
from model.layer import GraphConvolutionLayer, DEDICOMLayer
from collections import defaultdict
from utility.function import fixed_unigram_candidate_sampler


### graph convolutional encoder
class GCNEncoder(nn.Module):
    def __init__(self, dp, args, log):
        super().__init__()

        self.log = log
        # self.log.info("Build GCN encoder")

        self.dropout = 0.

        self.rt_num_dict_s1 = dp.rt_num_dict_s1
        self.rt_num_dict_s2 = dp.rt_num_dict_s2

        self.layer_dict = {}
        # layers s1
        for rel_type in self.rt_num_dict_s1:
            i, j = rel_type
            self.layer_dict["s1_l1_i{}_j{}".format(i, j)] = GraphConvolutionLayer(
                rt_num=dp.rt_num_dict_s1[rel_type],
                adj=dp.adj_dict_s1_tr[rel_type],
                input_dim=dp.feat_num_dict[j],
                output_dim=args.hid_dims[0],
                nonzero_feat_num=dp.nonzero_feat_num_dict[j],
                is_sparse=True,
                device=args.device)
            self.layer_dict["s1_l2_i{}_j{}".format(i, j)] = GraphConvolutionLayer(
                rt_num=dp.rt_num_dict_s1[rel_type],
                adj=dp.adj_dict_s1_tr[rel_type],
                input_dim=args.hid_dims[0],
                output_dim=args.hid_dims[1],
                device=args.device)

        ### layers s2
        for rel_type in self.rt_num_dict_s2:
            i, j = rel_type
            self.layer_dict["s2_l1_i{}_j{}".format(i, j)] = GraphConvolutionLayer(
                rt_num=dp.rt_num_dict_s2[rel_type],
                adj=dp.adj_dict_s2_tr[rel_type],
                input_dim=args.hid_dims[1],
                output_dim=args.hid_dims[2],
                device=args.device)
            self.layer_dict["s2_l2_i{}_j{}".format(i, j)] = GraphConvolutionLayer(
                rt_num=dp.rt_num_dict_s2[rel_type],
                adj=dp.adj_dict_s2_tr[rel_type],
                input_dim=args.hid_dims[2],
                output_dim=args.hid_dims[3],
                device=args.device)
        
        ### add modules
        modules = {}
        modules.update(self.layer_dict)
        for k, v in modules.items():
            name = "gcn_layer_{}".format(k)
            self.add_module(name, v)

    ### update dropout
    def update_dropout(self, dropout):
        self.dropout = dropout

    ### get_embed
    def get_embed(self, prefix, rt_num_dict, inputs):
        # self.log.info(" -> forward GCN layer {}".format(prefix))
        embed_dict = defaultdict(list)
        for rel_type in rt_num_dict:
            i, j = rel_type
            embed_dict[i].append(self.layer_dict["{}_i{}_j{}".format(prefix, i, j)](inputs[j], self.dropout))
        result = {}
        for i, embeds in embed_dict.items():
            embeds = torch.stack(embeds)
            result[i] = torch.squeeze(torch.sum(embeds, dim=0, keepdim=True))
        return result

    ### forward
    def forward(self, inputs):

        # self.log.info(" -> forward GCN layers")

        ### embeds s1
        embed_s1_1 = self.get_embed("s1_l1", self.rt_num_dict_s1, inputs)
        embed_s1_2 = self.get_embed("s1_l2", self.rt_num_dict_s1, embed_s1_1)

        ### embeds s2
        embed_s2_1 = self.get_embed("s2_l1", self.rt_num_dict_s2, embed_s1_2)
        embed_s2_2 = self.get_embed("s2_l2", self.rt_num_dict_s2, embed_s2_1)

        return embed_s2_2


### tensor decomposition decoder
class TDDecoder(nn.Module):
    def __init__(self, dp, args, log):
        super().__init__()

        self.log = log
        # self.log.info("Build TD decoder")

        self.dedicom_layer_dict_s2 = {}
        for rel_type in dp.rt_num_dict_s2:
            self.dedicom_layer_dict_s2[rel_type] = DEDICOMLayer(
                rt_num=dp.rt_num_dict_s2[rel_type],
                input_dim=args.hid_dims[-1],
                device=args.device)

        ### add modules
        modules = {}
        modules.update(self.dedicom_layer_dict_s2)
        for k, v in modules.items():
            i, j = k
            name = "td_layer_{}_{}".format(i, j)
            self.add_module(name, v)

    ### forward
    def forward(self, inputs):
        ### inputs
        rel_type, rt_k, edges_row, edges_col, embeds, tag = inputs
        # self.log.info(" -> forward {} TD layers".format(tag))

        ### embeds
        embeds_row = embeds[rel_type[0]]
        embeds_col = embeds[rel_type[1]]

        ### edges embeds
        edges_embeds_row = embeds_row[edges_row]
        edges_embeds_col = embeds_col[edges_col]

        ### tensor decomposition
        preds = self.dedicom_layer_dict_s2[rel_type]((rt_k, edges_embeds_row, edges_embeds_col))

        return preds


### encoder-decoder
class CGINet(nn.Module):
    def __init__(self, dp, args, log):
        super().__init__()

        self.dp = dp
        self.args = args

        ### build encoder & decoder
        self.encoder = GCNEncoder(dp, args, log)
        self.decoder = TDDecoder(dp, args, log)

        # ### encode node embeddings
        # self.embeds = self.encoder(self.dp.feat_dict)
        """put it here, would throw <inplace error> at 'xw = torch.mm(x, self.weight)' in layer.py"""

    ### forward
    def forward(self, inputs):
        ### inputs
        rel_type, rt_k, edges = inputs
        ### outputs
        preds_pos = None
        preds_neg = None

        ### edges row & edges col
        edges_row = torch.squeeze(torch.index_select(edges, 1, torch.tensor([0]).to(self.args.device)))
        edges_col = torch.squeeze(torch.index_select(edges, 1, torch.tensor([1]).to(self.args.device)))
        edges_row = edges_row.type(torch.LongTensor)
        edges_col = edges_col.type(torch.LongTensor)

        ## encode node embeddings
        self.encoder.update_dropout(self.dp.dropout)
        self.embeds = self.encoder(self.dp.feat_dict)

        ### decode, positive samples
        inputs_pos = (rel_type, rt_k, edges_row, edges_col, self.embeds, "Pos")
        preds_pos = self.decoder(inputs_pos)
        preds_pos = torch.diag(preds_pos)

        ### negative sampling for training
        if self.dp.is_train:
            ### decode, negative samples
            edges_col = edges_col.view(-1, 1)
            rel_type_trans = (rel_type[1], rel_type[0])
            edges_col_neg = fixed_unigram_candidate_sampler(
                true_classes=edges_col,
                num_samples=len(edges_col),
                unigrams=self.dp.deg_dict_s2[rel_type_trans][rt_k],
                distortion=self.args.distortion)
            inputs_neg = (rel_type, rt_k, edges_row, edges_col_neg, self.embeds, "Neg")
            preds_neg = self.decoder(inputs_neg)
            preds_neg = torch.diag(preds_neg)

        return preds_pos, preds_neg

