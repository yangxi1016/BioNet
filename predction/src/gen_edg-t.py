import sys
import argparse
import pickle
import scipy.sparse as sp
import numpy as np
from utility.function import mkdir, sparse_to_tuple
from utility.classer import logger
from utility.prepare import DataPreparation

parser = argparse.ArgumentParser(description='CGINet')
parser.add_argument('--log_file', default="edg_log", help='')
parser.add_argument('--adj_pkl_path', default="../data/adj-pkl/", help='')
parser.add_argument('--edg_pkl_path', default="../data/edg-pkl/", help='')
parser.add_argument('--va_te_rate', default=0.1, help='')


class Handler():
    def __init__(self, arg, log):
        # arg, log
        self.arg = arg
        self.log = log

        # relation type adjacency dictionary; s1: subgraph 1, s2: subgraph 2
        self.adj_dict_s1 = {}
        self.adj_dict_s2 = {}

        # relation type num dictionary
        self.rt_num_dict_s1 = {}
        self.rt_num_dict_s2 = {}

        # train & valid & test edges
        self.tr_edges = {}
        self.va_edges = {}
        self.te_edges = {}

        # relation type adjacency for new train edges
        self.adj_dict_s1_tr = {}
        self.adj_dict_s2_tr = {}

        # valid & test false edges
        self.te_false_edges = {}
        self.va_false_edges = {}


    def pre_adj(self):
        self.log.info("Preparing adjacency")
        
        # data preparation
        dp = DataPreparation(self.arg)
        self.adj_dict_s1 = dp.adj_dict_s1
        self.adj_dict_s2 = dp.adj_dict_s2
        self.rt_num_dict_s1 = dp.rt_num_dict_s1
        self.rt_num_dict_s2 = dp.rt_num_dict_s2
        n_rel_type_s1 = sum(self.rt_num_dict_s1.values())
        n_rel_type_s2 = sum(self.rt_num_dict_s2.values())

        self.log.info("Num: chem->" + str(dp.n_chems) + ", gene->" + str(dp.n_genes) + ", path->" + str(dp.n_paths) + ", all->" + str(dp.n_chems + dp.n_genes + dp.n_paths))
        self.log.info("Relation type num: s1->" + str(n_rel_type_s1) + ", s2->" + str(n_rel_type_s2) + ", all->" + str(n_rel_type_s1 + n_rel_type_s2))


    # preprocess graph
    def preprocess_graph(self, adj):
        # adj = sp.coo_matrix(adj)
        # if adj.shape[0] == adj.shape[1]:
        #     adj_ = adj + sp.eye(adj.shape[0])
        #     rowsum = np.array(adj_.sum(1))
        #     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        #     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        # else:
        #     rowsum = np.array(adj.sum(1))
        #     colsum = np.array(adj.sum(0))
        #     rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
        #     coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
        #     adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
        # return sparse_to_tuple(adj)
        return adj

    # is member
    def _ismember(self, a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)
        return np.any(rows_close)


    # generate valid & test false edges
    def gen_va_te_false_edges(self, rel_type, type_idx, edges_all, va_edges, te_edges):
        va_false_edges = []
        while len(va_false_edges) < len(va_edges):
            idx_i = np.random.randint(0, self.adj_dict_s2[rel_type][type_idx].shape[0])
            idx_j = np.random.randint(0, self.adj_dict_s2[rel_type][type_idx].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if va_false_edges:
                if self._ismember([idx_i, idx_j], va_false_edges):
                    continue
            va_false_edges.append([idx_i, idx_j])

        te_false_edges = []
        while len(te_false_edges) < len(te_edges):
            idx_i = np.random.randint(0, self.adj_dict_s2[rel_type][type_idx].shape[0])
            idx_j = np.random.randint(0, self.adj_dict_s2[rel_type][type_idx].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if te_false_edges:
                if self._ismember([idx_i, idx_j], te_false_edges):
                    continue
            te_false_edges.append([idx_i, idx_j])

        return va_false_edges, te_false_edges


    # generate train & valid & test edges
    def gen_tr_va_te_edges(self, rel_type, type_idx):
        edges_all, _, _ = sparse_to_tuple(self.adj_dict_s2[rel_type][type_idx])
        num_va = max(20, int(np.floor(edges_all.shape[0] * self.arg.va_te_rate)))
        num_te = max(20, int(np.floor(edges_all.shape[0] * self.arg.va_te_rate)))
        all_edge_idx = list(range(edges_all.shape[0]))
        np.random.shuffle(all_edge_idx)
        va_edge_idx = all_edge_idx[:num_va]
        va_edges = edges_all[va_edge_idx]
        te_edge_idx = all_edge_idx[num_va:(num_va + num_te)]
        te_edges = edges_all[te_edge_idx]
        tr_edges = np.delete(edges_all, np.hstack([te_edge_idx, va_edge_idx]), axis=0)

        # generate valid & test false edges
        va_false_edges, te_false_edges = self.gen_va_te_false_edges(rel_type, type_idx, edges_all, va_edges, te_edges)

        # rebuild adjacency
        data = np.ones(tr_edges.shape[0])
        adj_dict_s2_tr = sp.csr_matrix(
            (data, (tr_edges[:, 0], tr_edges[:, 1])),
            shape=self.adj_dict_s2[rel_type][type_idx].shape
        )
        self.adj_dict_s2_tr[rel_type][type_idx] = self.preprocess_graph(adj_dict_s2_tr)

        self.tr_edges[rel_type][type_idx] = tr_edges
        self.va_edges[rel_type][type_idx] = va_edges
        self.te_edges[rel_type][type_idx] = te_edges
        self.va_false_edges[rel_type][type_idx] = np.array(va_false_edges)
        self.te_false_edges[rel_type][type_idx] = np.array(te_false_edges)
        self.log.info("<" + str(rel_type) + ", " + str(type_idx) + ">: "
                      + "tr->" + str(len(tr_edges)) + ", va->" + str(len(va_edges))
                      + ", te->" + str(len(te_edges)) + ", all->" + str(len(tr_edges) + len(va_edges) + len(te_edges))
                      + " | va-false->" + str(len(va_false_edges)) + ", te-false->" + str(len(te_false_edges)))


    # generate train & valid & test edges | valid & test false edges
    def gen_edges(self):
        self.log.info("Generating train & valid & test edges | valid & test false edges")

        # train & valid & test edges
        self.tr_edges = {rel_type: [None] * n for rel_type, n in self.rt_num_dict_s2.items()}
        self.va_edges = {rel_type: [None] * n for rel_type, n in self.rt_num_dict_s2.items()}
        self.te_edges = {rel_type: [None] * n for rel_type, n in self.rt_num_dict_s2.items()}
        self.te_false_edges = {rel_type: [None] * n for rel_type, n in self.rt_num_dict_s2.items()}
        self.va_false_edges = {rel_type: [None] * n for rel_type, n in self.rt_num_dict_s2.items()}

        # relation type adjacency for new train edges
        self.adj_dict_s1_tr = {edge_type: [None] * n for edge_type, n in self.rt_num_dict_s1.items()}
        self.adj_dict_s2_tr = {edge_type: [None] * n for edge_type, n in self.rt_num_dict_s2.items()}

        # preprocess adj_dict_s1_tr
        for i, j in self.rt_num_dict_s1:
            for k in range(self.rt_num_dict_s1[(i, j)]):
                self.adj_dict_s1_tr[(i, j)][k] = self.preprocess_graph(self.adj_dict_s1[(i, j)][k])

        # generate train & valid & test edges | valid & test false edges
        for i, j in self.rt_num_dict_s2:
            for k in range(self.rt_num_dict_s2[(i, j)]):
                rel_type = (i, j)
                type_idx = k
                self.gen_tr_va_te_edges(rel_type, type_idx)

    def save_edges_pkl(self):
        pickle.dump(self.tr_edges, open(self.arg.edg_pkl_path + "tr_edges.pkl", 'wb'))
        pickle.dump(self.va_edges, open(self.arg.edg_pkl_path + "va_edges.pkl", 'wb'))
        pickle.dump(self.te_edges, open(self.arg.edg_pkl_path + "te_edges.pkl", 'wb'))
        pickle.dump(self.va_false_edges, open(self.arg.edg_pkl_path + "va_false_edges.pkl", 'wb'))
        pickle.dump(self.te_false_edges, open(self.arg.edg_pkl_path + "te_false_edges.pkl", 'wb'))
        pickle.dump(self.adj_dict_s1_tr, open(self.arg.edg_pkl_path + "adj_dict_s1_tr.pkl", 'wb'))
        pickle.dump(self.adj_dict_s2_tr, open(self.arg.edg_pkl_path + "adj_dict_s2_tr.pkl", 'wb'))


def gen_edg():
    arg = parser.parse_args()
    log = logger(arg.log_file)
    log.info("Running gen_edg.py")
    mkdir(arg.edg_pkl_path)

    hr = Handler(arg, log)
    hr.pre_adj()
    hr.gen_edges()
    hr.save_edges_pkl()

    log.info("Finishing gen_edg.py")

if __name__ == '__main__':
    gen_edg()


