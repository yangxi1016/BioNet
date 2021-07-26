import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset


class DataPreparation:

    def __init__(self, args):

        self.args = args

        self.is_train = True
        self.dropout = 0.

        '''
        load node dict
        '''
        ### load n_chems & n_genes & n_paths
        self.cid2num_dict = pickle.load(open(args.adj_pkl_path + "cid2num_dict.pkl", "rb"))
        self.gid2num_dict = pickle.load(open(args.adj_pkl_path + "gid2num_dict.pkl", "rb"))
        self.pid2num_dict = pickle.load(open(args.adj_pkl_path + "pid2num_dict.pkl", "rb"))
        self.n_chems = len(self.cid2num_dict.keys())
        self.n_genes = len(self.gid2num_dict.keys())
        self.n_paths = len(self.pid2num_dict.keys())

        '''
        load adj
        '''
        ### chem1-chem2; chem2-chem1; gene1-gene2; gene2-gene1
        self.chem1_chem2_adj = pickle.load(open(args.adj_pkl_path + "chem1_chem2_adj.pkl", "rb"))
        self.gene1_gene2_adj = pickle.load(open(args.adj_pkl_path + "gene1_gene2_adj.pkl", "rb"))
        ### chem-path; path-chem; gene-path; path-gene
        self.chem_path_adj = pickle.load(open(args.adj_pkl_path + "chem_path_adj.pkl", "rb"))
        self.gene_path_adj = pickle.load(open(args.adj_pkl_path + "gene_path_adj.pkl", "rb"))
        self.path_chem_adj = pickle.load(open(args.adj_pkl_path + "path_chem_adj.pkl", "rb"))
        self.path_gene_adj = pickle.load(open(args.adj_pkl_path + "path_gene_adj.pkl", "rb"))
        ### chem-gene; gene-chem
        self.chem_gene_adj_list = pickle.load(open(args.adj_pkl_path + "chem_gene_adj_list.pkl", "rb"))
        self.gene_chem_adj_list = pickle.load(open(args.adj_pkl_path + "gene_chem_adj_list.pkl", "rb"))

        '''
        load deg
        '''
        ### chem1-chem2; chem2-chem1; gene1-gene2; gene2-gene1
        self.chem1_chem2_deg = pickle.load(open(args.adj_pkl_path + "chem1_chem2_deg.pkl", "rb"))
        self.gene1_gene2_deg = pickle.load(open(args.adj_pkl_path + "gene1_gene2_deg.pkl", "rb"))
        ### chem-path; path-chem; gene-path; path-gene
        self.chem_path_deg = pickle.load(open(args.adj_pkl_path + "chem_path_deg.pkl", "rb"))
        self.gene_path_deg = pickle.load(open(args.adj_pkl_path + "gene_path_deg.pkl", "rb"))
        self.path_chem_deg = pickle.load(open(args.adj_pkl_path + "path_chem_deg.pkl", "rb"))
        self.path_gene_deg = pickle.load(open(args.adj_pkl_path + "path_gene_deg.pkl", "rb"))
        # chem-gene; gene-chem
        self.chem_gene_deg_list = pickle.load(open(args.adj_pkl_path + "chem_gene_deg_list.pkl", "rb"))
        self.gene_chem_deg_list = pickle.load(open(args.adj_pkl_path + "gene_chem_deg_list.pkl", "rb"))

        '''
        set adj, deg
        '''
        ### relation type adjacency dictionary; s1: subgraph 1, s2: subgraph 2
        self.adj_dict_s1 = {}
        self.adj_dict_s2 = {}
        self.set_adj_dict()

        ### train edges => relation type adjacency dictionary; s1: subgraph 1, s2: subgraph 2
        self.adj_dict_s1_tr = {}
        self.adj_dict_s2_tr = {}
        self.set_adj_dict_tr(args)

        ### relation type num dictionary
        self.rt_num_dict_s1 = {}
        self.rt_num_dict_s2 = {}
        self.set_rt_num_dict()

        ### relation type dimension dictionary
        self.rt_dim_dict_s1 = {}
        self.rt_dim_dict_s2 = {}
        self.set_rt_dim_dict()

        ### relation category to index dictionary
        self.rc2idx_dict_s1 = {}
        self.rc2idx_dict_s2 = {}
        self.idx2rc_dict_s1 = {}
        self.idx2rc_dict_s2 = {}
        self.set_rc_idx_dict()

        ### relation type degree dictionary
        self.deg_dict_s1 = {}
        self.deg_dict_s2 = {}
        self.set_deg_dict()

        '''
        set feat
        '''
        self.feat_dict = {}
        self.feat_num_dict = {}
        self.nonzero_feat_num_dict = {}
        self.set_feat_dict()

        '''
        set adj dimension
        '''
        self.adj_dim_dict_s1 = {}
        self.adj_dim_dict_s2 = {}
        self.set_adj_dim_dict()


    ### set_adj_dict
    def set_adj_dict(self):
        '''
        relation type definition
        '''
        # relation type adjacency dictionary; s1: subgraph 1, s2: subgraph 2
        self.adj_dict_s1 = {
            (0, 0): [self.chem1_chem2_adj],
            (0, 2): [self.chem_path_adj],
            (1, 1): [self.gene1_gene2_adj],
            (1, 2): [self.gene_path_adj],
            (2, 0): [self.path_chem_adj],
            (2, 1): [self.path_gene_adj],
        }
        self.adj_dict_s2 = {
            (0, 1): self.chem_gene_adj_list,
            (1, 0): self.gene_chem_adj_list,
        }


    ### set_adj_dict_tr
    def set_adj_dict_tr(self, args):
        path1 = args.edg_pkl_path + "adj_dict_s1_tr.pkl"
        path2 = args.edg_pkl_path + "adj_dict_s2_tr.pkl"
        if os.path.exists(path1) and os.path.exists(path2):
            self.adj_dict_s1_tr = pickle.load(open(path1, "rb"))
            self.adj_dict_s2_tr = pickle.load(open(path2, "rb"))


    ### set rt_num_dict
    def set_rt_num_dict(self):
        # relation type num dictionary
        self.rt_num_dict_s1 = {k: len(v) for k, v in self.adj_dict_s1.items()}
        self.rt_num_dict_s2 = {k: len(v) for k, v in self.adj_dict_s2.items()}


    ### set_rt_dim_dict
    def set_rt_dim_dict(self):
        # relation type dimension dictionary
        self.rt_dim_dict_s1 = {k: [adj.shape for adj in adjs] for k, adjs in self.adj_dict_s1.items()}
        self.rt_dim_dict_s2 = {k: [adj.shape for adj in adjs] for k, adjs in self.adj_dict_s2.items()}


    ### set_rc_idx_dict
    def set_rc_idx_dict(self):
        idx = 0
        for i, j in self.rt_num_dict_s1:
            for k in range(self.rt_num_dict_s1[(i, j)]):
                self.rc2idx_dict_s1[(i, j, k)] = idx
                self.idx2rc_dict_s1[idx] = (i, j, k)
                idx = idx + 1

        idx = 0
        for i, j in self.rt_num_dict_s2:
            for k in range(self.rt_num_dict_s2[(i, j)]):
                self.rc2idx_dict_s2[(i, j, k)] = idx
                self.idx2rc_dict_s2[idx] = (i, j, k)
                idx = idx + 1


    ### set_deg_dict
    def set_deg_dict(self):
        self.deg_dict_s1 = {
            (0, 0): [self.chem1_chem2_deg],
            (0, 2): [self.chem_path_deg],
            (1, 1): [self.gene1_gene2_deg],
            (1, 2): [self.gene_path_deg],
            (2, 0): [self.path_chem_deg],
            (2, 1): [self.path_gene_deg],
        }
        self.deg_dict_s2 = {
            (0, 1): self.chem_gene_deg_list,
            (1, 0): self.gene_chem_deg_list,
        }


    ### set_feat_dict
    def set_feat_dict(self):
        # feature (chem & gene & path)
        chem_feat = sp.identity(self.n_chems, dtype=float)
        chem_nonzero_feat_num, chem_feat_num = chem_feat.shape
        gene_feat = sp.identity(self.n_genes, dtype=float)
        gene_nonzero_feat_num, gene_feat_num = gene_feat.shape
        path_feat = sp.identity(self.n_paths, dtype=float)
        path_nonzero_feat_num, path_feat_num = path_feat.shape
        self.feat_dict = {
            0: chem_feat,
            1: gene_feat,
            2: path_feat,
        }
        self.feat_num_dict = {
            0: chem_feat_num,
            1: gene_feat_num,
            2: path_feat_num,
        }
        self.nonzero_feat_num_dict = {
            0: chem_nonzero_feat_num,
            1: gene_nonzero_feat_num,
            2: path_nonzero_feat_num,
        }


    ### set_adj_dim_dict
    def set_adj_dim_dict(self):
        self.adj_dim_dict_s1 = {k: [adj.shape for adj in adjs] for k, adjs in self.adj_dict_s1.items()}
        self.adj_dim_dict_s2 = {k: [adj.shape for adj in adjs] for k, adjs in self.adj_dict_s2.items()}


    ### set dropout
    def set_dropout(self, dropout):
        self.dropout = dropout


    ### load edges
    def load_edges(self, file):
        ### load train edges
        edges = pickle.load(open(self.args.edg_pkl_path + file, "rb"))
        rc_idx = 0  # relation category index
        rc_lens_list = []
        edges_list = []
        labels_list = []
        for rel_type, edge_lists in edges.items():
            for edge_list in edge_lists:
                rc_lens_list.append(len(edge_list))
                ### shuffle
                np.random.shuffle(edge_list)
                for edge in edge_list:
                    edges_list.append(edge)
                    labels_list.append(rc_idx)
                rc_idx = rc_idx + 1
        edges_array = np.array(edges_list)
        labels_array = np.array(labels_list)
        edges = TensorDataset(torch.from_numpy(edges_array), torch.from_numpy(labels_array))
        return edges, rc_lens_list





