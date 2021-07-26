import sys
import argparse
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
import scipy.sparse as sp
from utility.classer import logger
from utility.function import mkdir


parser = argparse.ArgumentParser(description='CGINet')
parser.add_argument('--log_file', default="./adj_log", help='')
parser.add_argument('--raw_pkl_path', default="../data/raw-pkl/", help='')
parser.add_argument('--adj_pkl_path', default="../data/adj-pkl/", help='')


class Handler():
    def __init__(self, args, log):
        # arg, log
        self.args = args
        self.log = log

        # num of chem, gene, path
        self.n_chems = 0
        self.n_genes = 0
        self.n_paths = 0

        # raw pkl (edges)
        self.chem1_chem2_list = None
        self.gene1_gene2_list = None
        self.chem_path_list = None
        self.gene_path_list = None
        self.chem_gene_list = None

        # x2num & num2x dictionary
        self.cid2num_dict = {}
        self.num2cid_dict = {}
        self.gid2num_dict = {}
        self.num2gid_dict = {}
        self.pid2num_dict = {}
        self.num2pid_dict = {}

        # adj & degree
        # chem1-chem2; chem2-chem1; gene1-gene2; gene2-gene1
        self.chem1_chem2_graph = nx.Graph()
        self.gene1_gene2_graph = nx.Graph()
        self.chem1_chem2_adj = None
        self.gene1_gene2_adj = None
        self.chem2_chem1_adj = None
        self.gene2_gene1_adj = None
        self.chem1_chem2_deg = None
        self.gene1_gene2_deg = None
        self.chem2_chem1_deg = None
        self.gene2_gene1_deg = None
        # chem-path; path-chem; gene-path; path-gene
        self.chem_path_adj = None
        self.gene_path_adj = None
        self.path_chem_adj = None
        self.path_gene_adj = None
        self.chem_path_deg = None
        self.gene_path_deg = None
        self.path_chem_deg = None
        self.path_gene_deg = None
        # chem-gene; gene-chem
        self.chem_gene_dict = defaultdict(list)
        self.chem_gene_adj_list = []
        self.gene_chem_adj_list = None
        self.chem_gene_deg_list = None
        self.gene_chem_deg_list = None


    # load raw pkl
    def load_raw_pkl(self):
        self.log.info("Loading raw pkl")
        self.chem1_chem2_list = pickle.load(open(self.args.raw_pkl_path + "chem1_chem2_list.pkl", "rb"))
        self.gene1_gene2_list = pickle.load(open(self.args.raw_pkl_path + "gene1_gene2_list.pkl", "rb"))
        self.chem_path_list = pickle.load(open(self.args.raw_pkl_path + "chem_pathway_list.pkl", "rb"))
        self.gene_path_list = pickle.load(open(self.args.raw_pkl_path + "gene_pathway_list.pkl", "rb"))
        self.chem_gene_list = pickle.load(open(self.args.raw_pkl_path + "chem_gene_tiny_list.pkl", "rb"))

        is_tiny = True
        if is_tiny:
            self.chem1_chem2_list = np.random.permutation(self.chem1_chem2_list)[0:2000]
            self.gene1_gene2_list = np.random.permutation(self.gene1_gene2_list)[0:2000]
            self.chem_path_list = np.random.permutation(self.chem_path_list)[0:2000]
            self.gene_path_list = np.random.permutation(self.gene_path_list)[0:2000]
            # list1 = [ins for ins in self.chem_gene_list if ins[2] == "increases^expression"][0:2000]
            # list2 = [ins for ins in self.chem_gene_list if ins[2] == "decreases^reaction"][0:2000]
            # list3 = [ins for ins in self.chem_gene_list if ins[2] == "affects^activity"][0:2000]
            # list4 = [ins for ins in self.chem_gene_list if ins[2] == "affects^phosphorylation"][0:2000]
            # list6 = [ins for ins in self.chem_gene_list if ins[2] == "increases^methylation"]
            list7 = [ins for ins in self.chem_gene_list if ins[2] == "increases^ubiquitination"]
            self.chem_gene_list = list7
            # self.chem_gene_list = list1

    # update chem dict
    def chem_update(self, cid):
        if cid not in self.cid2num_dict.keys():
            self.cid2num_dict[cid] = len(self.cid2num_dict.keys())
            self.num2cid_dict[self.cid2num_dict[cid]] = cid

    # update gene dict
    def gene_update(self, gid):
        if gid not in self.gid2num_dict.keys():
            self.gid2num_dict[gid] = len(self.gid2num_dict.keys())
            self.num2gid_dict[self.gid2num_dict[gid]] = gid

    # update path dict
    def path_update(self, pid):
        if pid not in self.pid2num_dict.keys():
            self.pid2num_dict[pid] = len(self.pid2num_dict.keys())
            self.num2pid_dict[self.pid2num_dict[pid]] = pid

    # create chem & gene & path dict
    def create_dict(self):
        self.log.info("Creating chem & gene & path dictionary")
        for each in self.chem_gene_list:
            self.chem_update(each[0])
            self.gene_update(each[1])
        for each in self.gene_path_list:
            self.gene_update(each[0])
            self.path_update(each[1])
        for each in self.chem_path_list:
            self.chem_update(each[0])
            self.path_update(each[1])
        for each in self.chem1_chem2_list:
            self.chem_update(each[0])
            self.chem_update(each[1])
        for each in self.gene1_gene2_list:
            self.gene_update(each[0])
            self.gene_update(each[1])

    # update the num of chem & gene & path
    def update_num(self):
        # self.log.info("Updating the number of chem & gene & path")
        self.n_chems = len(self.cid2num_dict.keys())
        self.n_genes = len(self.gid2num_dict.keys())
        self.n_paths = len(self.pid2num_dict.keys())
        self.log.info("n_chems->{}, n_genes->{}, n_paths->{}".format(self.n_chems, self.n_genes, self.n_paths))

    # construct chem1-chem2 graph
    def construct_chem1_chem2_graph(self):
        self.log.info("Constructing chem1-chem2 graph")
        self.chem1_chem2_graph.add_edges_from(self.chem1_chem2_list)
        for each in self.cid2num_dict.keys():
            self.chem1_chem2_graph.add_node(each)

    # construct gene1-gene2 graph
    def construct_gene1_gene2_graph(self):
        self.log.info("Constructing gene1-gene2 graph")
        self.gene1_gene2_graph.add_edges_from(self.gene1_gene2_list)
        for each in self.gid2num_dict.keys():
            self.gene1_gene2_graph.add_node(each)

    # construct chem-path adjacency
    def construct_chem_path_adj(self):
        self.log.info("Constructing chem-path adjacency")
        self.chem_path_adj = np.zeros((self.n_chems, self.n_paths))
        for pair in self.chem_path_list:
            self.chem_path_adj[self.cid2num_dict[pair[0]], self.pid2num_dict[pair[1]]] = 1
        self.chem_path_adj = sp.csr_matrix(self.chem_path_adj)
        # self.chem_path_adj = self.chem_path_adj

    # construct gene-path adjacency
    def construct_gene_path_adj(self):
        self.log.info("Constructing gene-path adjacency")
        self.gene_path_adj = np.zeros((self.n_genes, self.n_paths))
        for pair in self.gene_path_list:
            self.gene_path_adj[self.gid2num_dict[pair[0]], self.pid2num_dict[pair[1]]] = 1
        self.gene_path_adj = sp.csr_matrix(self.gene_path_adj)
        # self.gene_path_adj = self.gene_path_adj

    # construct chem-gene adjacency
    def construct_chem_gene_adj(self):
        self.log.info("Constructing chem-gene adjacency")
        for each in self.chem_gene_list:
            self.chem_gene_dict[each[2]].append((each[0], each[1], each[3]))  # (cid, gid, relation, direction)

        i = 0
        rel_num = open(self.args.adj_pkl_path + "rel_num.txt", "w")
        for rel in list(self.chem_gene_dict.keys()):
            self.log.info(str(i) + ": " + rel)
            mat = np.zeros((self.n_chems, self.n_genes))
            for edge in self.chem_gene_dict[rel]:
                chem = edge[0]
                gene = edge[1]
                mat[self.cid2num_dict[chem], self.gid2num_dict[gene]] = 1
            self.chem_gene_adj_list.append(sp.csr_matrix(mat))
            rel_num.write(str(i) + "\t" + rel + "\t" + str(len(self.chem_gene_dict[rel])) + "\n")
            i += 1

    # generate adjacency & degree
    def generate_adj_deg(self):
        self.log.info("Generating adjacency & degree")

        # chem1-chem2; chem2-chem1; gene1-gene2; gene2-gene1
        self.construct_chem1_chem2_graph()
        self.construct_gene1_gene2_graph()
        self.chem1_chem2_adj = nx.adjacency_matrix(self.chem1_chem2_graph)
        self.gene1_gene2_adj = nx.adjacency_matrix(self.gene1_gene2_graph)
        self.chem2_chem1_adj = self.chem1_chem2_adj.transpose()
        self.gene2_gene1_adj = self.gene1_gene2_adj.transpose()
        self.chem1_chem2_deg = np.array(self.chem1_chem2_adj.sum(axis=1)).squeeze()
        self.gene1_gene2_deg = np.array(self.gene1_gene2_adj.sum(axis=1)).squeeze()
        self.chem2_chem1_deg = np.array(self.chem2_chem1_adj.sum(axis=1)).squeeze()
        self.gene2_gene1_deg = np.array(self.gene2_gene1_adj.sum(axis=1)).squeeze()

        # chem-path; path-chem; gene-path; path-gene
        self.construct_chem_path_adj()
        self.construct_gene_path_adj()
        self.path_chem_adj = self.chem_path_adj.transpose()
        self.path_gene_adj = self.gene_path_adj.transpose()
        self.chem_path_deg = np.array(self.chem_path_adj.sum(axis=1)).squeeze()
        self.gene_path_deg = np.array(self.chem_path_adj.sum(axis=1)).squeeze()
        self.path_chem_deg = np.array(self.path_chem_adj.sum(axis=1)).squeeze()
        self.path_gene_deg = np.array(self.path_gene_adj.sum(axis=1)).squeeze()

        # chem-gene; gene-chem
        self.construct_chem_gene_adj()
        self.gene_chem_adj_list = [mat.transpose() for mat in self.chem_gene_adj_list]
        self.chem_gene_deg_list = [np.array(mat.sum(axis=1)).squeeze() for mat in self.chem_gene_adj_list]
        self.gene_chem_deg_list = [np.array(mat.sum(axis=1)).squeeze() for mat in self.gene_chem_adj_list]

    # save dictionary & adjacency & degree
    def save_adj_pkl(self):
        self.log.info("Saving dictionary & adjacency & degree pkl")

        # dictionary
        pickle.dump(self.cid2num_dict, open(self.args.adj_pkl_path + "cid2num_dict.pkl", "wb"))
        pickle.dump(self.num2cid_dict, open(self.args.adj_pkl_path + "num2cid_dict.pkl", "wb"))
        pickle.dump(self.gid2num_dict, open(self.args.adj_pkl_path + "gid2num_dict.pkl", "wb"))
        pickle.dump(self.num2gid_dict, open(self.args.adj_pkl_path + "num2gid_dict.pkl", "wb"))
        pickle.dump(self.pid2num_dict, open(self.args.adj_pkl_path + "pid2num_dict.pkl", "wb"))
        pickle.dump(self.num2pid_dict, open(self.args.adj_pkl_path + "num2pid_dict.pkl", "wb"))

        # chem1-chem2; chem2-chem1; gene1-gene2; gene2-gene1
        pickle.dump(self.chem1_chem2_adj, open(self.args.adj_pkl_path + "chem1_chem2_adj.pkl", 'wb'))
        pickle.dump(self.gene1_gene2_adj, open(self.args.adj_pkl_path + "gene1_gene2_adj.pkl", 'wb'))
        pickle.dump(self.chem2_chem1_adj, open(self.args.adj_pkl_path + "chem2_chem1_adj.pkl", 'wb'))
        pickle.dump(self.gene2_gene1_adj, open(self.args.adj_pkl_path + "gene2_gene1_adj.pkl", 'wb'))
        pickle.dump(self.chem1_chem2_deg, open(self.args.adj_pkl_path + "chem1_chem2_deg.pkl", 'wb'))
        pickle.dump(self.gene1_gene2_deg, open(self.args.adj_pkl_path + "gene1_gene2_deg.pkl", 'wb'))
        pickle.dump(self.chem2_chem1_deg, open(self.args.adj_pkl_path + "chem2_chem1_deg.pkl", 'wb'))
        pickle.dump(self.gene2_gene1_deg, open(self.args.adj_pkl_path + "gene2_gene1_deg.pkl", 'wb'))

        # chem-path; path-chem; gene-path; path-gene
        pickle.dump(self.chem_path_adj, open(self.args.adj_pkl_path + "chem_path_adj.pkl", 'wb'))
        pickle.dump(self.gene_path_adj, open(self.args.adj_pkl_path + "gene_path_adj.pkl", 'wb'))
        pickle.dump(self.path_chem_adj, open(self.args.adj_pkl_path + "path_chem_adj.pkl", 'wb'))
        pickle.dump(self.path_gene_adj, open(self.args.adj_pkl_path + "path_gene_adj.pkl", 'wb'))
        pickle.dump(self.chem_path_deg, open(self.args.adj_pkl_path + "chem_path_deg.pkl", 'wb'))
        pickle.dump(self.gene_path_deg, open(self.args.adj_pkl_path + "gene_path_deg.pkl", 'wb'))
        pickle.dump(self.path_chem_deg, open(self.args.adj_pkl_path + "path_chem_deg.pkl", 'wb'))
        pickle.dump(self.path_gene_deg, open(self.args.adj_pkl_path + "path_gene_deg.pkl", 'wb'))

        # chem-gene; gene-chem
        pickle.dump(self.chem_gene_adj_list, open(self.args.adj_pkl_path + "chem_gene_adj_list.pkl", 'wb'))
        pickle.dump(self.gene_chem_adj_list, open(self.args.adj_pkl_path + "gene_chem_adj_list.pkl", 'wb'))
        pickle.dump(self.chem_gene_deg_list, open(self.args.adj_pkl_path + "chem_gene_deg_list.pkl", 'wb'))
        pickle.dump(self.gene_chem_deg_list, open(self.args.adj_pkl_path + "gene_chem_deg_list.pkl", 'wb'))

def gen_adj():
    args = parser.parse_args()
    mkdir(args.adj_pkl_path)
    log = logger(args.adj_pkl_path + args.log_file)
    log.info("Running gen_adj.py")

    hr = Handler(args, log)
    hr.load_raw_pkl()
    hr.create_dict()
    hr.update_num()
    hr.generate_adj_deg()
    hr.save_adj_pkl()

    log.info("Finishing gen_adj.py")


if __name__ == '__main__':
    gen_adj()
