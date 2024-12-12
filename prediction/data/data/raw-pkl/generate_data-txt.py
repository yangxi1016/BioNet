
import sys
import csv
import pickle
from collections import defaultdict

chem2cid_dict = {}

cid2chem_dict = {}
gid2gene_dict = {}
pid2pathway_dict = {}

cid2num_dict = {}
num2cid_dict = {}
gid2num_dict = {}
num2gid_dict = {}
pid2num_dict = {}
num2pid_dict = {}

def chem2cid_query():
    print("Loading chem2cid dictionary ...")

    reader = open("../data/raw-data/ctd-data-all/chem2cid-all.txt", "r").read().strip().split("\n")
    for line in reader:
        terms = line.split("\t")
        chem = terms[0]
        cid = terms[1]
        chem2cid_dict[chem] = cid

def chem_update(cid, chem):
    if cid not in cid2chem_dict.keys():
        cid2chem_dict[cid] = chem
        cid2num_dict[cid] = len(cid2num_dict.keys())
        num2cid_dict[len(cid2num_dict.keys())] = cid
def gene_update(gid, gene):
    if gid not in gid2gene_dict.keys():
        gid2gene_dict[gid] = gene
        gid2num_dict[gid] = len(gid2num_dict.keys())
        num2gid_dict[len(gid2num_dict.keys())] = gid
def pathway_update(pid, pathway):
    if pid not in pid2pathway_dict.keys():
        pid2pathway_dict[pid] = pathway
        pid2num_dict[pid] = len(pid2num_dict.keys())
        num2pid_dict[len(pid2num_dict.keys())] = pid

# chem-gene
def generate_chem_gene():
    #####替换测试文件########
    f = open(r"Melatonin-1203-test-19.txt", "r")
    csv_readr = f.readlines()

    lists = []
    chem_gene_list = []
    for line in csv_readr:
        lists=line.split("\t")
        chem = lists[0]
        cid = chem2cid_dict[chem]
        chem_update(cid, chem)

        gene = lists[3]
        gid = "GID" + str(lists[4])
        gene_update(gid, gene)

        relation_detail = lists[8]

        relations = lists[9].split("|")
        for relation in relations:
            c_idx = relation_detail.index(chem)
            g_idx = relation_detail.index(gene)
            if c_idx < g_idx:
                direction = "c->g"
            else:
                direction = "g<-c"
            chem_gene_list.append((cid, gid, relation, direction))
            # chem_gene_list.append((cid, gid, relation))


    chem_gene_tiny_list = []
    rel2cgs_dict = defaultdict(list)
    for each in set(chem_gene_list):
        rel2cgs_dict[each[2]].append((each[0], each[1], each[3]))
        # rel2cgs_dict[each[2]].append((each[0], each[1]))
    for rel, cgs in rel2cgs_dict.items():
        if len(cgs) >= 178: # 178  251
            for cg in cgs:
                chem_gene_tiny_list.append((cg[0], cg[1], rel, cg[2]))
                # chem_gene_tiny_list.append((cg[0], cg[1], rel))

    pickle.dump(list(set(chem_gene_tiny_list)), open("chem_gene_tiny_list.pkl", "wb"))
    pickle.dump(list(set(chem_gene_list)), open("chem_gene_list_CDH2.pkl", "wb"))
    print("Chemical-gene tiny counts: ", len(set(chem_gene_tiny_list)))
    print("Chemical-gene counts: ", len(set(chem_gene_list)))


# gene-pathway
def generate_gene_pathway():
    csv_readr = csv.reader(open("../data/raw-data/ctd-data-all/CTD_genes_pathways.csv", "r"))

    gene_pathway_list = []
    for line in csv_readr:

        gene = line[0]
        gid = "GID" + str(line[1])
        gene_update(gid, gene)

        pathway = line[2]
        pid = "PID" + str(line[3])
        pathway_update(pid, pathway)

        gene_pathway_list.append((gid, pid))

    pickle.dump(list(set(gene_pathway_list)), open("gene_pathway_list.pkl", "wb"))
    print("Cene-pathway counts: ", len(set(gene_pathway_list)))


# chem-pathway
def generate_chem_pathway():
    csv_readr = csv.reader(open("../data/raw-data/ctd-data-all/CTD_chem_pathways_enriched.csv", "r"))

    chem_pathway_list = []
    for line in csv_readr:

        chem = line[0]
        cid = chem2cid_dict[chem]
        chem_update(cid, chem)

        pathway = line[3]
        pid = "PID" + str(line[4])
        pathway_update(pid, pathway)

        chem_pathway_list.append((cid, pid))

    pickle.dump(list(set(chem_pathway_list)), open("chem_pathway_list.pkl", "wb"))
    print("Chemical-pathway counts: ", len(set(chem_pathway_list)))


# chem-chem
def generate_chem_chem():
    csv_readr = csv.reader(open("../data/raw-data/stitch-data-all/chemical_chemical.links.v5.0.tsv", "r"), delimiter="\t")

    chem1_chem2_list = []
    for line in csv_readr:

        cid1 = "CID" + str(line[0][4:]) + "#1"
        cid2 = "CID" + str(line[1][4:]) + "#1"

        # cid1 = "CID" + str(line[0][4:])
        # cid2 = "CID" + str(line[1][4:])

        # CID00018827

        if cid1 in cid2chem_dict.keys() and cid2 in cid2chem_dict.keys():

            # use cid as key and value
            chem_update(cid1, cid1)
            chem_update(cid2, cid2)

            chem1_chem2_list.append((cid1, cid2))

    pickle.dump(list(set(chem1_chem2_list)), open("chem1_chem2_list.pkl", "wb"))
    print("Chemical1-Chemical2 counts: ", len(set(chem1_chem2_list)))


# gene-gene
def generate_gene_gene():
    csv_readr = csv.reader(open("../data/raw-data/decagon-data/bio-decagon-ppi.csv", "r"))
    header = next(csv_readr)

    gene1_gene2_list = []
    for line in csv_readr:

        gid1 = "GID" + str(line[0])
        gid2 = "GID" + str(line[1])

        if gid1 in gid2gene_dict.keys() and gid2 in gid2gene_dict.keys():

            # use gid as key and value
            gene_update(gid1, gid1)
            gene_update(gid2, gid2)

            gene1_gene2_list.append((gid1, gid2))

    pickle.dump(list(set(gene1_gene2_list)), open("gene1_gene2_list.pkl", "wb"))
    print("Gene1-gene2 counts: ", len(set(gene1_gene2_list)))


def save():
    print("Saving pkl file ...")

    pickle.dump(cid2chem_dict, open("cid2chem_dict.pkl", "wb"))
    pickle.dump(gid2gene_dict, open("gid2gene_dict.pkl", "wb"))
    pickle.dump(pid2pathway_dict, open("pid2pathway_dict.pkl", "wb"))

    pickle.dump(cid2num_dict, open("cid2num_dict.pkl", "wb"))
    pickle.dump(num2cid_dict, open("num2cid_dict.pkl", "wb"))
    pickle.dump(gid2num_dict, open("gid2num_dict.pkl", "wb"))
    pickle.dump(num2gid_dict, open("num2gid_dict.pkl", "wb"))
    pickle.dump(pid2num_dict, open("pid2num_dict.pkl", "wb"))
    pickle.dump(num2pid_dict, open("num2pid_dict.pkl", "wb"))


chem2cid_query()
generate_chem_gene()
generate_gene_pathway()
generate_chem_pathway()
generate_chem_chem()
generate_gene_gene()
save()

print("Chemical count: ", len(cid2chem_dict.keys()))
print("Gene count: ", len(gid2gene_dict.keys()))
print("Pathway count: ", len(pid2pathway_dict.keys()))


'''
Loading chem2cid dictionary ...
Chemical-gene tiny counts:  1798796
Chemical-gene counts:  1801222
Cene-pathway counts:  135809
Chemical-pathway counts:  1285158
Chemical1-Chemical2 counts:  720155
Gene1-gene2 counts:  713469
Saving pkl file ...
Chemical count:  14273
Gene count:  51070
Pathway count:  2363






Loading chem2cid dictionary ...
Chemical-gene tiny counts:  1785942
Chemical-gene counts:  1788352
Cene-pathway counts:  135809
Chemical-pathway counts:  1285158
Chemical1-Chemical2 counts:  0
Gene1-gene2 counts:  713469
Saving pkl file ...
Chemical count:  32991
Gene count:  51070
Pathway count:  2363




Loading chem2cid dictionary ...
Chemical-gene tiny counts:  1784864
Chemical-gene counts:  1787274
Cene-pathway counts:  135809
Chemical-pathway counts:  1272439
Chemical1-Chemical2 counts:  720155
Gene1-gene2 counts:  713469
Chemical count:  32789
Gene count:  51070
Pathway count:  2363
'''



