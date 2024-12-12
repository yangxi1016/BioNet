
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


# chem-gene
def generate_chem_gene():
    #####替换测试文件########
    f = open(r"G0S2-1118-test.txt", "r")
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


    pickle.dump(list(set(chem_gene_list)), open("../data/pkl-raw/chem_gene_list_G0S2.pkl", "wb"))
    print("Chemical-gene counts: ", len(set(chem_gene_list)))



def save():
    print("Saving pkl file ...")





chem2cid_query()
generate_chem_gene()
save()

print("Chemical count: ", len(cid2chem_dict.keys()))
print("Gene count: ", len(gid2gene_dict.keys()))



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



