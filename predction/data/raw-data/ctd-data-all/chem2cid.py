import csv
import pickle
import pubchempy as pcp
import sys

# chem_dict = {}
# csv_read1 = csv.reader(open("CTD_chem_gene_ixns.csv", "r"))
# for line in csv_read1:
#     chem = line[0]
#     chem_dict[chem] = len(chem_dict.keys())
# print("chem-gene done!")
#
# csv_read2 = csv.reader(open("CTD_chem_pathways_enriched.csv", "r"))
# for line in csv_read2:
#     chem = line[0]
#     chem_dict[chem] = len(chem_dict.keys())
# print("chem-pathway done!")
#
# print(len(chem_dict.keys()))
# pickle.dump(chem_dict, open("chem_dict.pkl", "wb"))

#
# writer =  open("chem2cid_dict", "a")
# writer.write("Actinoid Series Elements\tFID00000016\n")
# sys.exit()

# 11-Hydroxycorticosteroids FID00000001
# 17-Hydroxycorticosteroids FID00000002
# 17-Ketosteroids   FID00000003
# 2-Pyridinylmethylsulfinylbenzimidazoles   FID00000004
# Abietanes FID00000005
# Acetic Anhydrides FID00000006
# Acetoacetates FID00000007
# Acetonitriles FID00000008
# Acidic Glycosphingolipids FID00000009
# Acids, Acyclic    FID00000010
# Acids, Aldehydic  FID00000011
# Acids, Carbocyclic    FID00000012
# Acids, Heterocyclic   FID00000013
# Acids, Noncarboxylic  FID00000014
# Acridones FID00000015
# Actinoid Series Elements  FID00000016

flag = 17
while True:
    try:

        chem2cid_dict = {}
        reader = open("chem2cid_dict.txt", "r").read().strip().split("\n")
        for line in reader:
            # print(line)
            terms = line.split("\t")
            if terms[0] not in chem2cid_dict:
                chem = terms[0]
                cid = terms[1]
                chem2cid_dict[chem] = cid
                print(line)
        # print(len(chem2cid_dict.keys()))

        chem_dict = {}
        chem_dict = pickle.load(open("chem_dict.pkl", "rb"))
        # writer =  open("chem2cid_dict", "a")
        count = 0
        for k, v in chem_dict.items():
            chem = k
            if chem not in chem2cid_dict.keys():
                writer = open("chem2cid_dict.txt", "a")
                print(chem)
                compounds = pcp.get_compounds(chem, 'name')
                if len(compounds):
                    cid = "CID" + str(compounds[0].cid).zfill(8)
                else:
                    substances = pcp.get_substances(chem, 'name')
                    if len(substances) == 0:
                        cid = "FID" + str(flag).zfill(8)
                        flag += 1
                        print(cid + "--------")
                    else:
                        cid = "SID" + str(substances[0].sid).zfill(8)
                chem2cid_dict[chem] = cid
                writer.write(chem + "\t" + cid + "\n")

            count += 1
            if count%10==0: print(str(len(chem_dict.keys())) + "-" + str(count))
    except:
        print("error")



