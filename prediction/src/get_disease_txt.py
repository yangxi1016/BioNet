import csv

disease = "xx"
destination_file = "./xxx-1202-test.txt"
source_data = "../CTD_chem_gene_ixns.csv"
copy_file = "../MAPK1.txt"

with open(copy_file, "r") as copy, open(destination_file, "w") as destination, open(source_data, "r") as source:
    id = ""
    srcs = csv.reader(source)
    for src in srcs:
        if src[0][0]=="#":
            continue
        else:
            if src[3]==disease:
                id = src[4]
                break

    lines = copy.readlines()
    for line in lines:
        line_splited = line.split('\t')
        line_splited[8] = line_splited[8].replace(line_splited[3], disease)
        line_splited[3] = disease
        line_splited[4] = id
        destination.write('\t'.join(line_splited))

# rel_65 = []
# with open("./Relations-65.txt", "r") as r:
#     lines = r.read().splitlines()
#     for line in lines:
#         rel_65.append(line)

# with open(source_data, "r") as source, open(destination_file, "w") as destination:
#     lines = csv.reader(source)
#     for line in lines:
#         if line[0][0]=='#':
#             continue
#         else:
#             ### relation in rel_65?
#             rels = []
#             not_in_65 = 0
#             if '|' in line[9]:
#                 rels = line[9].split('|')
#             else:
#                 rels.append(line[9])
#             for rel in rels:
#                 if rel not in rel_65:
#                     not_in_65 = 1
#                     break
#             if not_in_65:
#                 continue
#             ### disease(gene) is the one?
#             # print(line)
#             if disease in line[3]:
#                 destination.write('\t'.join(line) + '\n')
#             else:
#                 continue

