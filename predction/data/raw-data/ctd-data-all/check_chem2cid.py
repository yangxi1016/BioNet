

writer = open("chem2cid_check.txt", "w")


cid_dict = {}
reader = open("chem2cid.txt", "r").read().strip().split("\n")
for line in reader:
    terms = line.split("\t")
    chem = terms[0]
    cid = terms[1]
    if cid not in cid_dict.keys():
        cid_dict[cid] = 1
    else:
        cid_dict[cid] = cid_dict[cid] + 1

    cid_new = cid + "#" + str(cid_dict[cid])
    writer.write(chem + "\t" + cid_new + "\n")

