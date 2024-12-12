import csv

def fresh_csv(file):
    with open(file, "r", encoding='utf-8') as source, \
         open(''.join(file.split("_raw")), "w", newline='', encoding='utf-8') as destination:
        print(file.split('/')[-1])
        src = csv.reader(source)
        dest = csv.writer(destination)
        count = 0
        for row in src:
            if row[0][0]=="#":
                continue
            else:
                if count<10:
                    print(row)
                    count += 1
                dest.writerow(row)

if __name__ == '__main__':
    fresh_csv("../data/raw-data/ctd-data-all/CTD_chem_gene_ixns_raw.csv")
    fresh_csv("../data/raw-data/ctd-data-all/CTD_chem_pathways_enriched_raw.csv")
    fresh_csv("../data/raw-data/ctd-data-all/CTD_genes_pathways_raw.csv")
    fresh_csv("../data/raw-data/ctd-data-all/CTD_chemicals_diseases_raw.csv")
    fresh_csv("../data/raw-data/ctd-data-all/CTD_diseases_pathways_raw.csv")
    fresh_csv("../data/raw-data/ctd-data-all/CTD_genes_diseases_raw.csv")
    fresh_csv("../data/raw-data/ctd-data-all/CTD_chemicals_raw.csv")