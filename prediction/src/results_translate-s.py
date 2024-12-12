import pickle
import csv
num_pkl_path = r'../data/adj-pkl/'
name_pkl_path = r'../data/raw-pkl/'
# out_path = r'results/'
result_path = r'../out/'
#######################################################################
# dis = ['all','covid','cn','nsclc','escc']
disease = "covid"
raw_result = 'predict_results_{}.txt'.format(disease)
#######################################################################
rdict_all =  {0:'decreases^expression',1:'affects^cotreatment',2:'increases^activity',3:'decreases^reaction',4:'increases^abundance',5:'increases^expression',6:'affects^binding',7:'increases^transport',8:'increases^chemical synthesis',9:'affects^expression',10:'decreases^methylation',11:'increases^hydrolysis',12:'increases^response to substance',13:'decreases^activity',14:'increases^methylation',15:'increases^reaction',16:'increases^uptake',17:'increases^glutathionylation',18:'increases^sulfation',19:'affects^localization',20:'increases^mutagenesis',21:'decreases^secretion',22:'increases^oxidation',23:'increases^phosphorylation',24:'decreases^response to substance',25:'increases^degradation',26:'affects^reaction',27:'affects^chemical synthesis',28:'affects^splicing',29:'increases^stability',30:'increases^export',31:'affects^phosphorylation',32:'affects^methylation',33:'decreases^cleavage',34:'decreases^abundance',35:'decreases^uptake',36:'affects^folding',37:'decreases^transport',38:'increases^secretion',39:'increases^sumoylation',40:'decreases^degradation',41:'affects^transport',42:'decreases^sumoylation',43:'increases^localization',44:'decreases^phosphorylation',45:'increases^metabolic processing',46:'affects^abundance',47:'decreases^metabolic processing',48:'increases^glucuronidation',49:'decreases^acetylation',50:'increases^hydroxylation',51:'increases^reduction',52:'increases^ubiquitination',53:'affects^export',54:'increases^O-linked glycosylation',55:'decreases^chemical synthesis',56:'affects^activity',57:'affects^response to substance',58:'affects^metabolic processing',59:'increases^import',60:'increases^ADP-ribosylation',61:'increases^cleavage',62:'increases^acetylation',63:'affects^secretion',64:'decreases^localization'}
rdict_covid ={0:'decreases^expression',1:'affects^cotreatment',2:'increases^activity',3:'decreases^reaction',4:'increases^abundance',5:'increases^expression',6:'affects^binding',7:'increases^transport',8:'increases^chemical synthesis',9:'affects^expression',10:'decreases^methylation',11:'increases^hydrolysis',12:'increases^response to substance',13:'decreases^activity',14:'increases^methylation',15:'increases^reaction',16:'increases^uptake',17:'increases^glutathionylation',18:'increases^sulfation',19:'affects^localization',20:'increases^mutagenesis',21:'decreases^secretion',22:'increases^oxidation',23:'increases^phosphorylation',24:'decreases^response to substance',25:'increases^degradation',26:'affects^reaction',27:'affects^chemical synthesis',28:'affects^splicing',29:'increases^stability',30:'increases^export',31:'affects^phosphorylation',32:'affects^methylation',33:'decreases^cleavage',34:'decreases^abundance',35:'decreases^uptake',36:'affects^folding',37:'decreases^transport',38:'increases^secretion',39:'increases^sumoylation',40:'decreases^degradation',41:'affects^transport',42:'decreases^sumoylation',43:'increases^localization',44:'decreases^phosphorylation',45:'increases^metabolic processing',46:'affects^abundance',47:'decreases^metabolic processing',48:'increases^glucuronidation',49:'decreases^acetylation',50:'increases^hydroxylation',51:'increases^reduction',52:'increases^ubiquitination',53:'affects^export',54:'increases^O-linked glycosylation',55:'decreases^chemical synthesis',56:'affects^activity',57:'affects^response to substance',58:'affects^metabolic processing',59:'increases^import',60:'increases^ADP-ribosylation',61:'increases^cleavage',62:'increases^acetylation',63:'affects^secretion',64:'decreases^localization'}
rdict_cn = {0:'decreases^expression',1:'affects^cotreatment',2:'increases^activity',3:'decreases^reaction',4:'increases^abundance',5:'increases^expression',6:'affects^binding',7:'increases^transport',8:'increases^chemical synthesis',9:'affects^expression',10:'decreases^methylation',11:'increases^hydrolysis',12:'increases^response to substance',13:'decreases^activity',14:'increases^methylation',15:'increases^reaction',16:'increases^uptake',17:'increases^glutathionylation',18:'increases^sulfation',19:'affects^localization',20:'increases^mutagenesis',21:'decreases^secretion',22:'increases^oxidation',23:'increases^phosphorylation',24:'decreases^response to substance',25:'increases^degradation',26:'affects^reaction',27:'affects^chemical synthesis',28:'affects^splicing',29:'increases^stability',30:'increases^export',31:'affects^phosphorylation',32:'affects^methylation',33:'decreases^cleavage',34:'decreases^abundance',35:'decreases^uptake',36:'affects^folding',37:'decreases^transport',38:'increases^secretion',39:'increases^sumoylation',40:'decreases^degradation',41:'affects^transport',42:'decreases^sumoylation',43:'increases^localization',44:'decreases^phosphorylation',45:'increases^metabolic processing',46:'affects^abundance',47:'decreases^metabolic processing',48:'increases^glucuronidation',49:'decreases^acetylation',50:'increases^hydroxylation',51:'increases^reduction',52:'increases^ubiquitination',53:'affects^export',54:'increases^O-linked glycosylation',55:'decreases^chemical synthesis',56:'affects^activity',57:'affects^response to substance',58:'affects^metabolic processing',59:'increases^import',60:'increases^ADP-ribosylation',61:'increases^cleavage',62:'increases^acetylation',63:'affects^secretion',64:'decreases^localization'}
rdict_nsclc =  {0:'decreases^expression',1:'affects^cotreatment',2:'increases^activity',3:'decreases^reaction',4:'increases^abundance',5:'increases^expression',6:'affects^binding',7:'increases^transport',8:'increases^chemical synthesis',9:'affects^expression',10:'decreases^methylation',11:'increases^hydrolysis',12:'increases^response to substance',13:'decreases^activity',14:'increases^methylation',15:'increases^reaction',16:'increases^uptake',17:'increases^glutathionylation',18:'increases^sulfation',19:'affects^localization',20:'increases^mutagenesis',21:'decreases^secretion',22:'increases^oxidation',23:'increases^phosphorylation',24:'decreases^response to substance',25:'increases^degradation',26:'affects^reaction',27:'affects^chemical synthesis',28:'affects^splicing',29:'increases^stability',30:'increases^export',31:'affects^phosphorylation',32:'affects^methylation',33:'decreases^cleavage',34:'decreases^abundance',35:'decreases^uptake',36:'affects^folding',37:'decreases^transport',38:'increases^secretion',39:'increases^sumoylation',40:'decreases^degradation',41:'affects^transport',42:'decreases^sumoylation',43:'increases^localization',44:'decreases^phosphorylation',45:'increases^metabolic processing',46:'affects^abundance',47:'decreases^metabolic processing',48:'increases^glucuronidation',49:'decreases^acetylation',50:'increases^hydroxylation',51:'increases^reduction',52:'increases^ubiquitination',53:'affects^export',54:'increases^O-linked glycosylation',55:'decreases^chemical synthesis',56:'affects^activity',57:'affects^response to substance',58:'affects^metabolic processing',59:'increases^import',60:'increases^ADP-ribosylation',61:'increases^cleavage',62:'increases^acetylation',63:'affects^secretion',64:'decreases^localization'}
rdict_escc = {}

rel_dict = {}
with open(num_pkl_path+"rel_num.txt", "r") as rel_num:
    lines = rel_num.readlines()
    for line in lines:
        index = int(line.split("\t")[0])
        rel = line.split("\t")[1]
        rel_dict[index] = rel

def results_sort(file_name):
    with open(result_path + file_name, 'r') as pos_file:
        lines = pos_file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].split('\t')
        lines.sort(key=(lambda x: [x[3], x[1], x[2], x[0]]), reverse=True)

        out_file_name = 'predict_sorted_{}.txt'.format(disease)
        with open(result_path + out_file_name, 'w') as pos_file_sorted:
            for line in lines:
                pos_file_sorted.writelines(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\t' + str(line[3]))
    return out_file_name

def sort_translate(file_name):
    f_chem = open(name_pkl_path+'cid2chem_dict.pkl','rb')
    f_cid = open(num_pkl_path+'num2cid_dict.pkl','rb')
    f_gene = open(name_pkl_path+'gid2gene_dict.pkl','rb')
    f_gid = open(num_pkl_path+'num2gid_dict.pkl','rb')
    # f_pathway = open(pkl_path+'pid2pathway_dict.pkl','rb')
    # f_pid = open(pkl_path+'num2pid_dict.pkl','rb')

    f_cgi = open(result_path + file_name,'r')

    num2cid_dict = pickle.load(f_cid)
    cid2chem_dict = pickle.load(f_chem)
    num2gid_dict = pickle.load(f_gid)
    gid2gene_dict = pickle.load(f_gene)
    # num2pid_dict = pickle.load(f_pid)
    # pid2pathway_dict = pickle.load(f_pathway)
    # if disease == 'all':
    #     rdict = rdict_all
    # if disease == 'covid':
    #     rdict = rdict_covid
    # if disease == 'cn':
    #     rdict = rdict_cn
    # if disease == 'nsclc':
    #     rdict = rdict_nsclc
    # if disease == 'escc':
    #     rdict = rdict_escc
    out_file_name = 'predict_translated_{}.txt'.format(disease)
    with open(result_path + out_file_name,'w') as f:
        lines = f_cgi.readlines()
        for line in lines:
            lst_line = line.split('\t')
            rel = [lst_line[0][2], lst_line[0][5], lst_line[0][9:-1]]
        
            if rel[0]=='0' and rel[1]=='1':
                cnum = lst_line[1]
                gnum = lst_line[2]
            elif rel[0]=='1' and rel[1]=='0':
                cnum = lst_line[2]
                gnum = lst_line[1]
            score = lst_line[3]

            cid = num2cid_dict[int(cnum)]
            gid = num2gid_dict[int(gnum)]
            if gid in gid2gene_dict.keys() and cid in cid2chem_dict.keys():
                chem = cid2chem_dict[cid]
                gene = gid2gene_dict[gid]
            else:
                continue
        
            f.write('{}\t {}\t {}\t {}'.format(rel_dict[int(rel[2])], chem, gene, str(score)))
    return out_file_name

if __name__ == '__main__':
    result_sort = results_sort(raw_result)
    result_translate = sort_translate(result_sort)
    print("The result is in "+result_path+result_translate)

        



        

    
