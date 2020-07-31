import pickle
import analysis as an
import os
import multiprocessing
from multiprocessing import Pool
import glob
basepath='/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets/'
#Load data from pickle if needed
p = os.path.join(basepath,'tmp','*file_dict*')
list_of_files = glob.glob(p) # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)

file_dict=an.onetoughjar(latest_file)

mean_dict={'MZ':{'no':{},'ov':{},'ob':{}},'DZ':{'no':{},'ov':{},'ob':{}},'NR':{'no':{},'ov':{},'ob':{}}}
for key, value in file_dict.items():
    print(key)
    for k,v in value.items():
        print(k)
        mean_dict[key][k]=an.make_total_graphs(v)

an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',mean_dict,'4_mean_dict')

list1=[mean_dict['MZ'],'MZ','positive',0]
list2=[mean_dict['DZ'], 'DZ', 'positive',0]
list3=[mean_dict['NR'], 'NR','positive',0]


# test_dict=an.permuatator2(list1)
# an.adillyofapickle('/Users/gracer/Google Drive/HCP_graph/1200/datasets',test_dict,'test_dict')
if __name__ == '__main__':

    pool_size =3
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=an.start_process,
                                )

    [MZ,DZ,NR] = pool.map(an.permuatator3, [list1, list2, list3])
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks


    summary_dict={'MZ':MZ,'DZ':DZ,'NR':NR}
    an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',summary_dict,'5_summary_dict')
