import pandas as pd
import glob
import os
import numpy as np

import pickle

import statistics
# import community
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import community
import analysis as an
import multiprocessing
from multiprocessing import Pool
import glob

basepath='/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets/'
###### Label specific #######
labels = pd.read_csv(os.path.join(basepath,'tmp','mod_labels.csv'), sep=',')
labels.set_index('Index', inplace=True)

note_dict={}
for i,j in labels.iterrows():
    print(i)
    print(j['area'])
    note_dict[i]=j['area']

#### Open data from 3
p = os.path.join(basepath,'tmp','5_summary_dict*')
list_of_files = glob.glob(p) # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)

summary_dict=an.onetoughjar(latest_file)


list1=[summary_dict['NR']['no'],'normal']
list2=[summary_dict['NR']['ov'],'overweight']
list3=[summary_dict['NR']['ob'],'obese']

if __name__ == '__main__':

    pool_size =3
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=an.start_process,
                                )

    [no,ov,ob] = pool.map(an.permuatator4, [list1, list2, list3])
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks


    submod_dict={'no':no,'ov':ov,'ob':ob}
    an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',submod_dict,'6_submod_dict')
    z={}
    for group, dat in submod_dict.items():
        print(group)
        z[group]={}
        for module, data in dat.items():
            print(module)
            z[group][module]=data.join(labels)
    no=pd.concat(list(z['no'].values()))
    ov=pd.concat(list(z['ov'].values()))
    ob=pd.concat(list(z['ob'].values()))
    total=pd.concat([no,ov,ob])
    total.to_csv(os.path.join(basepath,'tmp','submodule_data.csv'), sep=',')
