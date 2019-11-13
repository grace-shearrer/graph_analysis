# This will create all the correlation matrices you will need for the analyses
import pandas as pd
import glob
import os
import numpy as np
from zipfile import ZipFile
import tarfile
import pickle

from functools import partial
from itertools import repeat
import multiprocessing
from multiprocessing import Pool

import analysis as an
basepath='/Users/gracer/Google Drive/HCP_graph/1200/datasets'
dim='100'

# Cols of interest
myvars=['Subject', 'Age_in_Yrs', 'ZygosityGT', 'Race', 'Ethnicity', 'BMI', 'HbA1C', 'Hypothyroidism',
 'Hyperthyroidism',
 'OtherEndocrn_Prob', 'Mother_ID','Father_ID']
exlcude=['Hypothyroidism', 'Hyperthyroidism','OtherEndocrn_Prob']

# Load data
df = pd.read_csv('/Users/gracer/Google Drive/HCP_graph/1200/datasets/RESTRICTED_gshearrer_4_19_2018_11_33_34.csv', sep=',')
data_dict=an.data_cleaner(df, myvars, exlcude)
an.adillyofapickle(basepath, data_dict,'data_dict')

for key, value in data_dict.items():
    print(key)
    data_dict[key]=an.BMI_calc(value)

for key, value in data_dict.items():
    value.to_csv('/Users/gracer/Google Drive/HCP_graph/1200/datasets/%s.csv'%key, sep=',')

# find the data in the zip
for key, value in data_dict.items():
    list_int = an.sub_interest(dim, value)
    an.getit(dim, basepath, list_int, key)


datapath = os.path.join(basepath,'HCP_PTN1200','graph_analysis','%s'%type,'node_timeseries','3T_HCP1200_MSMAll_d%s_ts2'%dim)


demo_dict={'MZ':{}, 'DZ':{}, 'NT':{}, 'NR':{}}
for key, value in data_dict.items():
    value=value.set_index('Subject')
    demo_dict[key]=value.to_dict('index')

an.adillyofapickle(basepath, demo_dict,'demo_dict')

file_dict={'MZ':{}, 'DZ':{}, 'NT':{}, 'NR':{}}
# print(demo_dict)
for key, value in demo_dict.items():
    print(key)
    ovob_dict={'no':{},'ov':{},'ob':{}}
    file_dict[key]=ovob_dict
    for k,v in value.items():
        print(k)
        weight = v['ov_ob']
        print(weight)
        datapath = os.path.join(basepath,'HCP_PTN1200','graph_analysis','%s'%key,'node_timeseries','3T_HCP1200_MSMAll_d%s_ts2'%dim)
        file_dict[key][weight][k]= os.path.join(datapath,'%s.txt'%k)


list1=[file_dict['MZ'],'MZ']
list2=[file_dict['DZ'], 'DZ']
list3=[file_dict['NR'], 'NR']



if __name__ == '__main__':

    pool_size =3
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=an.start_process,
                                )

    [MZ,DZ,NR] = pool.map(an.permuatator1, [list1, list2, list3])
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
