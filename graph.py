import pandas as pd
import glob
import os
import numpy as np
import pickle

import multiprocessing
from multiprocessing import Pool

import analysis as an
import  pdb

# basepath='/Users/gracer/Google Drive/HCP_graph/1200/datasets/HCP_PTN1200/graph_analysis/'
# file_dict={'MZ':{'no':{},'ov':{},'ob':{}},'DZ':{'no':{},'ov':{},'ob':{}},'NR':{'no':{},'ov':{},'ob':{}}}
# #read in all the data
# for key, value in file_dict.items():
#     # print(key)
#     for k,v in value.items():
#         # print(k)
#         pat=os.path.join(basepath,key,'matrices',k,'*')
#         for fil in glob.glob(pat):
#             # print(fil)
#             if fil.endswith('_r_matrix.csv'):
#                 sub=fil.split('/')[-1].split('_')[0]
#                 # print(sub)
#                 v[sub]=pd.read_csv(fil, sep=',', index_col=False, header=None)
#
# an.adillyofapickle('/Users/gracer/Google Drive/HCP_graph/1200/datasets',file_dict,'file_dict')
basepath='/Users/gracer/Google Drive/HCP_graph/1200/datasets/'

file_dict=an.onetoughjar(os.path.join(basepath,'tmp','file_dict_11-14-2019_10-07-24'))

list1=[file_dict['MZ'],'MZ','positive',0]
list2=[file_dict['DZ'], 'DZ', 'positive',0]
list3=[file_dict['NR'], 'NR','positive',0]


# test_dict=an.permuatator2(list1)
# an.adillyofapickle('/Users/gracer/Google Drive/HCP_graph/1200/datasets',test_dict,'test_dict')
if __name__ == '__main__':

    pool_size =3
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=an.start_process,
                                )

    [MZ,DZ,NR] = pool.map(an.permuatator2, [list1, list2, list3])
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks


    save_dict={'MZ':MZ,'DZ':DZ,'NR':NR}
    an.adillyofapickle('/Users/gracer/Google Drive/HCP_graph/1200/datasets',save_dict,'save_dict')
