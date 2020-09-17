import pandas as pd
import glob
import os
import numpy as np
import networkx as nx
import pickle

import statistics
# import community
import community
import analysis as an
import multiprocessing
from multiprocessing import Pool
import glob

basepath='/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets/'
#### Open data from 3
p = os.path.join(basepath,'tmp','5_summary_dict*')
list_of_files = glob.glob(p) # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)

update_dict=an.onetoughjar(latest_file)
# update_dict = {}
# for key, value in summary_dict.items():
#     print(key)
#     update_dict[key] = {**value[0], **value[1]}

# get label stuff
labels = pd.read_csv(os.path.join(basepath,'tmp','mod_labels.csv'), sep=',')
labels.set_index('Index', inplace=True)
label_dict = labels.to_dict('index')


list1=[update_dict['no'],'normal']
list2=[update_dict['ov'],'overweight']
list3=[update_dict['ob'],'obese']

if __name__ == '__main__':

    pool_size =3
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=an.start_process,
                                )

    [no,ov,ob] = pool.map(an.permuatator5, [list1, list2, list3])
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks


    subgraph_dict={'no':no,'ov':ov,'ob':ob}
    #apply label stuff
    for group, dat in subgraph_dict.items():
        print(group)
        for mod, graph in dat.items():
            nx.set_node_attributes(graph, label_dict, 'labels')


    an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',subgraph_dict,'7_subgraph_dict')
