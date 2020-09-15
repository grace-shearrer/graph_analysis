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
import networkx as nx

basepath='/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets/'
###### Label specific #######
labels = pd.read_csv(os.path.join(basepath,'tmp','mod_labels.csv'), sep=',')
labels.set_index('Index', inplace=True)

note_dict={}
for i,j in labels.iterrows():
    print(i)
    print(j['area'])
    note_dict[i]=j['area']

p='/Users/gracer/Google Drive/HCP/HCP_graph/1200/'
atlas=pd.read_csv(os.path.join(p,'brains','atlas.csv'), sep=',')
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

nx.set_node_attributes(update_dict['no']['modules']['graph'], note_dict, 'area')
nx.set_node_attributes(update_dict['ov']['modules']['graph'], note_dict, 'area')
nx.set_node_attributes(update_dict['ob']['modules']['graph'], note_dict, 'area')

# Make community graph
for k,v in update_dict.items():
    # community.induced_graph(partition dictionary, graph)
     comm_graph = community.induced_graph(v['modules']['partition'], v['modules']['graph'])
     v.update(comm_graph = comm_graph)

# Normalize the edges
edges = {}

for group, stuff in update_dict.items():
    print(group)
    _df = nx.to_pandas_edgelist(stuff['comm_graph'])
    _df.loc[(_df['source'] == _df['target']), 'weight'] = 0
    _df['group']=group
    edges[group]=_df
edge_df=pd.concat(list(edges.values()))
edge_df['z_weight']=an.zscore(edge_df['weight'])

# Set normalized edge as an get_edge_attributes
for k,v in update_dict.items():
    test=edge_df[edge_df['group']==k]
    keyz = list(zip(test['source'],test['target']))
    values=test['z_weight']
    up_dict={}
    for i in range(len(keyz)):
        up_dict[keyz[i]]={'z_edge':values[i]}
    nx.set_edge_attributes(v['comm_graph'], up_dict)


an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',update_dict,'5_summary_dict')



list1=[update_dict['no'],'normal']
list2=[update_dict['ov'],'overweight']
list3=[update_dict['ob'],'obese']

if __name__ == '__main__':

    pool_size =3
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=an.start_process,
                                )

    [no,ov,ob] = pool.map(an.permuatator4, [list1, list2, list3])
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks


    submod_dict={'no':no,'ov':ov,'ob':ob}
    for group, item in submod_dict.items():
        print(group)
        for mod, df in item.items():
            print(mod)
            df['module'] = mod
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
    total.rename(columns={'Unnamed: 2': 'ROI'}, inplace=True)
    new_df = total.merge(atlas[['X','Y','Z','ROI']], left_on='ROI', right_on = 'ROI', how='left')
    new_df.to_csv(os.path.join(basepath,'tmp','submodule_data.csv'), sep=',')
    an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',submod_dict,'6_submod_dict')
