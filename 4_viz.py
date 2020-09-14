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
import glob
import networkx as nx

basepath='/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets/'
latest_file2=an.find_latest(os.path.join(basepath,'tmp'),'5_*')
summary_dict=an.onetoughjar(latest_file2)
update_dict = {}
for key, value in summary_dict.items():
    print(key)
    update_dict[key] = {**value[0], **value[1]}

latest_file=an.find_latest(os.path.join(basepath,'tmp'),'6_*')
submod_dict=an.onetoughjar(latest_file)



# Make community graph
for k,v in update_dict.items():
    # community.induced_graph(partition dictionary, graph)
     comm_graph = community.induced_graph(v['modules']['partition'], v['graphs'])
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

# Set viz Parameters
aes_dict={'no':{},
          'ov':{},
          'ob':{}}
for group, stuff in update_dict.items():
    print(group)
    G=stuff['comm_graph']
    aes_dict[group]=an.aesthetics(G,15000,100, 'sans-serif', 'Bold', 'z_edge', (80,50), 1)


an.module_fig(update_dict['no']['comm_graph'], 'Average BMI z-scored', basepath, aes_dict['no'])
an.module_fig(update_dict['ov']['comm_graph'], 'High BMI z-scored', basepath, aes_dict['ov'])
an.module_fig(update_dict['ob']['comm_graph'], 'Very High BMI z-scored', basepath, aes_dict['ob'])
