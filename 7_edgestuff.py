import pandas as pd
import os
import numpy as np
import pickle
import community
import networkx as nx

import analysis as an

basepath='/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets/'

# this is a dictionary of all subjects
latest_file=an.find_latest(os.path.join(basepath,'tmp'),'3_*')
save_dict=an.onetoughjar(latest_file)

# this is the graph
latest_file=an.find_latest(os.path.join(basepath,'tmp'),'5_*')
summary_dict=an.onetoughjar(latest_file)

for group, values in save_dict['NR'].items():
    print(group)
    values['modules'] = summary_dict[group]['modules']['partition']
    for sub, graph in values['graphs'].items():
        print(sub)
        G = graph
        modules = summary_dict[group]['modules']['partition']
        nx.set_node_attributes(G, modules, 'modules')

# Make community graph
for group,v in save_dict['NR'].items():
    print(group)
    v.update(comm_graph = {})
    for sub, data in v['graphs'].items():
        print(sub)
        comm_graph = community.induced_graph(v['modules'], data)
        v['comm_graph'][sub] = comm_graph



tmp_dict = {'no':{}, 'ov': {}, 'ob': {}}
for group, dat in save_dict['NR'].items():
    print(group)
    x=an.cal_edges(dat['comm_graph'], basepath, group)
    tmp_dict[group]=pd.concat(list(x.values()))
sub_comm_edge_df=pd.concat(list(tmp_dict.values()))
sub_comm_edge_df.to_csv(os.path.join(basepath,'tmp','sub_comm_edge_data.csv'), sep=',')
an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',save_dict['NR'],'10_subedge_dict')