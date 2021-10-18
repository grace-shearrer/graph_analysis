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
    values['modules'] = summary_dict[group][1]['modules']['partition']
    for sub, graph in values['graphs'].items():
        print(sub)
        G = graph
        modules = summary_dict[group][1]['modules']['partition']
        nx.set_node_attributes(G, modules, 'modules')

# Make community graph
for group,v in save_dict['NR'].items():
    print(group)
    v.update(comm_graph = {})
    for sub, data in v['graphs'].items():
        print(sub)
        comm_graph = community.induced_graph(v['modules'], data)
        v['comm_graph'][sub] = comm_graph


#tmp_dict = {'no':{}, 'ov': {}, 'ob': {}}
#for group, dat in save_dict['NR'].items():
#    print(group)
#    x=an.cal_edges(dat['comm_graph'], basepath, group)
#    tmp_dict[group]=pd.concat(list(x.values()))
#sub_comm_edge_df=pd.concat(list(tmp_dict.values()))
#sub_comm_edge_df.to_csv(os.path.join(basepath,'tmp','sub_comm_edge_data.csv'), sep=',')
#an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',save_dict['NR'],'10_subedge_dict')

for group, dat in save_dict['NR'].items():
    print(group)
    x=an.cal_edges(dat['graphs'], basepath, group)
    for key, value in x.items():
        value['subject'] = key
    save_dict['NR'][group]['edges']=pd.concat(list(x.values()))

liist = [save_dict['NR']['no']['edges'], save_dict['NR']['ov']['edges'], save_dict['NR']['ob']['edges']]
sub_edge_df=pd.concat(liist)
sub_edge_df = sub_edge_df[['weight', 'z_weight', 'subject']].groupby(['subject']).mean()
sub_edge_df['Subject'] = sub_edge_df.index
sub_edge_df.reset_index(drop=True)
sub_edge_df.to_csv(os.path.join(basepath,'tmp','sub_edge_data.csv'), sep=',')
an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',save_dict['NR'],'10_subedge_dict')


#making a dataframe
subedge_dict = save_dict['NR']
_dfs=[]
for group, data in subedge_dict.items():
    print(group)
    for sub, dat in data['graphs'].items():
        print(sub)
        tmp = pd.DataFrame.from_dict(dict(dat.nodes(data =True)), orient='index')
        tmp['Subject'] = sub
        tmp['group'] = group
        tmp['IC'] = tmp.index
        tmp['IC'] = 'IC_' + tmp['IC'].astype(str)
        _dfs.append(tmp)
total = pd.concat(_dfs)
latest_file=an.find_latest(os.path.join(basepath,'tmp'),'demo_dict*')
print(latest_file)
demo_dict=an.onetoughjar(latest_file)
demo_df = pd.DataFrame.from_dict(demo_dict['NR'], orient='index')
demo_df['Subject'] = demo_df.index
demo_df.reset_index(drop=True)
demo_df["Subject"]=pd.to_numeric(demo_df["Subject"])
total["Subject"]=pd.to_numeric(total["Subject"])
sub_edge_df["Subject"]=pd.to_numeric(sub_edge_df["Subject"])
complete_df=pd.merge(demo_df, total, on="Subject")
complete_df=pd.merge(complete_df, sub_edge_df, on="Subject")
complete_df.to_csv(os.path.join(basepath,'tmp','complete_data.csv'), sep=',')
an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',complete_df,'11_complete_df')
