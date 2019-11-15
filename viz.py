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

import analysis as an


basepath='/Users/gracer/Google Drive/HCP_graph/1200/datasets/'
summary_dict=an.onetoughjar(os.path.join(basepath,'tmp','summary_dict_11-14-2019_04-33-33'))
# dict_keys(['mean_FC', 'graphs', 'clustering_coeff', 'btn_centrality', 'PC', 'modules'])
# summary_dict['NR']['no']['modules']['partition']
summary_dict['NR']['no']['graphs'].nodes(data=True)

an.grace_graph(summary_dict['NR']['no']['graphs'], 'clustering', 'Normal weight', 1)
an.grace_graph(summary_dict['NR']['ov']['graphs'], 'clustering', 'Overweight', 1)
an.grace_graph(summary_dict['NR']['ob']['graphs'], 'clustering', 'Obese', 1)

for key, value in summary_dict.items():
    for k,v in value.items():
        # community.induced_graph(partition dictionary, graph)
         comm_graph = community.induced_graph(v['modules']['partition'], v['graphs'])
         v.update(comm_graph = comm_graph)

an.module_fig(summary_dict['NR']['no']['comm_graphs'])
