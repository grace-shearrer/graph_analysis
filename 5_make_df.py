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
