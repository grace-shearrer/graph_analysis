
import pandas as pd
import glob
import os
import numpy as np
from zipfile import ZipFile
import tarfile
import pickle
from datetime import datetime
from time import time
import statistics
import community
import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


#from https://wiki.biac.duke.edu/biac:analysis:resting_pipeline
import numpy.ma
import nibabel
from scipy import signal
import sys, subprocess
import string, random
import re
import networkx as nx
from optparse import OptionParser, OptionGroup
import logging
import math
from scipy import ndimage as nd
import pdb
import multiprocessing
from multiprocessing import Pool

datefmt='%m-%d-%Y_%I-%M-%S'
logging.basicConfig(format='%(asctime)s %(message)s ', datefmt=datefmt, level=logging.INFO)
def start_process():
    print ('Starting', multiprocessing.current_process().name)

def permuatator1(liist):
    Dict=liist[0]
    key=liist[1]
    print(key)
    ovob_dict={'no':{},'ov':{},'ob':{}}
    for subkey, subval in Dict.items():
        print(subkey)
        outpath=os.path.join('/Users/gracer/Google Drive/HCP_graph/1200/datasets/HCP_PTN1200/graph_analysis',key,'matrices',subkey)
        if os.path.exists(outpath):
            print('got it')
        else:
            os.makedirs(outpath)
        for k, v in subval.items():
            print(k)
            checkpath=os.path.join('/Users/gracer/Google Drive/HCP_graph/1200/datasets/HCP_PTN1200/graph_analysis',key,'node_timeseries',
                                   '3T_HCP1200_MSMAll_d100_ts2','%s.txt'%k)
            print(checkpath)
            if os.path.exists(checkpath):
                ovob_dict[subkey][k]=duke_corr(v, outpath)
            else:
                print('this is missing %s'%k)
                with open(os.path.join('/Users/gracer/Google Drive/HCP_graph/1200/datasets','tmp','%s_%s_missing.txt'%(subkey,k)), 'w') as miss:
                    miss.write('%s\n'%k)
        miss.close()
    return(ovob_dict)

def duke_corr(corrtxt, outpath):
    sub=corrtxt.split('/')[-1].split('.')[0]
    logging.info('starting correlation')
    rmat = os.path.join(outpath,'%s_r_matrix.nii.gz'%sub)
    rtxt = os.path.join(outpath,'%s_r_matrix.csv'%sub)
    zmat = os.path.join(outpath,'%s_zr_matrix.nii.gz'%sub)
    ztxt = os.path.join(outpath,'%s_zr_matrix.csv'%sub)
    maskname = os.path.join(outpath,'%s_mask_matrix.nii.gz'%sub)

    print(corrtxt)
    timeseries = np.loadtxt(corrtxt,unpack=True)
    myres = np.corrcoef(timeseries)
    myres = np.nan_to_num(myres)

    zrmaps = 0.5*np.log((1+myres)/(1-myres))
    #find the inf vals on diagonal
    infs = (zrmaps == np.inf).nonzero()

    #replace the infs with 0
    for idx in range(len(infs[0])):
        zrmaps[infs[0][idx]][infs[1][idx]] = 0

    nibabel.save(nibabel.Nifti1Image(myres,None) ,rmat)
    nibabel.save(nibabel.Nifti1Image(zrmaps,None) ,zmat)
    np.savetxt(ztxt,zrmaps,fmt='%f',delimiter=',')
    np.savetxt(rtxt,myres,fmt='%f',delimiter=',')

    #create a mask for higher level, include everything below diagonal
    mask = np.zeros_like(myres)
    maskx,masky = mask.shape

    for idx in range(maskx):
        for idy in range(masky):
            if idx > idy:
                mask[idx][idy] = 1

    nibabel.save(nibabel.Nifti1Image(mask,None) ,maskname)

    #check for the resulting files
    for fname in [rmat, zmat, maskname, ztxt, rtxt]:
        if os.path.isfile( fname ):
            logging.info('correlation matrix finished : ' + fname)
        else:
            logging.info('correlation failed')
            raise SystemExit()
    return(myres)


def adillyofapickle(basepath,dic, name):
    st = datetime.fromtimestamp(time()).strftime(datefmt)
    if os.path.exists(os.path.join(basepath,'tmp')):
        print('already have tmp')
    else:
        os.makedirs(os.path.join(basepath,'tmp'))
    pickle.dump(dic, open(os.path.join(basepath,'tmp','%s_%s'%(name,st)), 'wb'), protocol=4)

def data_cleaner(df, vars, exclude):
         # df expecting a csv file
         # vars expecting a list of variables
         # exclude is a list of variables in which a 1 value is excluded
     df=df[vars]
     for item in exclude:
         df=df.loc[df[item] != 1]
     df=df.loc[df['BMI'] > 18]
     df=df.dropna(subset=['BMI'])
     df_mz=df.loc[df['ZygosityGT'] == 'MZ']
     df_dz=df.loc[df['ZygosityGT'] == 'DZ']
     df_nt=df.loc[df['ZygosityGT'] == ' ']
     df_unt=df_nt.drop_duplicates(['Mother_ID','Father_ID'])
     data_dict={'MZ':df_mz, 'DZ':df_dz, 'NT':df_nt, 'NR':df_unt}
     return(data_dict)

def BMI_calc(df):
    # df must have a column named BMI
    df.loc[df['BMI'] <25 , 'ov_ob'] = 'no'
    df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'ov_ob'] = 'ov'
    df.loc[df['BMI'] >= 30, 'ov_ob'] = 'ob'
    return(df)

def tar_heel(X, liist, out):
    tar = tarfile.open(X)
    tar.extractall(out, members=[m for m in tar.getmembers() if m.name in liist])
    tar.close()

def sub_interest(dim, df):
    # import os
    list_int = [os.path.join('node_timeseries/3T_HCP1200_MSMAll_d%s_ts2/'%dim,'%s.txt'%x) for x in df['Subject']]
    return(list_int)


def getit(dim, basepath, liist, type):
    tarp=os.path.join(basepath,'HCP_PTN1200','NodeTimeseries_3T_HCP1200_MSMAll_ICAd%s_ts2.tar.gz'%dim)
    zarp=os.path.join(basepath,'HCP_PTN1200','graph_analysis','%s'%type,'node_timeseries', '3T_HCP1200_MSMAll_d%s_ts2'%dim)
    narp= os.path.join(basepath,'HCP_PTN1200','graph_analysis', '%s'%type)
    if os.path.exists(zarp):
        print('this is already opened, check the dim')
    elif os.path.exists(tarp):
        print('the tar exists, need to unzip')
        tar_heel(tarp, liist, narp)
    else:
        print('the tar file needs to be unzipped')
        for (dirpath, dirnames, filenames) in os.walk(basepath):
              for filename in filenames:
                  if filename == 'HCP1200_Parcellation_Timeseries_Netmats.zip':
                      tmppath=os.sep.join([dirpath, filename])
                      with ZipFile(tmppath, 'r') as zipObj:
                         # Get a list of all archived file names from the zip
                         listOfFileNames = zipObj.namelist()
                         for name in listOfFileNames:
                             print(name)
                             if name == "HCP_PTN1200/NodeTimeseries_3T_HCP1200_MSMAll_ICAd%s_ts2.tar.gz"%dim:
                                 zipObj.extract(name, os.path.join(basepath,'datasets'))
                                 tar_heel(tarp, liist, narp)
def onetoughjar(path2dic):
    with open(path2dic, 'rb') as pickle_file:
        try:
            while True:
                output = pickle.load(pickle_file)
        except EOFError:
            pass
    return(output)

def loaded(pat):
    for (dirpath, dirnames, filenames) in os.walk(pat):
        print('this is the dirpath %s'%dirpath)
        print('this is the dirnames %s'%dirnames)
        print('this is the filenames %s'%filenames)
          # for filename in filenames:
              # if filename == 'HCP1200_Parcellation_Timeseries_Netmats.zip':
              #     tmppath=os.sep.join([dirpath, filename])

def participation_award(Gs):
    allPC={}
    parts={}
    for keys, values in Gs.items():
        print(keys)
        g=threshold(values,'positive',0)
        (partition,vals,graph)=ges(g)
        cor_mat=nx.to_numpy_matrix(values)
        vals=np.array(vals)
        PC=participation_coef(W=cor_mat, ci=vals, degree="undirected")
        allPC[keys]=PC
        parts[keys]=(partition)
    totes={'PC':allPC, 'Parts':parts}
    return(allPC)

def make_graphs(dict_o_data, direction, min_cor):
    FC_dict={}
    graph_dict={}
    partition_dict={}
    clustering_dict ={}
    centrality_dict ={}
    PC_dict={}
    partition_dict={}
    for key, values in dict_o_data.items():
        # print(key)
        ########################################
        cor_matrix = np.asmatrix(values)
        # print(cor_matrix.shape)
        x=abs(cor_matrix)
        mu=x.mean()
        # print(mu)
        ########################################
        G = nx.from_numpy_matrix(cor_matrix)
        tG = threshold(G, direction, min_cor)
        ########################################
        (partition,vals,graph)=ges(tG)
        partition_dict[key]=(partition,vals)
        vals=np.array(vals)
        PC=participation_coef(W=cor_matrix, ci=vals, degree="undirected")
        pc_dict={}
        for i in range(len(PC)):
            pc_dict[i]=PC[i]
        PC_dict[key]=PC
        ########################################
        FC_dict[key]=mu
        ########################################
        clustering = nx.clustering(tG, weight=True)
        clustering_dict[key]=clustering
        ########################################
        centrality = nx.betweenness_centrality(tG, weight=True)
        centrality_dict[key]=centrality
        ########################################
        nx.set_node_attributes(G, centrality, 'centrality')
        nx.set_node_attributes(G, clustering, 'clustering')
        nx.set_node_attributes(G, pc_dict, 'PC')
        graph_dict[key]=G
        ########################################
    return({'mean_FC':FC_dict, 'graphs':graph_dict, 'clustering_coeff':clustering_dict, 'btn_centrality':centrality_dict, 'PC':PC_dict})


def threshold(G, corr_direction, min_correlation):
    ##Creates a copy of the graph
    H = G.copy()
    ##Checks all the edges and removes some based on corr_direction
    for stock1, stock2, weight in list(G.edges(data=True)):
        ##if we only want to see the positive correlations we then delete the edges with weight smaller than 0
        if corr_direction == "positive":
            ####it adds a minimum value for correlation.
            ####If correlation weaker than the min, then it deletes the edge
            # print(weight["weight"])
            # pdb.set_trace()
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        ##this part runs if the corr_direction is negative and removes edges with weights equal or largen than 0
        else:
            ####it adds a minimum value for correlation.
            ####If correlation weaker than the min, then it deletes the edge
            # print(weight["weight"])
            # pdb.set_trace()
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
    return(H)


def make_total_graphs(dict_o_data):
    mylist=[]
    for key, val_list in dict_o_data.items():
        for i in val_list:
            cor_matrix = np.asarray(i)
            mylist.append(cor_matrix)
    x=np.stack(mylist, axis=2)
    mu=np.median(x, axis=(2))
    return(mu)


def ges(mu):
    partition = community.best_partition(mu)
    vals = list(partition.values())
    nx.set_node_attributes(mu, partition, 'modules')
    return((partition,vals, mu))

def jenny_graph(graph):
    edges,weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
    nodes, color = zip(*nx.get_node_attributes(graph,'modules').items()) #if your modules are named different change here
    nodes, positions = zip(*nx.get_node_attributes(graph,'modules').items())
    #positions
    positions=nx.circular_layout(graph) #this is defining a circluar graph, if you want a different one you change the circular part of this line

    #Figure size
    plt.figure(figsize=(40,25))


    #draws nodes
    color = np.array(color)
    nColormap=plt.cm.Spectral #check here if you want different colors https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    cM=color.max()
    cm=color.min()
    y=nx.draw_networkx_nodes(graph,positions,
                           node_color=color,
                           node_size=40,
                           alpha=0.8,
                           cmap= nColormap,
                           vmin=cm ,vmax=cM)

    #Styling for labels
    nx.draw_networkx_labels(graph, positions, font_size=10,
                            font_family='sans-serif', fontweight = 'bold')


    #draw edges
    weights=np.array(weights)
    eColormap=plt.cm.bwr #check here if you want different colors https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    wt=weights*5
    M=wt.max()
    m=wt.min()
    x=nx.draw_networkx_edges(graph, positions, edge_list=edges, style='solid', width = wt, edge_color = wt,
                           cmap=eColormap,
                           edge_vmin=m,
                           edge_vmax=M)

    #format the colorbar
    node_bar=plt.colorbar(y)
    edge_bar=plt.colorbar(x)

    node_bar.set_label('Modularity',fontsize = 25)
    edge_bar.set_label('Strength of edge weight',fontsize = 25)

    plt.axis('off')
    plt.title("Modularity and Edge Weights of Average Graph", fontsize = 30)
    #plt.savefig(os.path.join(basepath,"betaseries_bevel/5_analysis/modularity_circle_reward.png", format="PNG")
    plt.show()

def module_fig(G, Type):
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    #nodes, size = zip(*nx.get_node_attributes(G,'clustering').items())


    positions=nx.circular_layout(G)
    plt.figure(figsize=(25,20))

    color = np.array(list(G.nodes))
    nColormap=plt.cm.Spectral #check here if you want different colors https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    cM=color.max()
    cm=color.min()
    nx.draw_networkx_nodes(G,positions,
                           node_color=color,
                           #node_size=size,
                           alpha=1.0,
                           cmap= 'Spectral',
                           vmin=cm,vmax=cM )

    #Styling for labels
    nx.draw_networkx_labels(G, positions, font_size=8, font_family='sans-serif')
    wt=np.array(weights)/5
    x=nx.draw_networkx_edges(G, positions, edge_list=edges,style='solid', width = wt, edge_color = weights)

    edge_bar=plt.colorbar(x)
    edge_bar.set_label('Strength of edge weight',fontsize = 25)

    plt.title("Module Connectivity Weights %s"%Type, fontsize = 30)
    #plt.savefig(os.path.join(basepath,"betaseries_bevel/5_analysis/results/modularity_edges_reward_weighted.png"), format="PNG")
    plt.axis('off')
    plt.show()

def permuatator2(liist):
    results_dict={'no':{},'ov':{},'ob':{}}
    Dict=liist[0]
    dir=liist[2]
    thresh=liist[3]
    key=liist[1]
    print(key)
    for k, v in Dict.items():
        print(k)
        results_dict[k]=make_graphs(v,dir,thresh)
    return(results_dict)



def participation_coef(W, ci, degree='undirected'):
#from bctpy
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 = Kc2 + np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P
