
# this is a collection of functions used in the HCP graph analysis
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
from scipy import stats
import community
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


#from https://wiki.biac.duke.edu/biac:analysis:resting_pipeline
import numpy.ma
#import nibabel
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
import bct as bct

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
        ########################################
        cor_matrix = np.asmatrix(values)
        x=abs(cor_matrix)
        mu=x.mean()
        ########################################
        G = nx.from_numpy_matrix(cor_matrix)
        tG = threshold(G, direction, min_cor)
        ########################################
        (partition,vals,graph)=ges(tG,'modules')
        partition_dict[key]=(partition,vals)
        vals=np.array(vals)
        ci=np.reshape(vals, (100, 1))
        # print(type(ci))
        W=np.array(cor_matrix)
        # print(W.shape)
        # print(type(W))
        PC=participation_coef(W=W, ci=ci, degree="undirected")
        # pdb.set_trace()
        pc_dict={}
        for i in range(len(PC)):
            # print(i)
            # print(PC[i])
            pc_dict[i]=PC[i]
        # pdb.set_trace()
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
    print(len(mylist))
    for key, value in dict_o_data.items():
        # print(key)
        # pdb.set_trace()
        cor_matrix = np.asarray(value)
        mylist.append(cor_matrix)
    print(len(mylist))
    x=np.stack(mylist, axis=2)
    mu=np.mean(x, axis=(2))
    return(mu)


def ges(mu,name):
    partition = community.best_partition(mu)
    vals = list(partition.values())
    nx.set_node_attributes(mu, partition, name)
    return((partition,vals, mu))

def threshold2(G, min_correlation):
    H = G.copy()
    for stock1, stock2, weight in list(G.edges(data=True)):
        if weight["weight"] < min_correlation:
            H.remove_edge(stock1, stock2)
    return(H)

def grace_graph(graph, group, basepath ,**kwargs):
    fs=50
    e,w = zip(*nx.get_edge_attributes(graph, 'weight').items())
    if bool(kwargs) == False:
        positions = nx.circular_layout(graph)
        size = 100
        title= "Modularity and edge weights \n of average %s graph"%(group)
        save="%s_graph.png"%(group)
    else:
        if 'position' in kwargs and kwargs['position']=='spectral':
            positions = nx.spectral_layout(graph)
            title= "Spectral modularity and edge weights \n of average %s graph"%(group)
            save="Spectral_%s_graph.png"%(group)
        elif 'position' in kwargs and kwargs['position']=='spring':
            positions = nx.sping_layout(graph)
            title= "Spring modularity and edge weights \n of average %s graph"%(group)
            save="Sping_%s_graph.png"%(group)
        else:
            positions = nx.circular_layout(graph)
            title= "Circle modularity and edge weights \n of average %s graph"%(group)
            save="Circle_%s_graph.png"%(group)
        if 'metric' in kwargs:
            nodes, size = zip(*nx.get_node_attributes(graph, kwargs['metric']).items())
        else:
            size = 100
            title = "basic"
            save="%s_graph.png"%(group)
        if 'thresh' in kwargs:
            tile=kwargs['thresh']
            purr=np.percentile(w, tile)
            print(purr)
            graph=threshold2(graph,purr)

    edges,weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
    nodes, color = zip(*nx.get_node_attributes(graph, 'modules').items()) #if your modules are named different change here
    # nodes, names = zip(*nx.get_node_attributes(graph, 'label').items()) #if your modules are named different change here
    g=graph
    #Figure size
    plt.figure(figsize=(80,50))

    #draws nodes
    color = np.array(color)
    n_color=len(list(set(color)))
    # nColormap=plt.cm.Set3 #check here if you want different colors https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    cM=color.max()
    cm=color.min()
    # get discrete colormap
    nColormap = plt.get_cmap('Set3', n_color)

    # scaling
    sz=np.array(size)
    scale=150/sz.max()
    sza=sz*scale
    # print(sz.shape)

    y=nx.draw_networkx_nodes(g,positions,
                           node_color=color,
                           node_size=sza,
                           alpha=0.8,
                           cmap= nColormap,
                           vmin=cm ,vmax=cM)

    #Styling for labels
    nx.draw_networkx_labels(g, positions,
                            # labels = label_dict,
                            font_size=fs,
                            font_family='sans-serif',
                            fontweight = 'bold')

    #draw edges
    weights=np.array(weights)
    eColormap=plt.cm.gist_rainbow #check here if you want different colors https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    # scaling
    wt=list(set(weights))
    wt=np.array(wt)
    wt2=-np.sort(-wt)
    wt0=wt2[1]

    escale=1/wt0
    esza=weights*escale
    E=list(set(esza))
    E2=-np.sort(-np.array(E))
    M=E2[1]
    m=esza.min()

    x=nx.draw_networkx_edges(g, positions,
                           edge_list=edges,
                           style='solid',
                           width = np.square(esza)*5,
                           edge_color = esza,
                           edge_cmap=eColormap,
                           edge_vmin=m,
                           edge_vmax=M)

    #COLORBAR STUFF
    node_bar=plt.colorbar(y, label='Module value')

    tick_locs = (np.arange(n_color) + 0.5)*(n_color-1)/n_color
    node_bar.set_ticks(tick_locs)

    # set tick labels (as before)
    node_bar.set_ticklabels(np.arange(n_color))


    sm = plt.cm.ScalarMappable(cmap=eColormap, norm=plt.Normalize(vmin = m, vmax=M))
    sm._A = []
    edge_bar=plt.colorbar(sm)

    for l in edge_bar.ax.yaxis.get_ticklabels():
        l.set_size(fs)
    for l in node_bar.ax.yaxis.get_ticklabels():
        l.set_size(fs)
        l.set_verticalalignment('center')

    node_bar.set_label('Modularity',fontsize = fs)
    edge_bar.set_label('Strength of edge weight',fontsize = fs)
    # Final plot stuff
    plt.axis('off')

    plt.title(title, fontsize = fs*2)
    basepath=basepath

    plt.savefig(os.path.join(basepath,save), format="PNG")
    plt.show()
    return()

# def module_fig(G, Type, basepath):
#     edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
#     #nodes, size = zip(*nx.get_node_attributes(G,'clustering').items())
#
#     positions=nx.circular_layout(G)
#     plt.figure(figsize=(80,50))
#     ### NODES ####
#     color = np.array(list(G.nodes))
#     color = np.array(color)
#     n_color=len(list(set(color)))
#     nColormap = plt.get_cmap('Set3', n_color)
#     cM=color.max()
#     cm=color.min()
#     y=nx.draw_networkx_nodes(G,positions,
#                            node_color=color,
#                            node_size=150000,
#                            alpha=1.0,
#                            cmap= nColormap,
#                            vmin=cm,vmax=cM )
#
#     #Styling for labels
#     nx.draw_networkx_labels(G, positions, font_size=25,
#                             font_family='sans-serif', fontweight = 'bold')
#
#     ### EDGES ####
#     weights=np.array(weights)
#     # scaling
#     wt=list(set(weights))
#     wt=np.array(wt)
#     wt2=-np.sort(-wt)
#     wt0=wt2[1]
#
#     escale=1/wt0
#     esza=weights*escale
#     E=list(set(esza))
#     E2=-np.sort(-np.array(E))
#     M=E2[1]
#     m=esza.min()
#
#
#     eColormap=plt.cm.gist_rainbow
#
#     x=nx.draw_networkx_edges(G, positions,
#                              edge_list=edges,
#                              style='solid',
#                              width = np.square(esza*5),
#                              edge_color = weights,
#                              edge_vmin=m,
#                              edge_vmax=M,
#                              edge_cmap= eColormap)
#
#
#     node_bar=plt.colorbar(y, label='Module value')
#
#     tick_locs = (np.arange(n_color) + 0.5)*(n_color-1)/n_color
#     node_bar.set_ticks(tick_locs)
#
#     # set tick labels (as before)
#     node_bar.set_ticklabels(np.arange(n_color))
#
#
#     sm = plt.cm.ScalarMappable(cmap=eColormap, norm=plt.Normalize(vmin = m, vmax=M))
#     sm._A = []
#     edge_bar=plt.colorbar(sm)
#
#     for l in edge_bar.ax.yaxis.get_ticklabels():
#         l.set_size(35)
#     for l in node_bar.ax.yaxis.get_ticklabels():
#         l.set_size(35)
#         l.set_verticalalignment('center')
#
#     node_bar.set_label('Modularity',fontsize = 25)
#     edge_bar.set_label('Strength of edge weight',fontsize = 25)
#
#     plt.title("Module Connectivity Weights %s"%Type, fontsize = 50)
#     plt.axis('off')
#     basepath= os.path.join(basepath,'images')
#     plt.savefig(os.path.join(basepath,"modularity_%s.png"%(Type)), format="PNG")
#     plt.show()

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

def permuatator3(liist):
    results_dict={'no':{},'ov':{},'ob':{}}
    Dict=liist[0]
    dir=liist[2]
    thresh=liist[3]
    key=liist[1]
    print(key)
    for k,v in Dict.items():
        results_dict[k]=mu_make_graphs(k,v,dir,thresh)
    return(results_dict)


def mu_make_graphs(key, values, direction, min_cor):
    ########################################
    cor_matrix = np.asmatrix(values)
    x=abs(cor_matrix)
    mu=x.mean()
    ########################################
    G = nx.from_numpy_matrix(cor_matrix)
    tG = threshold(G, direction, min_cor)
    ########################################
    (partition,vals,graph)=ges(tG,'modules')
    # partition_dict[key]=(partition,vals)
    vals=np.array(vals)
    ci=np.reshape(vals, (100, 1))
    W=np.array(cor_matrix)
    # PC=participation_coef(W=W, ci=ci, degree="undirected")
    PC=bct.participation_coef_sign(W,ci)
    PC=list(PC)
    PCpos=PC[0]
    pc_dict={}
    for i in range(len(PCpos)):
        pc_dict[i]=PCpos[i]
        # print(pc_dict)
    clustering = nx.clustering(tG, weight=True)
    # clustering_dict[key]=clustering
    ########################################
    centrality = nx.betweenness_centrality(tG, weight=True)
    # centrality_dict[key]=centrality
    ########################################
    print('start zdegree')
    zdegree=bct.module_degree_zscore(W, ci, flag=0)
    zzip=dict(zip(list(vals), zdegree))
    ########################################
    nx.set_node_attributes(G, centrality, 'centrality')
    nx.set_node_attributes(G, clustering, 'clustering')
    nx.set_node_attributes(G, pc_dict, 'PC')
    nx.set_node_attributes(G, partition, 'modules')
    ########################################
    zD = {}
    for node, mod in nx.get_node_attributes(G,'modules').items():
        zD[node]=zzip[mod]
    print(zD)
    nx.set_node_attributes(G, zD, 'zDegree')
    ########################################

    return({'mean_FC':mu, 'graphs':G, 'clustering_coeff':clustering, 'btn_centrality':centrality, 'PC':PCpos, 'modules':{'partition':partition,
    'values':vals,'graph':graph,'zdegree':zdegree}})


def corrector(x, alpha):
    results=x[1].ravel()
    mask = np.isfinite(results)
    pval_corrected = np.empty(results.shape)
    pval_corrected.fill(np.nan)
    pval_corrected[mask] = st.stats.multitest.multipletests(results[mask],alpha=alpha,method='fdr_bh')[1]
    p=np.reshape(pval_corrected, (100, 100))
    print(np.nanmin(p))
    ps = 1-p
    print(np.nanmax(ps))
    coor_fig(ps)
    return(p)

def coor_fig(df):
    plt.figure(figsize=(40,25))
    m=np.nanmin(df)
    M=np.nanmax(df)
    print('The max p value is %f'%M)
    sns.heatmap(df, linewidth=0.5,
                vmin=m, vmax=1,
                cmap=sns.cubehelix_palette(10000),
                cbar_kws={'ticks': [0.0, 0.2, 0.4, 0.5, 0.7, 0.8,0.975 ,1.0]})
    return(plt.show())

def inner_mod(graph, metric, group, tile, style):
    e,w = zip(*nx.get_edge_attributes(graph, 'weight').items())
    purr=np.percentile(w, tile)
    print(purr)
    g=threshold2(graph,purr)

    edges,weights = zip(*nx.get_edge_attributes(g, 'weight').items())
    weights=np.array(weights)
    print(weights.min())

    nodes, color = zip(*nx.get_node_attributes(g, metric).items())
    nodes, size = zip(*nx.get_node_attributes(g, metric).items())
    nodes, positions = zip(*nx.get_node_attributes(g,'modules').items())
    #positions
    if style == 'spectral':
        positions=nx.spectral_layout(g) #this is defining a circluar graph, if you want a different one you change the circular part of this line
    elif style == 'spring':
        positions=nx.spring_layout(g)
    else:
        positions=nx.circular_layout(g)
    #Figure size
    plt.figure(figsize=(80,50))

    #draws nodes
    color = np.array(color)
    colz=stats.zscore(color)
    nColormap=plt.cm.cool #check here if you want different colors https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    cM=colz.max()
    cm=colz.min()


    scale=15000/colz.max()
    y=nx.draw_networkx_nodes(g,positions,
                           node_color=colz,
                           node_size=np.square(colz)*scale,
                           alpha=0.8,
                           cmap= nColormap,
                           vmin=cm ,vmax=cM)

    #Styling for labels
    keeps=g.nodes()
    # dict_you_want = { your_key: note_dict[your_key] for your_key in keeps }
    nx.draw_networkx_labels(g, positions, font_size=50,
                            font_family='sans-serif',
                            fontweight = 'bold')

    #draw edges
    weights=np.array(weights)
    eColormap=plt.cm.gist_rainbow #check here if you want different colors https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    # scaling
    wt=list(set(weights))
    wt=np.array(wt)
    wt2=-np.sort(-wt)
    wt0=wt2[1]

    escale=1/wt0
    esza=weights*escale
    E=list(set(esza))
    E2=-np.sort(-np.array(E))
    M=E2[1]
    m=esza.min()

    x=nx.draw_networkx_edges(g, positions,
                           edge_list=edges,
                           style='solid',
                           width = np.square(esza)*5,
                           edge_color = esza,
                           edge_cmap=eColormap,
                           edge_vmin=m,
                           edge_vmax=M)

    #format the colorbar
    node_bar=plt.colorbar(y, label='Module value')

    sm = plt.cm.ScalarMappable(cmap=eColormap, norm=plt.Normalize(vmin = m, vmax=M))
    sm._A = []
    edge_bar=plt.colorbar(sm)


    for l in edge_bar.ax.yaxis.get_ticklabels():
        l.set_size(50)
    for l in node_bar.ax.yaxis.get_ticklabels():
        l.set_size(50)

    node_bar.set_label('%s'%metric,fontsize = 50)
    edge_bar.set_label('Strength of edge weight',fontsize = 50)

    plt.axis('off')
    plt.title("%s and edge weights of \n average %s graph"%(metric, group), fontsize = 100)
    basepath='/Users/gracer/Google Drive/HCP_graph/1200/images'

    # plt.savefig(os.path.join(basepath,"%s_%s_%s.png"%(metric,style,group)), format="PNG")
    plt.show()

def participation_coef(W, ci, degree='undirected'):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree
    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors
    # ci=np.reshape(ci, (100, 1))
    # Kc2=np.reshape(Kc2, (100, 1))
    # print('this is the shape of ci')
    # print(ci.shape)

    for i in range(1, int(np.max(ci)) + 1):
        # print(Kc2)
        # print(np.square(np.sum(W * (Gc == i), axis=1)))
        # pdb.set_trace()
        # np.add(Kc2,np.square(np.sum(W * (Gc == i), axis=1)), out=Kc2)
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P


def mod_world(dicti):
    mod_dict={}
    for key, value in dicti.items():
        if key == 'modules':
            dicti[key]['Q']=community.modularity(value['partition'], value['graph'], weight='weight')

    edge_btw=nx.edge_betweenness_centrality(dicti['graphs'], normalized=True, weight='weight')
    dicti['edge_btw']=edge_btw
    nx.set_edge_attributes(dicti['graphs'], edge_btw, 'betweenness')

    unique, counts = np.unique(dicti['modules']['values'], return_counts=True)
    for i in unique:
        mod_dict.update({i:[]})
    for q, w in dicti['modules']['partition'].items():
        mod_dict[w].append(q)
    return(dicti, mod_dict)


def sub_G(dicti, mod_dicti):
    subgraph_dict={}
    for key, value in mod_dicti.items():
        print(key)
        G=dicti['graphs']
        H = G.subgraph(value).copy()
        H = zed(H)
        subgraph_dict[key]=H
    return(subgraph_dict)

def zed(G):
    (partition,vals,graph)=ges(G, 'sub_modules')
    W=nx.to_numpy_matrix(graph)
    vals=np.array(vals)
    ci=np.reshape(vals, (len(vals), 1))
    print('start zdegree')
    sub_zdegree=bct.module_degree_zscore(np.array(W), vals, flag=0)
    zzip=dict(zip(list(vals), sub_zdegree))
    ########################################
    nx.set_node_attributes(G, partition, 'sub_modules')
    ########################################
    zD = {}
    for node, mod in nx.get_node_attributes(G,'sub_modules').items():
        zD[node]=zzip[mod]
    print(zD)
    nx.set_node_attributes(G, zD, 'sub_zDegree')
    return(G)

def df_maker(subgraph_dict, group):
    modstat_dict={}
    for mod, nodes in subgraph_dict.items():
        print(mod)
        modstat_dict[mod]={}
        for i in nodes:
            print(i)
            modstat_dict[mod].update({i:[]})
        modstat_dict[mod]=pd.DataFrame.from_dict(dict(subgraph_dict[mod].nodes(data=True)), orient='index')
        modstat_dict[mod]['group']=group
    return(modstat_dict)



# list1=[mean_dict['MZ'],'MZ','positive',0]
# list2=[mean_dict['DZ'], 'DZ', 'positive',0]
# list3=[mean_dict['NR'], 'NR','positive',0]
# list1=[summary_dict['NR']['no'],group]


def hubby(_df):
    _df.loc[abs(_df['sub_zDegree']) >= 2.5 , 'hub'] = 'yes'
    _df.loc[abs(_df['sub_zDegree'] < 2.5 ), 'hub'] = 'no'

def node_type(_df):
    _df.loc[(_df['hub'] == 'yes') & (_df['PC'] > 0) & (_df['PC'] < 0.3), 'node_type'] = 'provincial'
    _df.loc[(_df['hub'] == 'yes') & (_df['PC'] >= 0.3) & (_df['PC'] < 0.75), 'node_type'] = 'connector'
    _df.loc[(_df['hub'] == 'yes') & (_df['PC'] >= 0.75) & (_df['PC'] < 1), 'node_type'] = 'kinless'

    _df.loc[(_df['hub'] == 'no') & (_df['PC'] > 0) & (_df['PC'] < 0.05), 'node_type'] = 'ultra-peripheral'
    _df.loc[(_df['hub'] == 'no') & (_df['PC'] >= 0.05) & (_df['PC'] < 0.62), 'node_type'] = 'peripheral'
    _df.loc[(_df['hub'] == 'no') & (_df['PC'] >= 0.62) & (_df['PC'] < 0.8), 'node_type'] = 'connector'
    _df.loc[(_df['hub'] == 'no') & (_df['PC'] >= 0.8) & (_df['PC'] < 1), 'node_type'] = 'kinless'



def permuatator4(liist):
    dicti=liist[0]
    group=liist[1]
    [dicti,mod_dicti]=mod_world(dicti)
    subgraph_dict=sub_G(dicti, mod_dicti)
    modstat_dict=df_maker(subgraph_dict, group)
    for i in range(len(list(modstat_dict.values()))):
        hubby(modstat_dict[i])
        node_type(modstat_dict[i])
    return(modstat_dict)


def permuatator5(liist):
    dicti=liist[0]
    group=liist[1]
    [dicti,mod_dicti]=mod_world(dicti)
    subgraph_dict=sub_G(dicti, mod_dicti)
    return(subgraph_dict)



def find_latest(basepath, fi):
    p = os.path.join(basepath, fi)
    list_of_files = glob.glob(p) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return(latest_file)

def zscore(col):
    col_z = (col - col.mean())/col.std(ddof=0)
    return(col_z)


def module_fig(G, Type, basepath, aes):
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    #nodes, size = zip(*nx.get_node_attributes(G,'clustering').items())

    positions=nx.circular_layout(G)
    plt.figure(figsize= aes['general']['plot_size'])
    ### NODES ####
    print(len(G.nodes()))
    y=nx.draw_networkx_nodes(
        G,positions,
        node_color = aes['nodes']['color'],
        node_size = aes['nodes']['node_size'],
        alpha=1.0,
        cmap= plt.get_cmap('Set3', aes['nodes']['n_color']),
        vmin=aes['nodes']['min'],
        vmax=aes['nodes']['max'])

    #Styling for node labels
    nx.draw_networkx_labels(G,
                            positions,
                            font_size = aes['nodes']['font_size'],
                            font_family= aes['nodes']['font_family'],
                            fontweight = aes['nodes']['font_weight'])
    #Node color bar stuff
    n_color = aes['nodes']['n_color']
    print(n_color)
    node_bar=plt.colorbar(y, label='Module value')
    tick_locs = (np.arange(n_color) + 0.5)*(n_color-1)/n_color
    print(tick_locs)
    node_bar.set_ticks(tick_locs)

    # set tick labels (as before)
    node_bar.set_ticklabels(np.arange(n_color))
    node_bar.set_label('Modularity', fontsize =  aes['nodes']['font_size'])

    ## EDGES ##
    x=nx.draw_networkx_edges(G,
                             positions,
                             edge_list=edges,
                             style='solid',
                             width = weights*aes['edges']['width mod'],
                             edge_color = weights,
                             edge_vmin=aes['edges']['min'],
                             edge_vmax=aes['edges']['max'],
                             edge_cmap= aes['edges']['colormap'])
    # Edge color bar stuff
    sm = plt.cm.ScalarMappable(cmap=aes['edges']['colormap'],
                               norm=plt.Normalize(vmin = aes['edges']['min'], vmax=aes['edges']['max']))
    sm._A = []
    edge_bar=plt.colorbar(sm)

    for l in edge_bar.ax.yaxis.get_ticklabels():
        l.set_size(aes['nodes']['font_size'])
    for l in node_bar.ax.yaxis.get_ticklabels():
        l.set_size(aes['nodes']['font_size'])
        l.set_verticalalignment('center')

    edge_bar.set_label('Strength of edge weight', fontsize = aes['nodes']['font_size'])
    # Title things
    plt.title("Module Connectivity Weights %s"%Type, fontsize = aes['nodes']['font_size']*2)
    plt.axis('off')
    basepath= os.path.join(basepath,'images')
    plt.savefig(os.path.join(basepath,"modularity_%s.png"%(Type)), format="PNG")
    # plt.show()

def aesthetics(graph,node_size,font_size, font_family, font_weight, edge_att, plot_size, mod):
    aes={'general':{},
         'nodes':{},
         'edges':{}}
    # nodes
    color = np.array(list(graph.nodes))
    color = np.array(color)
    n_color=len(list(set(color)))
    print(n_color)
    aes['nodes']['color'] = color
    aes['nodes']['colormap'] = ['Set3', n_color]
    aes['nodes']['n_color'] = n_color
    aes['nodes']['max'] = float(color.max())
    aes['nodes']['min'] = float(color.min())
    aes['nodes']['font_size'] = int(font_size)
    aes['nodes']['font_family'] = font_family
    aes['nodes']['font_weight'] = font_weight
    aes['nodes']['node_size'] = int(node_size)
    #edges
    aes['edges']['colormap'] = plt.cm.gist_rainbow
    edges,weights = zip(*nx.get_edge_attributes(graph,edge_att).items())
    weights=np.array(weights)
    aes['edges']['width mod'] = mod
    aes['edges']['min'] = weights.min()
    aes['edges']['max'] = weights.max()
    # General
    aes['general']['plot_size']=plot_size #tuple 80,50
    return(aes)



# for key, values in summary_dict.items():
#     print(key)
#     for k,v in values.items():
#         print(k)
#         unique, counts = np.unique(summary_dict[key][k]['modules']['values'], return_counts=True)
#         for i in unique:
#             mod_dict[key][k].update({i:[]})
#         for q, w in summary_dict[key][k]['modules']['partition'].items():
#             mod_dict[key][k][w].append(q)
