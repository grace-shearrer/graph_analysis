
# this is a collection of functions used in the HCP graph analysis for visualization
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
import analysis as an


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



def grace_graph(graph, group, basepath ,**kwargs):
    g=graph
    #Figure size
    plt.figure(figsize= aes['general']['plot_size'])

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
            graph=an.threshold2(graph,purr)
    # NODES #
    nodes, color = zip(*nx.get_node_attributes(graph, 'modules').items()) #if your modules are named different change here
    #draws nodes
    color = np.array(aes['nodes']['color'])
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
                            font_size=aes['nodes']['font_size'],
                            font_family=aes['nodes']['font_family'],
                            fontweight = aes['nodes']['font_weight'])
    #COLORBAR STUFF
    node_bar=plt.colorbar(y, label='Module value')

    tick_locs = (np.arange(n_color) + 0.5)*(n_color-1)/n_color
    node_bar.set_ticks(tick_locs)

    # set tick labels (as before)
    node_bar.set_ticklabels(np.arange(n_color))

    for l in node_bar.ax.yaxis.get_ticklabels():
        l.set_size(fs)
        l.set_verticalalignment('center')

    node_bar.set_label('Modularity',fontsize = fs)

    # EDGES #
    edges,weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
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
                           width = weights*aes['edges']['width mod'],
                           edge_color = esza,
                           edge_cmap=eColormap,
                           edge_vmin=m,
                           edge_vmax=M)
    sm = plt.cm.ScalarMappable(cmap=eColormap, norm=plt.Normalize(vmin = m, vmax=M))
    sm._A = []
    edge_bar=plt.colorbar(sm)

    for l in edge_bar.ax.yaxis.get_ticklabels():
        l.set_size(fs)

    edge_bar.set_label('Strength of edge weight',fontsize = fs)
    # Final plot stuff
    plt.axis('off')

    plt.title(title, fontsize = fs*2)
    basepath=basepath

    plt.savefig(os.path.join(basepath,save), format="PNG")
    plt.show()
    return()

def inner_mod(graph, metric, group, tile, style):
    e,w = zip(*nx.get_edge_attributes(graph, 'weight').items())
    purr=np.percentile(w, tile)
    print(purr)
    g=an.threshold2(graph,purr)

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
    edge_bar.set_label('Strength of edge weight',fontsize = 50)

    plt.axis('off')
    plt.title("%s and edge weights of \n average %s graph"%(metric, group), fontsize = 100)
    basepath='/Users/gracer/Google Drive/HCP_graph/1200/images'

    # plt.savefig(os.path.join(basepath,"%s_%s_%s.png"%(metric,style,group)), format="PNG")
    plt.show()

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
