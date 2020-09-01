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
