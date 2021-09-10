def ges(mu,name):
    partition = community.best_partition(mu)
    vals = list(partition.values())
    nx.set_node_attributes(mu, partition, name)
    return((partition,vals, mu))





# Define the named tuple
MyItem = namedtuple("MyItem", "shape colour")
def mu_make_graphs(key, values, direction, min_cor):
    ########################################
    cor_matrix = np.asmatrix(values)
    x=abs(cor_matrix)
    mu=x.mean()
    ########################################
    G = nx.from_numpy_matrix(cor_matrix)
    tG = threshold(G, direction, min_cor)
    ########################################
    info = []
    for n in range(0,10):
        (partition,vals,graph)=ges(tG,'modules') ##### HERE
        info.append(MyItem(shape=str(vals), colour="%s"%key))
    frequency = Counter(info)
    while frequency.most_common()[0][-1] < 1:
        print('back in')
        for n in range(0,100):
            (partition,vals,graph)=ges(tG,'modules') ##### HERE
            info.append( MyItem(shape=str(vals), colour="%s"%key))
        frequency = Counter(info)
    res = frequency.most_common()[0][0][0][1:-1]
    X=list(map(int, res.split(',')))
    P = dict(zip(partition.keys(), X))
    print(P)
    print(x.shape)
    nx.set_node_attributes(tG, P, 'modules')
    vals=np.array(frequency.most_common()[0][0][0])
    ci=np.reshape(frequency.most_common()[0][0][0], (100, 1))
    W=np.array(cor_matrix)
    print(frequency.most_common()[0])






def permuatator3(liist):
    results_dict={'no':{},'ov':{},'ob':{}}
    Dict=liist[0]
    dir=liist[2]
    thresh=liist[3]
    key=liist[1]
    print(key)
    for k,v in Dict.items():
        results_dict[k]=mu_make_graphs(k,v,dir,thresh)
    an.adillyofapickle('/Users/gracer/Google Drive/HCP/HCP_graph/1200/datasets',results_dict_dict,'4_intmed_dict')
    for k,v in results_dict.items():
        results_dict[k]=mu_make_graphs2(v)
    return(results_dict)
