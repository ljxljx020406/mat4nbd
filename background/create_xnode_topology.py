import networkx as nx
import numpy as np
from itertools import islice
from utils import Path
import pickle
import random

# 创建一个Watts-Strogatz小世界网络，k为每个节点的邻居数，p为重连概率
topology = nx.watts_strogatz_graph(200, 8, 0.1)
for (u, v) in topology.edges():
    topology.edges[u, v]['weight'] = 1
    topology.edges[u, v]['length'] = random.choice([100 * i for i in range(1, 11)])

for u, v in topology.edges():
    id = 0
    # 为每个链路生成一个1x80的数组
    wavelength_power = np.zeros(80, dtype=float)
    wavelength_utilization = np.zeros(80, dtype=float)
    wavelength_SNR = np.zeros(80, dtype=float)
    wavelength_service = np.zeros(80)
    topology[u][v]['wavelength_power'] = wavelength_power
    topology[u][v]['wavelength_utilization'] = wavelength_utilization
    topology[u][v]['wavelength_SNR'] = wavelength_SNR
    topology[u][v]['wavelength_service'] = wavelength_service
    topology[u][v]['numsp'] = 0
    topology[u][v]['edge_id'] = id
    id += 1

def get_k_shortest_paths(G, source, target, k, weight=None):
    '''
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    '''
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
def get_path_weight(graph, path, weight='weight'):
    return np.sum([graph[path[i]][path[i+1]][weight] for i in range(len(path) - 1)])

idp = 0
k_paths = 5
k_shortest_paths = {}
for idn1, n1 in enumerate(topology.nodes()):
    for idn2, n2 in enumerate(topology.nodes()):
        if idn1 < idn2:
            paths = get_k_shortest_paths(topology, n1, n2, k_paths)
            lengths = [get_path_weight(topology, path) for path in paths]
            objs = []
            # print('idp, path, lengths:', idp, paths, lengths)
            for path, length in zip(paths, lengths):
                # print(idp, path, length)
                objs.append(Path(path_id=idp, node_list=path, length=length))
                # print(idp, length, path)
                idp += 1
            k_shortest_paths[n1, n2] = objs
            k_shortest_paths[n2, n1] = objs
topology.graph['name'] = '200nodes_8k'
topology.graph['ksp'] = k_shortest_paths
topology.graph['k_paths'] = k_paths
topology.graph['node_indices'] = []
for idx, node in enumerate(topology.nodes()):
    topology.graph['node_indices'].append(node)
    topology.nodes[node]['index'] = idx

with open(f'../topology/200nodes_8k.h5', 'wb') as f:
    pickle.dump(topology, f)