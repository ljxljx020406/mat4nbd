import networkx as nx
import numpy as np
from itertools import islice
from utils import Path
import pickle

def read_txt_file(file):
    graph = nx.Graph()
    num_nodes = 0
    num_links = 0
    id_link = 0
    with open(file, 'r') as lines:
        # gets only lines that do not start with the # character
        nodes_lines = [value for value in lines if not value.startswith('#')]
        for idx, line in enumerate(nodes_lines):
            if idx == 0:
                num_nodes = int(line)
                for id in range(0, num_nodes):
                    graph.add_node(str(id), name=str(id))
            elif idx == 1:
                num_links = int(line)
            elif len(line) > 1:
                info = line.replace('\n', '').split(' ')
                graph.add_edge(info[0], info[1], id=id_link, index=id_link, weight=1, length=int(info[2]))
                # G.edges[3, 4]['weight'] = 0.3  可以改变边的属性值
                id_link += 1

    return graph

def get_k_shortest_paths(G, source, target, k, weight=None):
    '''
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    '''
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
def get_path_weight(graph, path, weight='weight'):
    return np.sum([graph[path[i]][path[i+1]][weight] for i in range(len(path) - 1)])

file_name = '../topology/usnet.txt'
topology = read_txt_file(file_name)

# 初始化每个边的频隙是否被占用的属性
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

idp = 0
k_paths = 1
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
topology.graph['name'] = 'usnet'
topology.graph['ksp'] = k_shortest_paths
topology.graph['k_paths'] = k_paths
topology.graph['node_indices'] = []
for idx, node in enumerate(topology.nodes()):
    topology.graph['node_indices'].append(node)
    topology.nodes[node]['index'] = idx

# 将 topology 对象以二进制形式写入文件 f，以便以后可以使用 pickle.load() 来从文件中读取并重新构建对象
with open(f'../topology/usnet_topology_1path.h5', 'wb') as f:
    pickle.dump(topology, f)