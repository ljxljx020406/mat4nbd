from back_utils import Path

import numpy as np
import pickle

from graph_utils import read_sndlib_topology, read_txt_file, get_k_shortest_paths, get_path_weight


def get_topology(file_name, topology_name, k_paths=5):
    k_shortest_paths = {}
    if file_name.endswith('.xml'):
        topology = read_sndlib_topology(file_name)
    elif file_name.endswith('.txt'):
        topology = read_txt_file(file_name)
    else:
        raise ValueError('Supplied topology is unknown')

    # 初始化每个边的频隙是否被占用的属性
    for u, v in topology.edges():
        id = 0
        # 为每个链路生成一个1x80的数组
        wavelength_power = np.zeros(80, dtype=float)
        wavelength_utilization = np.zeros(80, dtype=float)
        wavelength_SNR = np.zeros(80, dtype=float)
        wavelength_service = np.ones(80) * (-1)
        topology[u][v]['wavelength_power'] = wavelength_power
        topology[u][v]['wavelength_utilization'] = wavelength_utilization
        topology[u][v]['wavelength_SNR'] = wavelength_SNR
        topology[u][v]['wavelength_service'] = wavelength_service
        topology[u][v]['numsp'] = 0
        topology[u][v]['edge_id'] = id
        id += 1

    idp = 0
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
    topology.graph['name'] = topology_name
    topology.graph['ksp'] = k_shortest_paths
    topology.graph['k_paths'] = k_paths
    topology.graph['node_indices'] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph['node_indices'].append(node)
        topology.nodes[node]['index'] = idx
    # print('node_index:', topology.graph['node_indices'])
    # node_index: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']


    return topology


k_paths = 1

topology = get_topology('../topology/usnet.txt', 'USNET', k_paths=k_paths)
#
# # 将 topology 对象以二进制形式写入文件 f，以便以后可以使用 pickle.load() 来从文件中读取并重新构建对象
# with open(f'/Users/liujiaxin/Desktop/bishe/code/ljx-optical-rl-gym-multiband-main/optical-rl-gym-multiband-main/formal_code/topologies/nsfnet_chen_{k_paths}-paths.h5', 'wb') as f:
#     pickle.dump(topology, f)

# print('topology:',topology)

# topology2 = get_topology('../topologies/usnet.txt', 'USNET', k_paths=k_paths)


# 将 topology 对象以二进制形式写入文件 f，以便以后可以使用 pickle.load() 来从文件中读取并重新构建对象
with open(f'../topology/usnet_1path.h5', 'wb') as f:
    pickle.dump(topology, f)

# print('topology:',topology2)
# print(topology['1']['2']['wavelength_utilization'])
# print(topology.graph['node_indices'].index('1'))