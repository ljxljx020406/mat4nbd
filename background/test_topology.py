import pickle
import networkx as nx

with open('../topology/usnet_topology2.h5', 'rb') as f:
    topology = pickle.load(f)

# 计算度中心性
degree_centrality = nx.degree_centrality(topology)
print("Degree Centrality:", degree_centrality)

# 计算介数中心性
betweenness_centrality = nx.betweenness_centrality(topology)
print("Betweenness Centrality:", betweenness_centrality)

# 计算紧密中心性
closeness_centrality = nx.closeness_centrality(topology)
print("Closeness Centrality:", closeness_centrality)

# 将结果排序
sorted_degree = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
sorted_closeness = sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True)

print("Sorted Degree Centrality:", sorted_degree)
print("Sorted Betweenness Centrality:", sorted_betweenness)
print("Sorted Closeness Centrality:", sorted_closeness)