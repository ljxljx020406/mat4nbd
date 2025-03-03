import pickle

topology_file='../topology/1030usnet_150service_with_fragmentation2.pkl'

with open('../service/1030usnet_150services_after_release2.pkl', 'rb') as f:
    service_dict = pickle.load(f)

with open(topology_file, 'rb') as f:
    topology = pickle.load(f)

for id, service in service_dict.items():
    path = service.path
    wave = service.wavelength
    total_utilization = 0
    for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]
        total_utilization += topology[u][v]['wavelength_utilization'][wave]
    service.utilization = total_utilization/len(path)

# 按utilization属性排序，返回排序后的键值对列表
sorted_items_reverse = sorted(service_dict.items(), key=lambda item: item[1].bit_rate, reverse=True)
# 转换回字典
reverse_sorted_service_dict = dict(sorted_items_reverse)
print(len(reverse_sorted_service_dict))

reverse_services_to_be_sorting = {}

for id, service in service_dict.items():
    path = service.path
    wave = service.wavelength
    total_utilization = 0
    for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]
        if topology[u][v]['wavelength_utilization'][wave] > 1:
            reverse_services_to_be_sorting[id] = service

for id, service in reverse_sorted_service_dict.items():
    if service.utilization < 0.37:
        reverse_services_to_be_sorting[id] = service

print(len(reverse_services_to_be_sorting))
print(reverse_services_to_be_sorting.keys())

# 保存service_dict到文件
with open('../service/1030usnet_150services_to_be_sorting2.pkl', 'wb') as f:
    pickle.dump(reverse_services_to_be_sorting, f)
