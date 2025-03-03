from main_functions import new_service, naive_RWA, release_service, check_utilization
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import pandas
import openpyxl

with open('../topology/1030usnet_150service_with_fragmentation1.pkl', 'rb') as f:
    topology = pickle.load(f)

with open('../service/1030usnet_150services_after_release1.pkl', 'rb') as f:
    service_dict = pickle.load(f)

print(len(service_dict))
to_delete = []
for service_id, service in service_dict.items():
    if service.wavelength < 20:
        to_delete.append(service_id)
print('to_delete:', len(to_delete), to_delete)
for service_id in to_delete:
    service = service_dict[service_id]
    # print('service:', service.bit_rate)
    release_service(topology, service, service_dict)

# release_influenced_services(topology, service_dict)

# 保存service_dict到文件
with open('../service/1030usnet_150services_after_release1.pkl', 'wb') as f:
    pickle.dump(service_dict, f)

# 保存topology对象到文件
with open('../topology/1030usnet_150service_with_fragmentation1.pkl', 'wb') as f:
    pickle.dump(topology, f)
