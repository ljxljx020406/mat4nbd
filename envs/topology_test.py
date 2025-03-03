import pickle

topology_file= '../topology/new_usnet_500service_with_fragmentation5-3.pkl'
with open(topology_file, 'rb') as f:
    topology = pickle.load(f)

# for u,v,data in topology.edges(data=True):
#     print(data['wavelength_SNR'])

with open('../service/new_usnet_500services_to_be_sorting5-3.pkl', 'rb') as f:
    service_to_be_sorting = pickle.load(f)
print(service_to_be_sorting.items())
print(service_to_be_sorting.keys())