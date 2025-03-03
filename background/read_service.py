import pickle

with open('../service/usnet_500services_after_release4.pkl', 'rb') as f:
    service1 = pickle.load(f)
with open('../service/usnet_500services_to_be_sorting4.pkl', 'rb') as f:
    service2 = pickle.load(f)

print(len(service1), len(service2))
topology_file= '../topology/usnet_500service_with_fragmentation4.pkl'
with open(topology_file, 'rb') as f:
    topology = pickle.load(f)

service_id_list = []
for service in service1:
    service_id_list.append(service)

net_service_id = []
for u, v, attributes in topology.edges(data=True):
    net_service_id.append(attributes.get('wavelength_service', []))

cnt = 0
for i in range(len(net_service_id)):
    if not set(net_service_id[i]).isdisjoint(set(service_id_list)):
        cnt += 1

print(cnt)