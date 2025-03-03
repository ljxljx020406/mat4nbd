import pickle
import pandas
from main_functions import release_service

from deepdiff import DeepDiff

# 加载 .pkl 文件
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compare_pkl_with_deepdiff(file1, file2):
    data1 = load_pkl(file1)
    data2 = load_pkl(file2)
    diff = DeepDiff(data1, data2)
    if diff:
        print("The contents of the two pkl files are different. Differences:")
        print(diff)
    else:
        print("The contents of the two pkl files are identical.")

with open('../service/1030usnet_150service_dict2.pkl', 'rb') as f:
    origin_service_dict = pickle.load(f)

with open('../topology/1030usnet_150services2.pkl', 'rb') as f:
    origin_topology = pickle.load(f)

to_delete = []
for service_id, service in origin_service_dict.items():
    if service.wavelength < 20:
        to_delete.append(service_id)
print('to_delete:', to_delete)
for service_id in to_delete:
    service = origin_service_dict[service_id]
    # print('service:', service.bit_rate)
    release_service(origin_topology, service, origin_service_dict)

# 保存service_dict到文件
with open('../service/1030usnet_150services_after_release3.pkl', 'wb') as f:
    pickle.dump(origin_service_dict, f)

# 保存topology对象到文件
with open('../topology/1030usnet_150service_with_fragmentation3.pkl', 'wb') as f:
    pickle.dump(origin_topology, f)


with open('../service/1030usnet_150service_dict2.pkl', 'rb') as f:
    origin_service_dict = pickle.load(f)

with open('../topology/1030usnet_150services2.pkl', 'rb') as f:
    origin_topology = pickle.load(f)

to_delete = []
for service_id, service in origin_service_dict.items():
    if service.wavelength < 20:
        to_delete.append(service_id)
print('to_delete:', to_delete)
for service_id in to_delete:
    service = origin_service_dict[service_id]
    # print('service:', service.bit_rate)
    release_service(origin_topology, service, origin_service_dict)

# 保存service_dict到文件
with open('../service/1030usnet_150services_after_release4.pkl', 'wb') as f:
    pickle.dump(origin_service_dict, f)

# 保存topology对象到文件
with open('../topology/1030usnet_150service_with_fragmentation4.pkl', 'wb') as f:
    pickle.dump(origin_topology, f)

compare_pkl_with_deepdiff('../topology/1030usnet_150service_with_fragmentation3.pkl', '../topology/1030usnet_150service_with_fragmentation4.pkl')
# with open('../service/1030usnet_150services_after_release.pkl', 'rb') as f:
#     service_dict = pickle.load(f)
# with open('../topology/1030usnet_150service_with_fragmentation.pkl', 'rb') as f:
#     topology = pickle.load(f)
#
# with open('../service/1030usnet_150services_after_release2.pkl', 'rb') as f:
#     service_dict2 = pickle.load(f)
# with open('../topology/1030usnet_150service_with_fragmentation2.pkl', 'rb') as f:
#     topology2 = pickle.load(f)
#
# if service_dict == service_dict2:
#     print('service identical!')
# else:
#     print('service different!')
#
# if topology == topology2:
#     print('topology identical!')
# else:
#     print('topology different!')
