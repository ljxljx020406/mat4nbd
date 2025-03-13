# 包含一些主要功能函数。包含内容实时更新：
# 1. one_link_transmission
# 2. generate_service
# 3. naive_RWA
# 4. release_service
import copy

import numpy as np
from numpy import log10, abs, arange, arcsinh, arctan, isfinite, log, mean, pi, sum, zeros, exp
import math
from back_utils import Service
import pickle
import random
import sys
import os
import numba
from numba import jit, njit, prange
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import time

sys.path.append(os.path.abspath('../gnpy_transmission'))

# one_link_transmission references
# [1] A Generalized Raman Scattering Model for Real-Time SNR Estimation of Multi-Band Systems
# [2] Modeling and mitigation of fiber nonlinearity in wideband optical signal transmission

# 1. one_link_transmission function
@njit(nogil=True, cache=True)
def get_alpha(frequency):
    '''
    计算不同频率处的alpha值
    '''
    wavelength0 = 1550e-9
    wavelength = 3e8 / frequency
    alpha0 = 0.162
    alpha1 = -7.3764e-5
    alpha2 = 3.7685e-6
    alpha = alpha2*(wavelength-wavelength0)**2 + alpha1*(wavelength-wavelength0) + alpha0
    #print('alpha:', alpha)
    return alpha/ 4.343 / 1e3

@njit(nogil=True, cache=True)
def get_r_f(frequency, P_total):
    '''
    计算SRS的中间过程
    '''
    delta_f = 15e12
    f_m1 = 184.325e12
    f_M1 = 190.325e12
    f_m2 = 190.675e12
    f_M2 = 196.675e12
    B_t = 12e12
    if frequency-delta_f < f_m1 and frequency+delta_f > f_M1:
        return P_total*frequency
    elif frequency-delta_f > f_m1 and frequency+delta_f < f_M1:
        return 0.0
    elif frequency-delta_f < f_m1 and frequency+delta_f < f_M1:
        return P_total/B_t*(frequency**2/2 - frequency*f_m1 + (f_M1**2-delta_f**2)/2)
    elif frequency-delta_f > f_m1 and frequency+delta_f > f_M1:
        return P_total/B_t*(frequency*f_M1 - frequency**2/2 - (f_m1**2-delta_f**2)/2)
    elif frequency-delta_f < f_m2 and frequency+delta_f > f_M2:
        return P_total*frequency
    elif frequency-delta_f > f_m2 and frequency+delta_f < f_M2:
        return 0.0
    elif frequency-delta_f < f_m2 and frequency+delta_f < f_M2:
        return P_total/B_t*(frequency**2/2 - frequency*f_m2 + (f_M2**2-delta_f**2)/2)
    elif frequency-delta_f > f_m2 and frequency+delta_f > f_M2:
        return P_total/B_t*(frequency*f_M2 - frequency**2/2 - (f_m2**2-delta_f**2)/2)
    else:
        return 0.0

@njit(parallel=True, nogil=True, cache=True)
def get_Pi_z(distance, P_total, frequencies, Power, i, channels):
    '''
    计算经过distance传输后各信道的功率值
    '''
    C_r = 0.028 / 1e3 / 1e12
    alpha = get_alpha(frequencies[i])
    L_eff = (1-np.exp(-alpha*distance))/alpha
    tmp_frequency = frequencies[i]
    r_f = get_r_f(tmp_frequency, P_total)
    tmp_sum = 0.0
    for j in prange(channels):
        alpha_j = get_alpha(frequencies[j])
        L_eff_j = (1-np.exp(-alpha_j*distance))/alpha_j
        r_f_j = get_r_f(frequencies[j], P_total)
        # print('r_fj:', r_f_j)
        # print('re:',Power[j] * np.exp(-C_r*L_eff_j*r_f_j))
        # 确保每一步都是标量操作
        power_j = Power[j]
        exp_term = np.exp(-C_r * L_eff_j * r_f_j)
        contribution = power_j * exp_term  # 将乘积转换为标量

        tmp_sum += contribution
    result = Power[i] * (np.exp(-C_r*L_eff*r_f)*P_total)/tmp_sum
    # print("Power after z:", result)
    return result

@njit(nogil=True, cache=True)
def lin2db(value):
    '''
    ln -> dB
    '''
    return 10 * log10(value)
def watt2dbm(value):
    '''
    W -> dBm
    '''
    return lin2db(value * 1e3)

@njit(nogil=True, cache=True)
def removenan(x):
    '''
    除掉nan值
    '''
    x[~isfinite(x)] = 0
    return x

def todB(x):
    '''
    转化为dB
    '''
    return 10 * log(x) / log(10)

@njit(parallel=True, nogil=True, cache=True)
def calculate_ASE_noise(Att, fi, Bch, distance):
    channels, n = fi.shape
    c = 3e8
    n_sp = 1.41  # n_sp = NF(4.5dB)/2
    h = 6.62607015e-34
    RefLambda = 1575e-9  # 假设这是你的常量
    num_of_spans = int(np.ceil(distance / 100e3))
    Length = 100 * 1e3 * np.ones(num_of_spans)
    single_ASE = np.zeros((channels, n))

    for j in prange(n):
        for i in prange(channels):
            a_i = Att[i, j]  # \alpha of COI in fiber span j
            f_i = fi[i, j]  # f_i of COI in fiber span j
            B_i = Bch[i, j]  # B_i of COI in fiber span j
            length_i = Length[j]
            single_ASE[i, j] = 2 * n_sp * h * (f_i + c / RefLambda) * B_i * (np.exp(a_i * length_i) - 1)

    return np.sum(single_ASE, axis=1)

@njit(parallel=True, nogil=True, cache=True)
def _numba_one_link_transmission(distance, channels, Power, frequencies):
    P_total = np.sum(Power)
    for i in prange(channels):
        Power[i] = get_Pi_z(100e3, P_total, frequencies, Power, i, channels)  # 一个span一补偿

    num_of_spans = int(np.ceil(distance / 100e3))
    RefFreq = 190.5e12
    fi = np.array([(f - RefFreq) for f in frequencies])
    fi = fi.reshape(-1, 1)

    Bch = 150e9 * np.ones((channels, num_of_spans))
    Att = 0.2 / 4.343 / 1e3 * np.ones((channels, num_of_spans))

    ASE_noise = calculate_ASE_noise(Att, fi, Bch, distance)

    return ASE_noise, Power
    # return copy_Power

@njit(nogil=True, cache=True)
def cal_eps(B_i, f_i, a_i, mean_L, beta2, beta3):
    return (3 / 10) * log(1 + (6 / a_i) / (
                    mean_L * arcsinh(pi ** 2 / 2 * abs(mean(beta2) + 2 * pi * mean(beta3) * f_i) / a_i * B_i ** 2)))

@njit(nogil=True, cache=True)
def cal_SPM(phi_i, T_i, B_i, a, a_bar, gamma):
 return 4 / 9 * gamma ** 2 / B_i ** 2 * pi / (phi_i * a_bar * (2 * a + a_bar)) \
         * ((T_i - a ** 2) / a * arcsinh(phi_i * B_i ** 2 / a / pi) + ((a + a_bar) ** 2 - T_i) / (a + a_bar) * arcsinh(
         phi_i * B_i ** 2 / (a + a_bar) / pi))

@njit(nogil=True, cache=True)
def cal_XPM(Pi, Pk, phi_ik, T_k, B_i, B_k, a, a_bar, gamma):
    if Pi == 0:
        return 0
    else:
        return 32 / 27 * sum((Pk / Pi) ** 2 * gamma ** 2 / (B_k * phi_ik * a_bar * (2 * a + a_bar))
                             * ((T_k - a ** 2) / a * np.arctan(phi_ik * B_i / a)
                                + ((a + a_bar) ** 2 - T_k) / (a + a_bar) * np.arctan(phi_ik * B_i / (a + a_bar)))
                             )

@njit(parallel=True, nogil=True, cache=True)
def calculate_NLI_noise(
        Att, Att_bar, Cr, Pch, fi, Bch, Length, D, S, gamma, RefLambda):
    """
    Returns nonlinear interference power and coefficient for each WDM
    channel.

    Format:
    - channel dependent quantities have the format of a N_ch x n matrix,
      where N_ch is the number of channels slots and n is the number of spans.
    - channel independent quantities have the format of a 1 x n matrix
    - channel and span independent quantities are scalars

    INPUTS:
        Att: attenuation coefficient [Np/m] of channel i of span j,
            format: N_ch x n matrix
        Att_bar: attenuation coefficient (bar) [Np/m] of channel i of span j,
            format: N_ch x n matrix
        Cr[i,j]: the slope of the linear regression of the normalized Raman gain spectrum [1/W/m/Hz] of channel i of span j,
            format: N_ch x n matrix

        Pch[i,j]:  the launch power [W] of channel i of span j,
            format: N_ch x n matrix
        fi[i,j]: center frequency relative to the reference frequency (3e8/RefLambda) [Hz]
            of channel i of span j, format: N_ch x n matrix
        Bch[i,j]: the bandwidth [Hz] of channel i of span j,
            format: N_ch x n matrix

        Length[j]: the span length [m] of span j,
            format: 1 x n vector
        D[j]: the dispersion coefficient [s/m^2] of span j,
            format: 1 x n vector
        S[j]: the span length [s/m^3] of span j,
            format: 1 x n vector
        gamma[j]: the span length [1/W/m] of span j,
            format: 1 x n vector
        RefLambda: is the reference wavelength (where beta2, beta3 are defined) [m],
            format: 1 x 1 vector

        coherent: boolean for coherent or incoherent NLI accumulation across multiple fiber spans

    RETURNS:
        NLI: Nonlinear Interference Power[W],
            format: N_ch x 1 vector
        eta_n: Nonlinear Interference coeffcient [1/W^2],
            format: N_ch x 1 matrix
    """
    channels, n = Att.shape

    c = 3e8

    a = Att
    a_bar = Att_bar
    L = Length
    P_ij = Pch
    # print("Power:",P_ij)
    Ptot = np.sum(P_ij, axis=0)

    beta2 = -D * RefLambda ** 2 / (2 * pi * c)
    beta3 = RefLambda ** 2 / (2 * pi * c) ** 2 * (RefLambda ** 2 * S + 2 * RefLambda * D)
    # n_sp = 1.41  # n_sp = NF(4.5dB)/2  from "modeling and mitigation of fiber nonlinearity in wideband optical signal transmission"
    # h = 6.62607015*1e-34

    # Average Coherence Factor
    # mean_att_i = mean(a, axis=1)  # average attenuation coefficent for channel i
    mean_att_i = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        mean_att_i[i] = np.sum(a[i]) / a.shape[1]  # 计算每行的均值
    mean_L = np.mean(L)  # average fiber length

    eta_SPM = zeros((channels, n))
    eta_XPM = zeros((channels, n))

    for j in prange(n):
        """ Calculation of nonlinear interference (NLI) power in fiber span j """
        for i in prange(channels):
            """ Compute the NLI of each COI """
            not_i = arange(channels) != i
            a_i = a[i, j]  # \alpha of COI in fiber span j
            a_k = a[not_i, j]  # \alpha of INT in fiber span j
            a_bar_i = a_bar[i, j]  # \bar{\alpha} of COI in fiber span j
            a_bar_k = a_bar[not_i, j]  # \bar{\alpha} of INT in fiber span j
            f_i = fi[i, j]  # f_i of COI in fiber span j
            #print(f_i)
            f_k = fi[not_i, j]  # f_k of INT in fiber span j
            B_i = Bch[i, j]  # B_i of COI in fiber span j
            B_k = Bch[not_i, j]  # B_k of INT in fiber span j
            Cr_i = Cr[i, j]  # Cr  of COI in fiber span j
            Cr_k = Cr[not_i, j]  # Cr  of INT in fiber span j
            P_i = P_ij[i, j]  # P_i of COI in fiber span j
            P_k = P_ij[not_i, j]  # P_k of INT in fiber span j

            phi_i = 3 / 2 * pi ** 2 * (beta2[j] + pi * beta3[j] * (f_i + f_i))  # \phi_i of COI in fiber span j
            phi_ik = 2 * pi ** 2 * (f_k - f_i) * (
                        beta2[j] + pi * beta3[j] * (f_i + f_k))  # \phi_ik of COI-INT pair in fiber span j

            T_i = (a_i + a_bar_i - f_i * Ptot[j] * Cr_i) ** 2  # T_i of COI in fiber span j
            T_k = (a_k + a_bar_k - f_k * Ptot[j] * Cr_k) ** 2  # T_k of INT in fiber span j

            eta_SPM[i, j] = cal_SPM(phi_i, T_i, B_i, a_i, a_bar_i, gamma[j]) * n ** cal_eps(B_i, f_i, mean_att_i[i], mean_L, beta2, beta3)  # computation of SPM contribution in fiber span j
            eta_XPM[i, j] = cal_XPM(P_i, P_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma[j])  # computation of XPM contribution in fiber span j

    nonzero_mask = P_ij[:, 0] != 0  # 创建一个布尔掩码，表示 P_ij[:, 0] 是否不等于零
    eta_n = np.zeros(channels)  # 先将所有值初始化为零
    # eta_n[nonzero_mask] = np.sum(
    #     (eta_SPM[nonzero_mask] + eta_XPM[nonzero_mask]),
    #     axis=1)
    eta_n[nonzero_mask] = np.sum(eta_SPM[nonzero_mask] + eta_XPM[nonzero_mask], axis=1)

    # computation of NLI normalized to transmitter power, see Ref. [1, Eq. (5)]
    NLI = P_ij[:, 0] ** 3 * eta_n  # Ref. [1, Eq. (1)]

    #print("NLI:",NLI)
    return NLI

@njit(parallel=True, nogil=True, cache=True)
def calculate_GSNR(Power, noise, channels):
    GSNR = np.zeros(channels)
    for i in prange(channels):
        if Power[i] != 0:
            GSNR[i] = lin2db(Power[i] / noise[i])
        else:
            GSNR[i] = 0
    return GSNR


# def one_link_transmission(distance, channels, Power, frequencies):
#     # 现在将 RefLambda 作为参数传递给 _numba_one_link_transmission
#     start = time.time()
#     ASE_noise, Power_after_transmission = _numba_one_link_transmission(
#         distance, channels, Power, frequencies
#     )
#     srs_end = time.time()
#     print(f"SRS运行时间: {srs_end - start:.6f} 秒")
#     nli_start = time.time()
#
#     NLI_noise = ljx_propagate(Power_after_transmission)
#     nli_end = time.time()
#     print(f"NLI运行时间: {nli_end - nli_start:.6f} 秒")
#
#     noise = ASE_noise + NLI_noise
#
#     GSNR = np.zeros(channels)
#     for i in range(channels):
#         if Power_after_transmission[i] != 0:
#             GSNR[i] = lin2db(Power_after_transmission[i] / noise[i])
#         else:
#             GSNR[i] = 0
#     end = time.time()
#     print('one_link_trans', end-start)
#     return Power_after_transmission, GSNR

@njit(parallel=True, nogil=True, cache=True)
def tile_implementation(fi, num_of_spans):
    rows, cols = fi.shape
    expanded = np.empty((rows, num_of_spans))  # 创建一个新的数组，大小为 (rows, 8)
    for i in prange(rows):
        for j in prange(num_of_spans):
            expanded[i, j] = fi[i, 0]  # 将每一行复制到新的数组中
    return expanded


def one_link_transmission(distance, channels, Power, frequencies):
    '''
    new！！！不使用gnpy的函数，全部自写函数
    单链路传输，仅考虑信道间SRS、自发辐射噪声和非线性噪声
    input: distance-链路长度， channels-信道数量， Power-输入功率列表，1*channels， frequencies-中心频率列表，1*channels
    output: 该链路的GSNR列表，1*channels
    '''
    time0 = time.time()
    # P_total = np.sum(Power)
    # for i in range(channels):
    #     Power[i] = get_Pi_z(distance=distance, P_total=P_total, frequencies=frequencies, Power=Power, i=i, channels=channels)
    ASE_noise, Power = _numba_one_link_transmission(
        distance, channels, Power, frequencies
    )
    time1 = time.time()
    num_of_spans = math.ceil(distance / 100e3)
    RefFreq = 190.5e12  # 参考频率（Hz）
    # 计算频率偏移
    fi = [(f - RefFreq) for f in frequencies]

    Att = 0.2 / 4.343 / 1e3 * np.ones((channels, num_of_spans))
    Att_bar = Att
    Cr = 0.028 / 1e3 / 1e12 * np.ones((channels, num_of_spans))
    # Pch = 0.0031622776602 * np.ones((channels, num_of_spans)) # 需修改
    tmpPower = 0.0031622776602 * ((Power != 0).astype(int))
    tmpPower = tmpPower.reshape(channels, 1)
    Pch = tile_implementation(tmpPower, num_of_spans)
    fi = np.array(fi).reshape(-1, 1)
    fi = tile_implementation(fi, num_of_spans)
    # Bch = np.tile(150e9, [channels, num_of_spans])
    Bch = np.empty((channels, num_of_spans))
    for i in range(channels):
        for j in range(num_of_spans):
            Bch[i, j] = 150e9
    Length = 100 * 1e3 * np.ones(num_of_spans)
    D = 17 * 1e-12 / 1e-9 / 1e3 * np.ones(num_of_spans)
    S = 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * np.ones(num_of_spans)
    gamma = 1.21 / 1e3 * np.ones(num_of_spans)
    RefLambda = 1575e-9

    # print('power+ASE:', time1 - time0)
    NLI_noise = calculate_NLI_noise(Att, Att_bar, Cr, Pch, fi, Bch, Length, D, S, gamma, RefLambda)
    time2 = time.time()
    # ASE_noise = calculate_ASE_noise(Att, fi, Bch, distance)

    # print('NLI:', time2-time1)
    noise = NLI_noise + ASE_noise

    GSNR = calculate_GSNR(Power, noise, channels)
    time4 = time.time()
    # print('cal_GSNR:', time4-time2)

    return Power, GSNR

# # 使用one_link_transmission的一个实例
# distance = 600e3
# channels = 80
# Power = 0.0031622776602 * np.ones(channels)
# # Power[0:10] = 0
# # Power[40:50] = 0
# frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
# GSNR = one_link_transmission(distance, channels, Power, frequencies)
# print(GSNR)


# 2. generate_service
# # 读取网络拓扑结构
# with open('../topologies/nsfnet_chen_5-paths.h5', 'rb') as f:
#     topology = pickle.load(f)

# 各节点生成业务的可能性，共拓扑节点个数个，总和=1
# node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
#        0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
#        0.07607608, 0.12012012, 0.01901902, 0.16916917])
node_request_probabilities = 1/24 * np.ones(24, dtype=float)

# 固定随机数
rng1 = random.Random(41)
rng2 = random.Random(42)
services_processed_since_reset = 1

def _get_node_pair(topology):
    """
    Uses the `node_request_probabilities` variable to generate a source and a destination.

    :return: source node, source node id, destination node, destination node id
    """
    # 按概率生成源节点
    # print('node:', len(topology.nodes()), len(node_request_probabilities))
    src = rng1.choices([x for x in topology.nodes()], weights=node_request_probabilities)[0] #Escoge aleatoriamente un nodo, si se le pasa probabilidad la tiene en cuenta, sino genera trafico uniforme.
    # 原节点id
    src_id = topology.graph['node_indices'].index(src)
    new_node_probabilities = np.copy(node_request_probabilities)
    # 防止原节点被选为目的节点
    new_node_probabilities[src_id] = 0.
    new_node_probabilities = new_node_probabilities / np.sum(new_node_probabilities)
    # 选定目的节点
    dst = rng1.choices([x for x in topology.nodes()], weights=new_node_probabilities)[0]
    dst_id = topology.graph['node_indices'].index(dst)
    return src, src_id, dst, dst_id

def new_service(topology, services_processed_since_reset, bit_rate1, bit_rate2):
    '''
    生成一个随机业务
    '''
    src, src_id, dst, dst_id = _get_node_pair(topology)

    # list of possible bit-rates for the request
    bit_rate = rng2.randint(bit_rate1, bit_rate2)
    # values = [800, 400]
    # probabilities = [0.25, 0.75]
    #
    # # 随机选择
    # bit_rate = np.random.choice(values, p=probabilities)

    service = Service(service_id=services_processed_since_reset, source=src, source_id=src_id,
                      destination=dst, destination_id=dst_id,
                      bit_rate=bit_rate)

    # services_processed_since_reset += 1

    return service

# # generate_service实例
# service_not_allocated = []
# for i in range(100):
#     temp_service = new_service(topology)
#     service_not_allocated.append(temp_service)
# print(service_not_allocated[1].bit_rate)

# 3. allocate a service
def naive_RWA(topology, service:Service, service_dict):
    '''
    input: topology, service
    output: path_node_list, wavelength j
    '''
    src = service.source_id
    dst = service.destination_id
    bit_rate = service.bit_rate
    total_utilization = 0
    allocation = False
    reason = None
    # if bit_rate > 400:
    #     service.snr_requirement = 26.5
    # else:
    #     service.snr_requirement = 18.2
    if bit_rate < 150:
        service.snr_requirement = 9
    elif bit_rate < 300:
        service.snr_requirement = 12
    elif bit_rate < 450:
        service.snr_requirement = 16
    elif bit_rate < 600:
        service.snr_requirement = 18.6
    elif bit_rate < 750:
        service.snr_requirement = 21.6
    else:
        service.snr_requirement = 24.6

    for path in topology.graph['ksp'][str(src), str(dst)]:
        path_start_time = time.time()
        start_wavelength = rng1.randint(0, 80 - 1)
        wave_reason = np.zeros(80)
        for offset in range(80):
            j = (start_wavelength + offset) % 80
            allocation = True
            outer_break = False
            reason = None
            Power = np.zeros((len(path.node_list), 80), dtype=float)
            path_GSNR = np.zeros((len(path.node_list), 80), dtype=float)
            # Power_after_transmission = np.zeros(80, dtype=float)
            for i in range((len(path.node_list)-1)):
                if outer_break:
                    break
                # print("i=",i)
                u = path.node_list[i]
                v = path.node_list[i+1]
                # original_powers.append(topology[u][v]['wavelength_power'].copy())  # 先保存原始功率
                # print('power:', topology[u][v]['wavelength_power'][j], type(topology[u][v]['wavelength_power'][j]))

                # 检查波长是否空闲
                if not (topology[u][v]['wavelength_power'][j] == 0 or (np.isnan(topology[u][v]['wavelength_power'][j]))):
                    reason = 'no free wavelength'
                    wave_reason[j] = 1
                    # print('power:', topology[u][v]['wavelength_power'][j])
                    allocation = False
                    break
                else:
                    # 检查SNR是否满足要求
                    # 获取链路参数
                    distance = topology[u][v]['length']
                    channels = 80
                    # if i == 0:
                    Power[i] = copy.deepcopy(topology[u][v]['wavelength_power'])
                    Power[i][j] = service.power
                        # print('Power[i]:', Power[i])
                        # print("power:", Power)
                    frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
                    # if i != 0:
                    #     topology[u][v]['wavelength_power'] = Power_after_transmission
                    # 临时添加待分配业务的功率
                    # topology[u][v]['wavelength_power'][j] = service.power
                    tmp = copy.deepcopy(Power[i])
                    # tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
                    # tmp[j] = service.power
                    tmp = np.array(tmp)
                    # # print('tmp:', tmp)
                    # if i != 0:
                    #     # print('unupdate_Power[i]:', Power[i])
                    #     Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                    #                      range(len(Power[i]))]
                    #     tmp = Power[i]
                    #     tmp = np.array(tmp)
                        # print('update_Power[i]:', Power[i])
                    # original_powers.append(topology[u][v]['wavelength_power'].copy())  # 保存原始功率

                    # 计算链路的GSNR
                    Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                    # print('power:', Power_after_transmission)
                    # print('GSNR:', GSNR)
                    path_GSNR[i] = GSNR
                    # topology[u][v]['wavelength_power'] = Power_after_transmission  # 更新功率
                    # 检查SNR是否满足要求

                    #if Decimal("{:.5f}".format(GSNR[j])) < Decimal(str(snr_requirement)):
                    if service.snr_requirement > GSNR[j]:
                        reason = 'GSNR < snr_requirement' +'  ' + str(GSNR[j]) + ' < ' + str(service.snr_requirement)
                        wave_reason[j] = 2
                        allocation = False
                        break
                    else:
                        # Power[i+1] = Power_after_transmission
                        # print("power[i+1]:", Power[i+1])

                        # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                        for m in range(80):
                            if m != j and topology[u][v]['wavelength_service'][m] != 0:
                                tmp_service = service_dict.get(topology[u][v]['wavelength_service'][m], None)
                                # print('tmp_service:', tmp_service.snr_requirement, tmp_service.bit_rate, tmp_service.path)
                                # print(service_dict.keys())
                                # print('id:', topology[u][v]['wavelength_service'][m])
                                if tmp_service.snr_requirement >= GSNR[m]:
                                    # print('tmp_service:', tmp_service.service_id, tmp_service.path)
                                    allocation = False
                                    reason = 'interference!'
                                    wave_reason[j] = 3
                                    outer_break = True
                                    break

            path_iteration_end = time.time()
            # print(f"单条路径遍历耗时: {path_iteration_end - path_iteration_time:.6f} 秒")

            if allocation:
                # print('Path:', path.node_list, 'Wavelength:', j)
                service_dict[service.service_id] = service
                service.path = path.node_list
                service.wavelength = j
                for i in range(len(path.node_list) - 1):
                    u = path.node_list[i]
                    v = path.node_list[i + 1]
                    topology[u][v]['wavelength_power'] = Power[i]
                    topology[u][v]['wavelength_SNR'] = path_GSNR[i]
                    topology[u][v]['wavelength_bitrate'][j] = service.bit_rate
                    if topology[u][v]['wavelength_SNR'][j] > 24.6:
                        capacity = 900
                    elif topology[u][v]['wavelength_SNR'][j] > 21.6:
                        capacity = 750
                    elif topology[u][v]['wavelength_SNR'][j] > 18.6:
                        capacity = 600
                    elif topology[u][v]['wavelength_SNR'][j] > 16:
                        capacity = 450
                    elif topology[u][v]['wavelength_SNR'][j] > 12:
                        capacity = 300
                    else:
                        capacity = 150
                    # capacity = 800 if topology[u][v]['wavelength_SNR'][j] > 26.5 else 400
                    topology[u][v]['wavelength_utilization'][j] = bit_rate/capacity
                    total_utilization += bit_rate/capacity
                    topology[u][v]['wavelength_service'][j] = service.service_id
                    service.utilization = total_utilization / len(service.path)

                    # # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
                    for wave in range(80):
                        if wave != j and topology[u][v]['wavelength_service'][wave] != 0:
                            service_id = topology[u][v]['wavelength_service'][wave]
                            tmp_service = service_dict.get(service_id, None)
                            # print('tmp_service:', service_id)
                            if topology[u][v]['wavelength_SNR'][wave] > 24.6:
                                capacity = 900
                            elif topology[u][v]['wavelength_SNR'][wave] > 21.6:
                                capacity = 750
                            elif topology[u][v]['wavelength_SNR'][wave] > 18.6:
                                capacity = 600
                            elif topology[u][v]['wavelength_SNR'][wave] > 16:
                                capacity = 450
                            elif topology[u][v]['wavelength_SNR'][wave] > 12:
                                capacity = 300
                            else:
                                capacity = 150
                    #         # capacity = 800 if topology[u][v]['wavelength_SNR'][wave] > 26.5 else 400
                    #         # if tmp_service.bit_rate / capacity < 1:
                            topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity
                    #         if tmp_service.bit_rate / capacity > 1:
                    #             print('异常！！！！！！', tmp_service.service_id, tmp_service.snr_requirement, topology[u][v]['wavelength_SNR'][wave], tmp_service.bit_rate, capacity, tmp_service.path)

                return path.node_list, j, []

        path_end_time = time.time()
        # print(f"路径计算总耗时: {path_end_time - path_start_time:.6f} 秒")
    end_time = time.time()
    # print(f"函数总运行时间: {end_time - start_time:.6f} 秒")
    # print('分配失败', reason)
    return None, None, wave_reason

def select_sorting_services(service_dict, blocked_service, current_time, num_agent, upper_utilization):
    '''
    从service_dict中选出与blocked_service.path有重合链路的service，按照重合链路数排序，重合链路数多的排在前面。
    如果重合链路数相同，则按照带宽利用率和剩余时间（current_time - service.arrival_time）排序，带宽利用率低、剩余时间长的排在前面。
    :return: (dict) 排序后的service对象字典
    '''
    def path_to_links(path):
        """ 将路径节点序列转换为 **无向链路集合** """
        return {(min(int(path[i]), int(path[i + 1])), max(int(path[i]), int(path[i + 1])))
                for i in range(len(path) - 1)}

    overlapping_services = []
    blocked_links = path_to_links(blocked_service.path)  # 计算 blocked_service 的链路集合

    for service in service_dict.values():
        # 计算重合链路数
        service_links = path_to_links(service.path)  # 将 service 的路径转换为链路集合
        overlap_count = len(blocked_links & service_links)  # 计算与 blocked_service 的重合链路数

        if overlap_count > 0 and service.utilization < upper_utilization:
            remaining_time = current_time - service.arrival_time
            overlapping_services.append((service, service.utilization, overlap_count, remaining_time))
            # print('overlap:', service.service_id, service.utilization, overlap_count, remaining_time)

    # 先按重合链路数降序排序，再按剩余时间降序排序
    overlapping_services.sort(key=lambda x: (-x[2], x[1], -x[3]))

    # 返回排序后的服务对象列表
    return {s[0].service_id: s[0] for s in overlapping_services[:num_agent]}


def new_first_fit(topology, service:Service, service_dict):
    '''
    input: topology, service
    output: path_node_list, wavelength j
    非普通first-fit，是从前40个波长中随机选一个
    '''
    src = service.source_id
    dst = service.destination_id
    bit_rate = service.bit_rate
    allocation = False
    reason = None
    if bit_rate > 400:
        service.snr_requirement = 26.5
    else:
        service.snr_requirement = 18.2

    for path in topology.graph['ksp'][str(src), str(dst)]:
        start_wavelength = random.randint(0, 40 - 1)
        for offset in range(40):
            j = (start_wavelength + offset) % 40
            allocation = True
            outer_break = False
            reason = None
            Power = np.zeros((len(path.node_list), 80), dtype=float)
            path_GSNR = np.zeros((len(path.node_list), 80), dtype=float)
            # Power_after_transmission = np.zeros(80, dtype=float)
            for i in range((len(path.node_list)-1)):
                if outer_break:
                    break
                # print("i=",i)
                u = path.node_list[i]
                v = path.node_list[i+1]
                # original_powers.append(topology[u][v]['wavelength_power'].copy())  # 先保存原始功率
                # print('power:', topology[u][v]['wavelength_power'][j], type(topology[u][v]['wavelength_power'][j]))

                # 检查波长是否空闲
                if not (topology[u][v]['wavelength_power'][j] == 0 or (np.isnan(topology[u][v]['wavelength_power'][j]))):
                    reason = 'no free wavelength'
                    # print('power:', topology[u][v]['wavelength_power'][j])
                    allocation = False
                    break
                else:
                    # 检查SNR是否满足要求
                    # 获取链路参数
                    distance = topology[u][v]['length']
                    channels = 80
                    if i == 0:
                        Power[i] = copy.deepcopy(topology[u][v]['wavelength_power'])
                        Power[i][j] = service.power
                        # print('Power[i]:', Power[i])
                        # print("power:", Power)
                    frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
                    # if i != 0:
                    #     topology[u][v]['wavelength_power'] = Power_after_transmission
                    # 临时添加待分配业务的功率
                    # topology[u][v]['wavelength_power'][j] = service.power
                    tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
                    tmp[j] = service.power
                    tmp = np.array(tmp)
                    # print('tmp:', tmp)
                    if i != 0:
                        # print('unupdate_Power[i]:', Power[i])
                        Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                                         range(len(Power[i]))]
                        tmp = Power[i]
                        tmp = np.array(tmp)
                        # print('update_Power[i]:', Power[i])
                    # original_powers.append(topology[u][v]['wavelength_power'].copy())  # 保存原始功率

                    # 计算链路的GSNR
                    Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                    # print('power:', Power_after_transmission)
                    # print('GSNR:', GSNR)
                    path_GSNR[i] = GSNR
                    # topology[u][v]['wavelength_power'] = Power_after_transmission  # 更新功率
                    # 检查SNR是否满足要求

                    #if Decimal("{:.5f}".format(GSNR[j])) < Decimal(str(snr_requirement)):
                    if service.snr_requirement > GSNR[j]:
                        reason = 'GSNR < snr_requirement' +'  ' + str(GSNR[j]) + ' < ' + str(service.snr_requirement)
                        allocation = False
                        break
                    else:
                        Power[i+1] = Power_after_transmission
                        # print("power[i+1]:", Power[i+1])

                        # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                        for m in range(80):
                            if m != j and topology[u][v]['wavelength_service'][m] != 0:
                                tmp_service = service_dict.get(topology[u][v]['wavelength_service'][m], None)
                                # print('tmp_service:', tmp_service.snr_requirement, tmp_service.bit_rate, tmp_service.path)
                                # print(service_dict.keys())
                                # print('id:', topology[u][v]['wavelength_service'][m])
                                if tmp_service.snr_requirement >= GSNR[m]:
                                    # print('tmp_service:', tmp_service.service_id, tmp_service.path)
                                    allocation = False
                                    reason = 'interference!'
                                    outer_break = True
                                    break

            if allocation:
                # print('Path:', path.node_list, 'Wavelength:', j)
                service_dict[service.service_id] = service
                service.path = path.node_list
                service.wavelength = j
                for i in range(len(path.node_list) - 1):
                    u = path.node_list[i]
                    v = path.node_list[i + 1]
                    topology[u][v]['wavelength_power'] = Power[i]
                    topology[u][v]['wavelength_SNR'] = path_GSNR[i]
                    if topology[u][v]['wavelength_SNR'][j] > 26.5:
                        capacity = 800
                    elif topology[u][v]['wavelength_SNR'][j] > 18.2:
                        capacity = 400
                    else:
                        capacity = 200
                    # capacity = 800 if topology[u][v]['wavelength_SNR'][j] > 26.5 else 400
                    topology[u][v]['wavelength_utilization'][j] = bit_rate/capacity
                    topology[u][v]['wavelength_service'][j] = service.service_id

                    # # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
                    for wave in range(80):
                        if wave != j and topology[u][v]['wavelength_service'][wave] != 0:
                            service_id = topology[u][v]['wavelength_service'][wave]
                            tmp_service = service_dict.get(service_id, None)
                            # print('tmp_service:', service_id)
                            if topology[u][v]['wavelength_SNR'][wave] > 26.5:
                                capacity = 800
                            elif topology[u][v]['wavelength_SNR'][wave] > 18.2:
                                capacity = 400
                            else:
                                capacity = 200
                            # capacity = 800 if topology[u][v]['wavelength_SNR'][wave] > 26.5 else 400
                            # if tmp_service.bit_rate / capacity < 1:
                            topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity
                            if tmp_service.bit_rate / capacity > 1:
                                print('异常！！！！！！', tmp_service.service_id, tmp_service.snr_requirement, topology[u][v]['wavelength_SNR'][wave], tmp_service.bit_rate, tmp_service.path)

                return path.node_list, j

    # print('分配失败', reason)
    return None, None

def first_fit(topology, service:Service, service_dict):
    '''
    input: topology, service
    output: path_node_list, wavelength j
    '''
    src = service.source_id
    dst = service.destination_id
    bit_rate = service.bit_rate
    allocation = False
    reason = None
    if bit_rate > 400:
        service.snr_requirement = 26.5
    else:
        service.snr_requirement = 18.2

    for path in topology.graph['ksp'][str(src), str(dst)]:
        for j in range(80):
            allocation = True
            outer_break = False
            reason = None
            Power = np.zeros((len(path.node_list), 80), dtype=float)
            path_GSNR = np.zeros((len(path.node_list), 80), dtype=float)
            # Power_after_transmission = np.zeros(80, dtype=float)
            for i in range((len(path.node_list)-1)):
                if outer_break:
                    break
                # print("i=",i)
                u = path.node_list[i]
                v = path.node_list[i+1]
                # original_powers.append(topology[u][v]['wavelength_power'].copy())  # 先保存原始功率
                # print('power:', topology[u][v]['wavelength_power'][j], type(topology[u][v]['wavelength_power'][j]))

                # 检查波长是否空闲
                if not (topology[u][v]['wavelength_power'][j] == 0 or (np.isnan(topology[u][v]['wavelength_power'][j]))):
                    reason = 'no free wavelength'
                    # print('power:', topology[u][v]['wavelength_power'][j])
                    allocation = False
                    break
                else:
                    # 检查SNR是否满足要求
                    # 获取链路参数
                    distance = topology[u][v]['length']
                    channels = 80
                    if i == 0:
                        Power[i] = copy.deepcopy(topology[u][v]['wavelength_power'])
                        Power[i][j] = service.power
                        # print('Power[i]:', Power[i])
                        # print("power:", Power)
                    frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
                    # if i != 0:
                    #     topology[u][v]['wavelength_power'] = Power_after_transmission
                    # 临时添加待分配业务的功率
                    # topology[u][v]['wavelength_power'][j] = service.power
                    tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
                    tmp[j] = service.power
                    tmp = np.array(tmp)
                    # print('tmp:', tmp)
                    if i != 0:
                        # print('unupdate_Power[i]:', Power[i])
                        Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                                         range(len(Power[i]))]
                        tmp = Power[i]
                        tmp = np.array(tmp)
                        # print('update_Power[i]:', Power[i])
                    # original_powers.append(topology[u][v]['wavelength_power'].copy())  # 保存原始功率

                    # 计算链路的GSNR
                    start = time.time()

                    Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                    end = time.time()
                    print('one_link_trans:', end-start)
                    # print('power:', Power_after_transmission)
                    # print('GSNR:', GSNR)
                    path_GSNR[i] = GSNR
                    # topology[u][v]['wavelength_power'] = Power_after_transmission  # 更新功率
                    # 检查SNR是否满足要求

                    #if Decimal("{:.5f}".format(GSNR[j])) < Decimal(str(snr_requirement)):
                    if service.snr_requirement > GSNR[j]:
                        reason = 'GSNR < snr_requirement' +'  ' + str(GSNR[j]) + ' < ' + str(service.snr_requirement)
                        allocation = False
                        break
                    else:
                        Power[i+1] = Power_after_transmission
                        # print("power[i+1]:", Power[i+1])

                        # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                        for m in range(80):
                            if m != j and topology[u][v]['wavelength_service'][m] != 0:
                                tmp_service = service_dict.get(topology[u][v]['wavelength_service'][m], None)
                                # print('tmp_service:', tmp_service.snr_requirement, tmp_service.bit_rate, tmp_service.path)
                                # print(service_dict.keys())
                                # print('id:', topology[u][v]['wavelength_service'][m])
                                if tmp_service.snr_requirement >= GSNR[m]:
                                    # print('tmp_service:', tmp_service.service_id, tmp_service.path)
                                    allocation = False
                                    reason = 'interference!'
                                    outer_break = True
                                    break

            if allocation:
                # print('Path:', path.node_list, 'Wavelength:', j)
                service_dict[service.service_id] = service
                service.path = path.node_list
                service.wavelength = j
                for i in range(len(path.node_list) - 1):
                    u = path.node_list[i]
                    v = path.node_list[i + 1]
                    topology[u][v]['wavelength_power'] = Power[i]
                    topology[u][v]['wavelength_SNR'] = path_GSNR[i]
                    if topology[u][v]['wavelength_SNR'][j] > 26.5:
                        capacity = 800
                    elif topology[u][v]['wavelength_SNR'][j] > 18.2:
                        capacity = 400
                    else:
                        capacity = 200
                    # capacity = 800 if topology[u][v]['wavelength_SNR'][j] > 26.5 else 400
                    topology[u][v]['wavelength_utilization'][j] = bit_rate/capacity
                    topology[u][v]['wavelength_service'][j] = service.service_id

                    # # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
                    for wave in range(80):
                        if wave != j and topology[u][v]['wavelength_service'][wave] != 0:
                            service_id = topology[u][v]['wavelength_service'][wave]
                            tmp_service = service_dict.get(service_id, None)
                            # print('tmp_service:', service_id)
                            if topology[u][v]['wavelength_SNR'][wave] > 26.5:
                                capacity = 800
                            elif topology[u][v]['wavelength_SNR'][wave] > 18.2:
                                capacity = 400
                            else:
                                capacity = 200
                            # capacity = 800 if topology[u][v]['wavelength_SNR'][wave] > 26.5 else 400
                            # if tmp_service.bit_rate / capacity < 1:
                            topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity
                            if tmp_service.bit_rate / capacity > 1:
                                print('异常！！！！！！', tmp_service.service_id, tmp_service.snr_requirement, topology[u][v]['wavelength_SNR'][wave], tmp_service.bit_rate, tmp_service.path)

                return path.node_list, j

    # print('分配失败', reason)
    return None, None

def dummy_first_fit(topology, service:Service, service_dict, GSNRs):
    '''
    input: topology, service
    output: path_node_list, wavelength j
    '''
    src = service.source_id
    dst = service.destination_id
    bit_rate = service.bit_rate
    if bit_rate > 400:
        service.snr_requirement = 26.5
    else:
        service.snr_requirement = 18.2

    for path in topology.graph['ksp'][str(src), str(dst)]:
        for j in range(80):
            allocation = True
            outer_break = False
            # Power_after_transmission = np.zeros(80, dtype=float)
            for i in range((len(path.node_list) - 1)):
                if outer_break:
                    break
                # print("i=",i)
                u = path.node_list[i]
                v = path.node_list[i + 1]
                # original_powers.append(topology[u][v]['wavelength_power'].copy())  # 先保存原始功率
                # print('power:', topology[u][v]['wavelength_power'][j], type(topology[u][v]['wavelength_power'][j]))

                # 检查波长是否空闲
                if not topology[u][v]['wavelength_service'][j] == -1:
                    reason = 'no free wavelength'
                    # print('power:', topology[u][v]['wavelength_power'][j])
                    allocation = False
                    break

                    # if Decimal("{:.5f}".format(GSNR[j])) < Decimal(str(snr_requirement)):
                if service.snr_requirement > GSNRs[j]:
                    reason = 'GSNR < snr_requirement' + '  ' + str(GSNRs[j]) + ' < ' + str(service.snr_requirement)
                    allocation = False
                    break

            if allocation:
                # print('Path:', path.node_list, 'Wavelength:', j)
                service_dict[service.service_id] = service
                service.path = path.node_list
                service.wavelength = j
                for i in range(len(path.node_list) - 1):
                    u = path.node_list[i]
                    v = path.node_list[i + 1]
                    topology[u][v]['wavelength_service'][j] = service.service_id

                return path.node_list, j

    # print('分配失败', reason)
    return None, None


def check_utilization(topology, path, service_dict):
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
        for wave in range(80):
            if topology[u][v]['wavelength_utilization'][wave] > 1:
                service_released = service_dict.get(topology[u][v]['wavelength_service'][wave], None)
                release_service(topology, service_released, service_dict)

# 4. release a service
def release_service(topology, service:Service, service_dict):
    '''
    service_dict: 业务字典，键为service_id，值为service对象
    释放指定的某个业务，并更新相关链路的状态
    '''
    # 更新链路状态
    del service_dict[service.service_id]
    for i in range(len(service.path) - 1):
        u = service.path[i]
        v = service.path[i + 1]
        # 清除波长上的业务信息
        topology[u][v]['wavelength_power'][service.wavelength] = 0
        topology[u][v]['wavelength_utilization'][service.wavelength] = 0
        topology[u][v]['wavelength_SNR'][service.wavelength] = 0
        topology[u][v]['wavelength_service'][service.wavelength] = 0
        topology[u][v]['wavelength_bitrate'][service.wavelength] = 0

        # 重新计算链路状态（例如更新GSNR）
        distance = topology[u][v]['length']
        channels = 80
        # Power = [0.0031622776602 if not(power == 0 or np.isnan(power)) else 0 for power in topology[u][v]['wavelength_power']]
        # print("vs power:", topology[u][v]['wavelength_power'], Power)
        # if i == 0:
        #     topology[u][v]['wavelength_power'] = Power
        #     tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
        #     tmp = np.array(tmp)
        # else:
        #     topology[u][v]['wavelength_power'] = [
        #         0 if topology[u][v]['wavelength_power'][j] == 0 and Power_after_transmission[j] != 0
        #         else (topology[u][v]['wavelength_power'][j] if Power_after_transmission[j] == 0 and topology[u][v]['wavelength_power'][j] != 0
        #               else Power_after_transmission[j])
        #         for j in range(len(topology[u][v]['wavelength_power']))]
        #     tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
        #     tmp = np.array(tmp)
        tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
        tmp = np.array(tmp)

        # # print("Power:",Power)
        # # print('tmp_power:', tmp)
        # # print('all_0:', all(x == 0 for x in tmp))
        if not all(x == 0 for x in tmp):
            frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2),
                                          np.linspace(190.75e12, 196.6e12, channels // 2)])

            Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
            # print('power_after_transmission:', Power_after_transmission)
            topology[u][v]['wavelength_SNR'] = GSNR

        # 更新该链路所有传输载波的利用率
        for j in range(channels):
            if topology[u][v]['wavelength_service'][j] != 0:
                service_id = topology[u][v]['wavelength_service'][j]
                tmp_service1 = service_dict.get(int(service_id), None)
                if tmp_service1==None:
                    print('id:', j, service_id)
                    print('ids:', service_dict.keys())
                # capacity = 800 if topology[u][v]['wavelength_SNR'][j] > 26.5 else 400
                if topology[u][v]['wavelength_SNR'][j] > 24.6:
                    capacity = 900
                elif topology[u][v]['wavelength_SNR'][j] > 21.6:
                    capacity = 750
                elif topology[u][v]['wavelength_SNR'][j] > 18.6:
                    capacity = 600
                elif topology[u][v]['wavelength_SNR'][j] > 16:
                    capacity = 450
                elif topology[u][v]['wavelength_SNR'][j] > 12:
                    capacity = 300
                else:
                    capacity = 150
                topology[u][v]['wavelength_utilization'][j] = tmp_service1.bit_rate / capacity
                # print('service_id:', tmp_service.service_id, 'bitrate:', bit_rate, 'capacity:', capacity)
            else:
                topology[u][v]['wavelength_power'][j] = 0
                topology[u][v]['wavelength_utilization'][j] = 0
                topology[u][v]['wavelength_SNR'][j] = 0
                topology[u][v]['wavelength_bitrate'][j] = 0

    # # 打印释放后的链路状态
    # print("释放后的链路状态:")
    # for i in range(len(service.path) - 1):
    #     u = service.path[i]
    #     v = service.path[i + 1]
    #     print(f"链路 {u}-{v}: {topology[u][v]['wavelength_SNR']}")

    # 从拓扑中移除业务
    # for _ in service_list_after_release:
    #     if _.service_id == service.service_id:
    #         service_list_after_release.remove(_)

def dummy_release_service(topology, service:Service, service_dict):
    '''
    service_dict: 业务字典，键为service_id，值为service对象
    释放指定的某个业务，并更新相关链路的状态
    '''
    # print('service_id:', service.service_id)
    # 更新链路状态
    del service_dict[service.service_id]
    for i in range(len(service.path) - 1):
        u = service.path[i]
        v = service.path[i + 1]
        # 清除波长上的业务信息
        topology[u][v]['wavelength_power'][service.wavelength] = 0
        topology[u][v]['wavelength_utilization'][service.wavelength] = 0
        topology[u][v]['wavelength_SNR'][service.wavelength] = 0
        topology[u][v]['wavelength_service'][service.wavelength] = -1

def recalculate_all_gsnr(topology, service_list):
    for u, v, attrs in topology.edges(data=True):
        # 计算链路上每个波长的GSNR
        distance = attrs['length']
        channels = 80
        Power = attrs['wavelength_power'].copy()  # 假设这是当前波长的功率数组
        frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2),
                                      np.linspace(190.75e12, 196.6e12, channels // 2)])

        # 使用one_link_transmission函数重新计算GSNR
        Power_after_transmission, GSNR = one_link_transmission(distance, channels, Power, frequencies)
        attrs['wavelength_SNR'] = GSNR  # 更新链路的GSNR值

    # 对于每个业务，更新其路径上的GSNR值
    for service in service_list:
        min_gsnr = float('inf')  # 假设一个很大的初始值
        for i in range(len(service.path) - 1):
            u = service.path[i]
            v = service.path[i + 1]
            gsnr = topology[u][v]['wavelength_SNR'][service.wavelength]
            min_gsnr = min(min_gsnr, gsnr)  # 找到路径上最小的GSNR值
        service.GSNR = min_gsnr  # 更新业务的GSNR值
