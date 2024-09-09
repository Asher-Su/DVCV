import torch.optim as optim
import torch.nn as nn 
import torch
import logging
import scipy.stats as stats
import math
from utils import *
import numpy as np

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
    Baruch G, Baruch M, Goldberg Y. A little is enough: Circumventing defenses for distributed learning[J]. Advances in Neural Information Processing Systems, 2019, 32.
    AGR: fedavg
"""
def poison_lie(args, nets_this_round_after_train):
    s = math.floor(args.num_client/2 + 1) - args.num_client * args.mali_radio
    cdf_value = (args.num_client - args.num_client * args.mali_radio - s)/(args.num_client - args.num_client * args.mali_radio)
    z_max = stats.norm.ppf(cdf_value,0,1)
    mean_model = init_nets(args,1)[0]
    mean_model_w = mean_model.state_dict()
    for net_i,net in enumerate(nets_this_round_after_train.values()):
        net_para = net.state_dict()
        if net_i == 0:
            for key in net_para:
                mean_model_w[key] = net_para[key]
        else:
            for key in net_para:
                mean_model_w[key] += net_para[key]
    for key in mean_model_w:
        mean_model_w[key] /= len(nets_this_round_after_train)
    deviation = get_deviation(args,nets_this_round_after_train)
    for key in mean_model_w:
        mean_model_w[key] -= z_max * deviation
    return mean_model_w



"""
    Fang M, Cao X, Jia J, et al. Local model poisoning attacks to {Byzantine-Robust} federated learning[C]//29th USENIX security symposium (USENIX Security 20). 2020: 1605-1622.
    AGR: fedavg
"""
def poison_fang_avg(args, net_global, nets_this_round_after_train, net_dataidx_map,num_attacker_index):
    approximate_previous = 0 #保存上一轮次二分法获得的优化函数的结果,保存此次攻击最终的结果
    total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.num_client)])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.num_client)]
    w_Re_w = net_global.state_dict()
    w = init_nets(args,1)[0]
    w_w = w.state_dict()
    w_pie = init_nets(args,1)[0]
    w_pie_w = w_pie.state_dict()
    s = init_nets(args, 1)[0]
    s_w = s.state_dict()
    attack_result = init_nets(args,1)[0]
    attack_result_w = attack_result.state_dict()
    for net_i,net in enumerate(nets_this_round_after_train.values()):
        net_para = net.state_dict()
        if net_i == 0:
            for key in net_para:
                w_w[key] = net_para[key] * fed_avg_freqs[net_i]
        else:
            for key in net_para:
                w_w[key] += net_para[key] * fed_avg_freqs[net_i]
    # 获取s向量的值
    for key in s_w:
        temp = w_w[key] - w_Re_w[key]
        temp = temp.cpu()
        temp = np.sign(temp)
        temp = temp.to(args.gpu)
        s_w[key] = temp
    left,lambdaa = 0,0 # 二分法的左界点
    right = args.lambdaa
    while left <= right:
        lambdaa = (left+right)/2
        # 获取攻击者模型梯度
        for key in w_pie_w:
            w_pie_w[key] = w_Re_w[key] - lambdaa * s_w[key]
        w_pie.load_state_dict(w_pie_w)
        for i in range(int(num_attacker_index),args.num_client):
            nets_this_round_after_train[i] = w_pie
        # 发起攻击
        for net_i, net in enumerate(nets_this_round_after_train.values()):
            net_para = net.state_dict()
            if net_i == 0:
                for key in net_para:
                    w_pie_w[key] = net_para[key] * fed_avg_freqs[net_i]
            else:
                for key in net_para:
                    w_pie_w[key] += net_para[key] * fed_avg_freqs[net_i]
        for key in attack_result_w:
            attack_result_w[key] = (w_w[key] - w_pie_w[key]) * s_w[key]
        attack_result.load_state_dict(attack_result_w)

        result_sum = torch.tensor(0.0, dtype=torch.float, device='cuda:0')
        for param in attack_result.parameters():
            param_sum = torch.sum(param.data)
            result_sum += param_sum
        if torch.sum(result_sum) > approximate_previous:
            left = lambdaa + 1
            approximate_previous = torch.sum(result_sum)
        elif torch.sum(result_sum) < approximate_previous:
            right = lambdaa - 1
        else:
            break
    if lambdaa == 0:
        return w_Re_w
    elif lambdaa > 0:
        for key in w_pie_w:
            w_pie_w[key] = w_Re_w[key] - lambdaa * s_w[key]
        return w_pie_w




"""
    Shejwalkar V, Houmansadr A. Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning[C]//NDSS. 2021.
    AGR: fedavg
"""
def poison_agr_updates_avg(args, nets_this_round_after_train, net_dataidx_map,num_attacker_index):
    total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.num_client)])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.num_client)]
    der_b = init_nets(args,1)[0]
    der_b_w = der_b.state_dict()
    der_p = init_nets(args,1)[0]
    der_p_w = der_p.state_dict()
    der_m = init_nets(args, 1)[0]
    der_m_w = der_m.state_dict()
    substract_m = init_nets(args,1)[0]
    substract_m_w = substract_m.state_dict()
    # 获取derta_b的参数
    for net_i,net in enumerate(nets_this_round_after_train.values()):
        net_para = net.state_dict()
        if net_i == 0:
            for key in net_para:
                der_b_w[key] = net_para[key] * fed_avg_freqs[net_i]
        else:
            for key in net_para:
                der_b_w[key] += net_para[key] * fed_avg_freqs[net_i]
    if args.perturb_type == 'iuv':
        der_b.load_state_dict(der_b_w)
        l2_norm = calculate_model_l2_norm(der_b)
        for key in der_b_w:
            der_p_w[key] = (-1.0 * der_b_w[key])/l2_norm
    elif args.perturb_type == 'is':
        for key in der_b_w:
            if torch.cuda.is_available():
                tensor_temp = der_b_w[key].cpu()
                tensor_temp =  -1.0 * np.sign(tensor_temp)
                tensor_temp = tensor_temp.to(args.gpu)
                der_b_w[key] = der_b_w[key].to(args.gpu)
                der_p_w[key] = tensor_temp
            else:
                der_p_w[key] = -1.0 * np.sign(der_b_w[key])
    elif args.perturb_type == 'isd':
        deviation = get_deviation(args,nets_this_round_after_train)
        for key in der_b_w:
            der_p_w[key] = -1.0 * deviation
    #进行r优化过程
    r_succ, r_init = 0, 100
    step, r, prev= r_init/2, r_init, -1
    while abs(r_succ - r) > args.tao:
        t, flag, der_m_w = OutPut(args, prev, r, der_b_w, der_p_w, der_m_w, substract_m, substract_m_w, nets_this_round_after_train,
               fed_avg_freqs, num_attacker_index)
        if flag:
            r_succ = r
            r = r + step/2
            prev = t
        else :
            r = r - step/2
        step = step/2
    return der_m_w

def get_deviation(args, nets_this_round_after_train):
    mali_updates=[]
    for net_i in range(0,args.num_client):
        params = []
        for i,(name, param) in enumerate(nets_this_round_after_train[net_i].state_dict().items()):
            params = param.view(-1).data.type(torch.FloatTensor) if len(params) == 0 else torch.cat(
                (params, param.view(-1).data.type(torch.FloatTensor)))
        mali_updates = params[None, :] if len(mali_updates) == 0 else torch.cat((mali_updates, params[None, :]), 0)
    deviation = torch.std(mali_updates)
    return deviation


"""计算模型的l2范式"""
def calculate_model_l2_norm(model):
    l2_norm_squared = torch.tensor(0.0, dtype=torch.float,device='cuda:0')
    for param in model.parameters():
        param_norm_squared = torch.sum(torch.square(param.data))
        if torch.cuda.is_available():
            param_norm_squared.cuda()
        else:
            param_norm_squared.cpu()
        l2_norm_squared += param_norm_squared
    return torch.sqrt(l2_norm_squared).item()


def OutPut(args,prev,r,der_b_w,der_p_w,der_m_w,substract_m,substract_m_w,nets_this_round_after_train,fed_avg_freqs,num_attacker_index):
    for key in der_b_w:
        der_m_w[key] = der_b_w[key] + r * der_p_w[key]
    temp_avg = init_nets(args,1)[0]
    temp_avg_w = temp_avg.state_dict()
    for net_i,net in enumerate(nets_this_round_after_train.values()):
        net_para = net.state_dict()
        if net_i == 0:
            for key in net_para:
                temp_avg_w[key] = net_para[key] * fed_avg_freqs[net_i]
        else:
            if net_i < num_attacker_index:
                for key in net_para:
                    temp_avg_w[key] += net_para[key] * fed_avg_freqs[net_i]
            else:
                for key in net_para:
                    temp_avg_w[key] += der_m_w[key] * fed_avg_freqs[net_i]
    for key in der_b_w:
        substract_m_w[key] = der_b_w[key] - temp_avg_w[key]
    substract_m.load_state_dict(substract_m_w)
    if calculate_model_l2_norm(substract_m) > prev:
        prev = calculate_model_l2_norm(substract_m)
        return prev,True,der_m_w
    else:
        return None,False,der_m_w
