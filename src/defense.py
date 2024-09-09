import random
from torch.utils.data import Subset
import math
from utils import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


"""
    Our Defense Method
"""
def Our_defense(args, sub_models, sub_models_map,net_global, net_dataidx_map, total_data_points, client_classes,vote_weight,round,num_attacker_index):
    model_list = []
    logger.info(">>> Client-classes: ")
    logger.info(str(client_classes))
    for i in range(len(sub_models)):
        # 每个sub_model i进行验证
        logger.info(">>> Ours-Defense sub-model %d:" % i)
        votes, votes_history = [0 for i in range(args.num_classes)],[[0 for i in range(args.class_evaluate_num)] for j in range(args.num_client)]
        for j in range(args.num_client): # 第j个client用于进行评估
            idx, noise = net_dataidx_map[j], args.noise * j
            for k in range(args.class_evaluate_num): # 统计每个client的判断结果
                logger.info('>>> sub-model %d, evaluate-client %d, evaluate-class:%d ' %(i,j,client_classes[j][k]))
                _, _, data_dl_test_ds, _ = get_dataloader_local(args, noise, idx)
                data_dl_test_local_class = Subset(data_dl_test_ds,
                                                  indices=[la for la, sample in enumerate(data_dl_test_ds) if
                                                           is_label_xxx(sample, client_classes[j][k])])
                data_dl_test_local_class_loader = data.DataLoader(dataset=data_dl_test_local_class,
                                                                  batch_size=args.batch_size, shuffle=False)
                pre_global_acc, _ = compute_accuracy(args, net_global,
                                                     data_dl_test_local_class_loader)
                sub_model_acc, _ = compute_accuracy(args, sub_models[i],
                                                    data_dl_test_local_class_loader)
                logger.info('pre_global_ac %f,sub_model_ac %f' %(pre_global_acc,sub_model_acc))
                if (args.is_adaptive==0) or (args.is_adaptive==1 and round < args.adapvote_fromepoch):
                    if pre_global_acc > sub_model_acc:
                        votes[client_classes[j][k]] -= vote_weight[j][k]
                        votes_history[j][k] = -1
                    elif pre_global_acc < sub_model_acc:
                        votes[client_classes[j][k]] += vote_weight[j][k]
                        votes_history[j][k] = 1
                else:
                    if j < num_attacker_index:
                        if pre_global_acc > sub_model_acc:
                            votes[client_classes[j][k]] -= vote_weight[j][k]
                            votes_history[j][k] = -1
                        elif pre_global_acc < sub_model_acc:
                            votes[client_classes[j][k]] += vote_weight[j][k]
                            votes_history[j][k] = 1
                    else:
                        if pre_global_acc > sub_model_acc:
                            votes[client_classes[j][k]] += vote_weight[j][k]
                            votes_history[j][k] = 1
                        elif pre_global_acc < sub_model_acc:
                            votes[client_classes[j][k]] -= vote_weight[j][k]
                            votes_history[j][k] = -1
        logger.info(">>> Votes history: ")
        logger.info(str(votes_history))
        logger.info(">>> Votes result: "+str(votes))
        if args.num_client % args.dd == 0:
            nu = int(args.num_client / args.dd)
            # 根据判断结果进行投票权重再赋予
            vote_count_pos,vote_count_neg = 0,0
            for j in range(len(votes)):
                if votes[j] > 0:
                    vote_count_pos+=1
                elif votes[j] < 0:
                    vote_count_neg+=1
            if vote_count_pos > vote_count_neg:
                model_list.append(sub_models[i])
                logger.info('>>> vote_count_pos %d vote_count_neg %d sub-model %d yes!' %(vote_count_pos,vote_count_neg,i))
                for j in range(nu *i, nu*(i+1)):
                    for k in range(args.class_evaluate_num):
                        vote_weight[sub_models_map[j]][k] += (math.log2(args.v2+(round/args.epochs)+1))/(math.log2(args.v2+3)+sqrt(args.class_evaluate_num/args.num_classes))       ###############################替换权重变更算法1
            else:
                logger.info('>>> vote_count_pos %d vote_count_neg %d sub-model %d No!' % (vote_count_pos,vote_count_neg,i))
                for j in range(nu *i, nu*(i+1)):
                    for k in range(args.class_evaluate_num):
                        vote_weight[sub_models_map[j]][k] -= (math.log2(args.v2+(round/args.epochs)+1))/(math.log2(args.v2+3)+sqrt(args.class_evaluate_num/args.num_classes))        ###############################替换权重变更算法1
                        if vote_weight[sub_models_map[j]][k] <0:
                            vote_weight[sub_models_map[j]][k] = 0

            for j in range(len(votes)):
                if votes[j] >= 0:
                    for p in range(args.num_client):
                        for q in range(args.class_evaluate_num):
                            if client_classes[p][q] == j and votes_history[p][q] == 1:
                                vote_weight[p][q] += (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))               ###############################替换权重变更算法2
                            elif client_classes[p][q] == j and votes_history[p][q] == -1:
                                vote_weight[p][q] -= (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))
                                if vote_weight[p][q]<0:
                                    vote_weight[p][q] = 0
                else:
                    for p in range(args.num_client):
                        for q in range(args.class_evaluate_num):
                            if client_classes[p][q] == j and votes_history[p][q] == 1:
                                vote_weight[p][q] -= (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))                ###############################替换权重变更算法2
                                if vote_weight[p][q]<0:
                                    vote_weight[p][q] = 0
                            elif client_classes[p][q] == j and votes_history[p][q] == -1:
                                vote_weight[p][q] += (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))
            logger.info("After defense the vote weight is: "+str(vote_weight))
        elif args.num_client % args.dd != 0:
            nu= int((args.num_client - 1) / (args.dd - 1))
            remain = args.num_client - nu * (args.dd - 1)
            # 根据判断结果进行投票权重再赋予
            vote_count_pos,vote_count_neg = 0,0
            for j in range(len(votes)):
                if votes[j] > 0:
                    vote_count_pos += 1
                elif votes[j] < 0:
                    vote_count_neg += 1
            if vote_count_pos > vote_count_neg:
                model_list.append(sub_models[i])
                logger.info('>>> vote_count_pos %d vote_count_neg %d sub-model %d yes!' %(vote_count_pos,vote_count_neg,i))
                if i==0:
                    for j in range(0, remain):
                        for k in range(args.class_evaluate_num):
                            vote_weight[sub_models_map[j]][k] += (math.log2(args.v2+(round/args.epochs)+1))/(math.log2(args.v2+3)+sqrt(args.class_evaluate_num/args.num_classes))  ###############################替换权重变更算法1
                else:
                    for j in range(remain+(i-1)*nu, remain+i*nu):
                        for k in range(args.class_evaluate_num):
                            vote_weight[sub_models_map[j]][k] += (math.log2(args.v2+(round/args.epochs)+1))/(math.log2(args.v2+3)+sqrt(args.class_evaluate_num/args.num_classes))  ###############################替换权重变更算法1
            else:
                logger.info('>>> vote_count_pos %d vote_count_neg %d sub-model %d No!' % (vote_count_pos,vote_count_neg,i))
                if i==0:
                    for j in range(0, remain):
                        for k in range(args.class_evaluate_num):
                            vote_weight[sub_models_map[j]][k] -= (math.log2(args.v2+(round/args.epochs)+1))/(math.log2(args.v2+3)+sqrt(args.class_evaluate_num/args.num_classes))  ###############################替换权重变更算法1
                            if vote_weight[sub_models_map[j]][k]<0:
                                vote_weight[sub_models_map[j]][k] = 0
                else:
                    for j in range(remain+(i-1)*nu, remain+i*nu):
                        for k in range(args.class_evaluate_num):
                            vote_weight[sub_models_map[j]][k] -= (math.log2(args.v2+(round/args.epochs)+1))/(math.log2(args.v2+3)+sqrt(args.class_evaluate_num/args.num_classes))  ###############################替换权重变更算法1
                            if vote_weight[sub_models_map[j]][k]<0:
                                vote_weight[sub_models_map[j]][k] = 0
            for j in range(len(votes)):
                if votes[j] > 0:
                    for p in range(args.num_client):
                        for q in range(args.class_evaluate_num):
                            if client_classes[p][q] == j and votes_history[p][q] == 1:
                                vote_weight[p][q] += (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))  ###############################替换权重变更算法2
                            elif client_classes[p][q] == j and votes_history[p][q] == -1:
                                vote_weight[p][q] -= (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))
                                if vote_weight[p][q]<0:
                                    vote_weight[p][q]=0
                elif votes[j] < 0:
                    for p in range(args.num_client):
                        for q in range(args.class_evaluate_num):
                            if client_classes[p][q] == j and votes_history[p][q] == 1:
                                vote_weight[p][q] -= (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))  ###############################替换权重变更算法2
                                if vote_weight[p][q]<0:
                                    vote_weight[p][q]=0
                            elif client_classes[p][q] == j and votes_history[p][q] == -1:
                                vote_weight[p][q] += (math.log2(args.v1+(round/args.epochs)+1))/(math.log2(args.v1+3)+sqrt(args.class_evaluate_num/args.num_classes))
            logger.info("After defense the vote weight is: " + str(vote_weight))
    global_net_w = net_global.state_dict()
    if len(model_list)>0:
        for i in range(len(model_list)):
            net_para = model_list[i].state_dict()
            if i == 0:
                for key in net_para:
                    global_net_w[key] = net_para[key] * (1 / len(model_list))
            else:
                for key in net_para:
                    global_net_w[key] += net_para[key] * (1 / len(model_list))
    return global_net_w,vote_weight


"""
    Zhao L, Hu S, Wang Q, et al. Shielding collaborative learning: Mitigating poisoning attacks through client-side detection[J]. IEEE Transactions on Dependable and Secure Computing, 2020, 18(5): 2029-2041.
"""
# 实现TDSC中在iid和noiid的防御方法
def TDSC_defense_iid(args,sub_models,sub_models_map,net_global,net_dataidx_map,total_data_points,round,num_attacker_index):
    c_list= []
    if args.num_client % args.d == 0:
        nu, m_client= int(args.num_client / args.d), [args.m for i in range(args.num_client)]
        pre_global_acc,sub_model_acc = [],[]
        for i in range(len(sub_models)): # 每个sub_model进行验证
            logger.info("IID-Defense sub-model %d:"%i)
            for j in range(args.e): # 对于每个sub_model需要有多少client进行验证
                evaluate_client = random.randint(0,args.num_client-1)
                index,flag = i*nu, 0
                while index < (i+1)*nu:
                    if sub_models_map[index] == evaluate_client:
                        flag = 1
                    index+=1
                if flag==0 and m_client[evaluate_client] > 0:
                    m_client[evaluate_client]-=1
                else:
                    t = random.randint(0,args.num_client-1)
                    while t == evaluate_client:
                        t = random.randint(0, args.num_client - 1)
                    evaluate_client = t
                    m_client[evaluate_client]-=1
                logger.info('>>> evaluate_client: %d...'%evaluate_client)
                # 求验证client的测试集
                idx,noise = net_dataidx_map[evaluate_client], args.noise*evaluate_client
                data_dl_test_local, _, _, _ = get_dataloader_local(args, noise, idx)
                pre_global_ac,_ = compute_accuracy(args,net_global,data_dl_test_local) # 对上一个global_net进行准确性验证
                sub_model_ac,_ = compute_accuracy(args,sub_models[i],data_dl_test_local) # 对sub_model的准确性进行验证
                if (args.is_adaptive==0) or (args.is_adaptive==1 and round < args.adapvote_fromepoch and evaluate_client < num_attacker_index) or (args.is_adaptive==1 and round >= args.adapvote_fromepoch and evaluate_client < num_attacker_index):
                    pre_global_acc.append(pre_global_ac)
                    sub_model_acc.append(sub_model_ac)
                else:
                    pre_global_acc.append(sub_model_ac)
                    sub_model_acc.append(pre_global_ac)
                logger.info(
                    '>>> sub-model %d: pre_global_ac %f,sub_model_ac %f ' % (i, pre_global_ac, sub_model_ac))
        mean_de_ac = [sub_model_acc[k]-pre_global_acc[k] for k in range(args.e * len(sub_models)) if pre_global_acc[k] < sub_model_acc[k]]
        if len(mean_de_ac) > 0:
            mean_de = sum(mean_de_ac)/len(mean_de_ac)
        else :
            mean_de = 0
        for i in range(len(sub_models)):
            vote_mali = 0
            for j in range(args.e):
                if sub_model_acc[i*args.e + j] - pre_global_acc[i*args.e + j] <= mean_de :
                    vote_mali+=1
            logger.info('>>> sub-model %d: vote_mali %d' %(i,vote_mali))
            c = max(0,(-4*args.v*vote_mali*vote_mali)/((args.e-2)*(args.e-2))+(8*args.v*vote_mali)/((args.e-2)*(args.e-2))-(4*args.v)/((args.e-2)*(args.e-2))+args.v)
            c_list.append(c)
        logger.info(">>> The coefficient parameter to sub-model: "+str(c_list))
        global_net = net_global
        global_net_w = global_net.state_dict()
        if sum(c_list) !=0:
            for i in range(len(sub_models)):
                net_para = sub_models[i].state_dict()
                if i == 0:
                    for key in net_para:
                        # global_net_w[key] = net_para[key] * total_data_points[i]/All_data_points * c_list[i]
                        global_net_w[key] = net_para[key]  * (c_list[i]/sum(c_list))
                else:
                    for key in net_para:
                        global_net_w[key] += net_para[key]  * (c_list[i]/sum(c_list))
    else:
        nu = int((args.num_client - 1) / (args.d - 1))
        remain = args.num_client - nu * (args.d - 1)
        m_client = [args.m for i in range(args.num_client)]
        pre_global_acc,sub_model_acc = [],[]
        for i in range(len(sub_models)):  # 每个sub_model进行验证
            logger.info("IID-Defense sub-model %d:" % i)
            vote_mali = 0
            for j in range(args.e):  # 对于每个sub_model需要有多少client进行验证
                evaluate_client = random.randint(0, args.num_client - 1)
                if i == 0:
                    index, flag = 0, 0
                else:
                    index, flag = remain +(i-1)*nu,0
                if i==0:
                    while index < remain:
                        if sub_models_map[index] == evaluate_client:
                            flag = 1
                        index += 1
                else:
                    while index < remain + i*nu:
                        if sub_models_map[index] == evaluate_client:
                            flag = 1
                        index += 1
                if flag == 0 and m_client[evaluate_client] > 0:
                    m_client[evaluate_client] -= 1
                else:
                    t = random.randint(0, args.num_client - 1)
                    while t == evaluate_client:
                        t = random.randint(0, args.num_client - 1)
                    evaluate_client = t
                    m_client[evaluate_client] -= 1
                logger.info('>>> evaluate_client: %d...' % evaluate_client)
                # 求验证client的测试集
                idx, noise = net_dataidx_map[evaluate_client], args.noise * evaluate_client
                data_dl_test_local, _, _, _ = get_dataloader_local(args, noise, idx)
                pre_global_ac, _ = compute_accuracy(args, net_global, data_dl_test_local)  # 对上一个global_net进行准确性验证
                sub_model_ac, _ = compute_accuracy(args, sub_models[i], data_dl_test_local)  # 对sub_model的准确性进行验证
                if (args.is_adaptive==0) or (args.is_adaptive==1 and round < args.adapvote_fromepoch and evaluate_client < num_attacker_index) or (args.is_adaptive==1 and round >= args.adapvote_fromepoch and evaluate_client < num_attacker_index):
                    pre_global_acc.append(pre_global_ac)
                    sub_model_acc.append(sub_model_ac)
                else:
                    pre_global_acc.append(sub_model_ac)
                    sub_model_acc.append(pre_global_ac)
                logger.info('>>> sub-model %d: pre_global_ac %f,sub_model_ac %f ' %(i,pre_global_ac,sub_model_ac))
        mean_de_ac = [sub_model_acc[k]-pre_global_acc[k] for k in range(args.e * len(sub_models)) if pre_global_acc[k] < sub_model_acc[k]]
        if len(mean_de_ac) > 0:
            mean_de = sum(mean_de_ac)/len(mean_de_ac)
        else :
            mean_de = 0
        for i in range(len(sub_models)):
            vote_mali = 0
            for j in range(args.e):
                if sub_model_acc[i*args.e + j] - pre_global_acc[i*args.e + j] <= mean_de :
                    vote_mali+=1
            logger.info('>>> sub-model %d: vote_mali %d' %(i,vote_mali))
            c = max(0,(-4*args.v*vote_mali*vote_mali)/((args.e-2)*(args.e-2))+(8*args.v*vote_mali)/((args.e-2)*(args.e-2))-(4*args.v)/((args.e-2)*(args.e-2))+args.v)
            c_list.append(c)
        logger.info(">>> The coefficient parameter to sub-model: " + str(c_list))
        global_net =  net_global
        global_net_w = global_net.state_dict()
        if sum(c_list) !=0:
            for i in range(len(sub_models)):
                net_para = sub_models[i].state_dict()
                if i == 0:
                    for key in net_para:
                        # global_net_w[key] = net_para[key] * total_data_points[i]/All_data_points * c_list[i]
                        global_net_w[key] = net_para[key]  * (c_list[i]/sum(c_list))
                else:
                    for key in net_para:
                        global_net_w[key] += net_para[key]  * (c_list[i]/sum(c_list))

    return global_net_w

def TDSC_defense_noiid(args, sub_models, sub_models_map, net_global, net_dataidx_map, total_data_points,client_classes,d,round,num_attacker_index):
    c_list = []
    if args.num_client % d == 0:
        nu, m_client = int(args.num_client / d), [args.m for i in range(args.num_client)]
        for i in range(len(sub_models)):  # 每个sub_model进行验证
            logger.info("NOIID-Defense sub-model %d:" % i)
            vote_mali = 0
            for j in range(args.num_classes):  # 对于每个sub_model需要有多少client进行验证
                evaluate_client_list_1 = [row[j] for row in client_classes]
                evaluate_client_list = [inx for inx,value in enumerate(evaluate_client_list_1) if value == 1]
                evaluate_client = random.choice(evaluate_client_list)
                index, flag = i * nu, 0
                while index < (i + 1) * nu:
                    if sub_models_map[index] == evaluate_client:
                        flag = 1
                    index += 1
                if flag == 0 and m_client[evaluate_client] > 0:
                    m_client[evaluate_client] -= 1
                else:
                    t = random.choice(evaluate_client_list)
                    while t == evaluate_client:
                        t = random.choice(evaluate_client_list)
                    evaluate_client = t
                    m_client[evaluate_client] -= 1
                logger.info('>>> evaluate_client: %d...' % evaluate_client)
                # 求验证client的测试集
                idx, noise = net_dataidx_map[evaluate_client], args.noise * evaluate_client
                _, _, data_dl_test_ds, _ = get_dataloader_local(args, noise, idx)
                data_dl_test_local_class = Subset(data_dl_test_ds,
                                                  indices=[la for la, sample in enumerate(data_dl_test_ds) if
                                                           is_label_xxx(sample, j)])
                data_dl_test_local_class_loader = data.DataLoader(dataset=data_dl_test_local_class,batch_size=args.batch_size,shuffle=False)
                pre_global_acc, _ = compute_accuracy(args, net_global, data_dl_test_local_class_loader)  # 对上一个global_net进行准确性验证
                sub_model_acc, _ = compute_accuracy(args, sub_models[i], data_dl_test_local_class_loader)  # 对sub_model的准确性进行验证
                logger.info(
                    '>>> sub-model %d to label %d : pre_global_acc %f,sub_model_acc %f ' % (i,j, pre_global_acc, sub_model_acc))
                if (args.is_adaptive==0) or (args.is_adaptive==1 and round < args.adapvote_fromepoch and evaluate_client < num_attacker_index) or (args.is_adaptive==1 and round >= args.adapvote_fromepoch and evaluate_client < num_attacker_index):
                    if sub_model_acc >= pre_global_acc:
                        vote_mali += 1
                else:
                    if sub_model_acc < pre_global_acc:
                        vote_mali += 1
            c = max(0,
                    (-4 * args.v * vote_mali * vote_mali) / ((args.e - 2) * (args.e - 2)) + (8 * args.v * vote_mali) / (
                                (args.e - 2) * (args.e - 2)) - (4 * args.v) / ((args.e - 2) * (args.e - 2)) + args.v)
            c_list.append(c)
        logger.info(">>> The coefficient parameter to sub-model: " + str(c_list))
        All_data_points, global_net = sum([len(net_dataidx_map[r]) for r in range(args.num_client)]), net_global
        global_net_w = global_net.state_dict()
        if sum(c_list) != 0:
            for i in range(len(sub_models)):
                net_para = sub_models[i].state_dict()
                if i == 0:
                    for key in net_para:
                        global_net_w[key] = net_para[key] *  (c_list[i]/sum(c_list))
                else:
                    for key in net_para:
                        global_net_w[key] += net_para[key] * (c_list[i]/sum(c_list))
    else:
        nu = int((args.num_client - 1) / (d - 1))
        remain = args.num_client - nu * (d - 1)
        m_client = [args.m for i in range(args.num_client)]
        for i in range(len(sub_models)):  # 每个sub_model进行验证
            logger.info("NOIID-Defense sub-model %d:" % i)
            vote_mali = 0
            for j in range(args.num_classes):  # 对于每个sub_model需要有多少client进行验证
                evaluate_client_list_1 = [row[j] for row in client_classes]
                evaluate_client_list = [inx for inx, value in enumerate(evaluate_client_list_1) if value == 1]
                evaluate_client = random.choice(evaluate_client_list)
                if i == 0:
                    index, flag = 0, 0
                else:
                    index, flag = remain + (i - 1) * nu, 0
                if i == 0:
                    while index < remain:
                        if sub_models_map[index] == evaluate_client:
                            flag = 1
                        index += 1
                else:
                    while index < remain + i * nu:
                        if sub_models_map[index] == evaluate_client:
                            flag = 1
                        index += 1
                if flag == 0 and m_client[evaluate_client] > 0:
                    m_client[evaluate_client] -= 1
                else:
                    t = random.choice(evaluate_client_list)
                    while t == evaluate_client:
                        t = random.choice(evaluate_client_list)
                    evaluate_client = t
                    m_client[evaluate_client] -= 1
                logger.info('>>> evaluate_client: %d...' % evaluate_client)
                # 求验证client的测试集
                idx, noise = net_dataidx_map[evaluate_client], args.noise * evaluate_client
                _, _, data_dl_test_ds, _ = get_dataloader_local(args, noise, idx)
                data_dl_test_local_class = Subset(data_dl_test_ds,
                                                  indices=[la for la, sample in enumerate(data_dl_test_ds) if
                                                           is_label_xxx(sample, j)])
                data_dl_test_local_class_loader = data.DataLoader(dataset=data_dl_test_local_class,
                                                                  batch_size=args.batch_size, shuffle=False)
                pre_global_acc, _ = compute_accuracy(args, net_global, data_dl_test_local_class_loader)  # 对上一个global_net进行准确性验证
                sub_model_acc, _ = compute_accuracy(args, sub_models[i], data_dl_test_local_class_loader)  # 对sub_model的准确性进行验证
                logger.info(
                    '>>> sub-model %d to label %d : pre_global_acc %f,sub_model_acc %f ' % (
                    i, j, pre_global_acc, sub_model_acc))
                if (args.is_adaptive==0) or (args.is_adaptive==1 and round < args.adapvote_fromepoch and evaluate_client < num_attacker_index) or (args.is_adaptive==1 and round >= args.adapvote_fromepoch and evaluate_client < num_attacker_index):
                    if sub_model_acc < pre_global_acc:
                        vote_mali += 1
                else:
                    if sub_model_acc >= pre_global_acc:
                        vote_mali += 1
            c = max(0,
                    (-4 * args.v * vote_mali * vote_mali) / ((args.e - 2) * (args.e - 2)) + (8 * args.v * vote_mali) / (
                            (args.e - 2) * (args.e - 2)) - (4 * args.v) / ((args.e - 2) * (args.e - 2)) + args.v)
            c_list.append(c)
        logger.info(">>> The coefficient parameter to sub-model: " + str(c_list))
        All_data_points, global_net = sum([len(net_dataidx_map[r]) for r in range(args.num_client)]), net_global
        global_net_w = global_net.state_dict()
        if sum(c_list) != 0:
            for i in range(len(sub_models)):
                net_para = sub_models[i].state_dict()
                if i == 0:
                    for key in net_para:
                        global_net_w[key] = net_para[key] * (c_list[i]/sum(c_list))
                else:
                    for key in net_para:
                        global_net_w[key] += net_para[key] * (c_list[i]/sum(c_list))
    return global_net_w