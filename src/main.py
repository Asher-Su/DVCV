import random

from args import *
from utils import *
from poison import *
import datetime
import json
import logging
import numpy as np
import torch
from tqdm import tqdm
from defense import *

if __name__ == '__main__':
    args = args_parser()
    mkdirs(args.datadir)
    mkdirs(args.logdir)

    """完成对json文件的创建并写入"""
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)


    """完成对log文件的初始化操作"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # 对日志文件进行基础配置
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    """完成对种子的初始化"""
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.gpu != None:
        torch.cuda.manual_seed(seed)

    """进行数据集分配操作"""
    logger.info("Partitioning data: "+str(args.iid))
    train_data, train_label,test_data, test_label,net_dataidx_map, n_idxs= partition_data(args)
    logger.info('noise-base: %f' %args.noise)
    logger.info('#'*100)

    """完成神经网络的初始化操作"""
    logger.info("Initializing nets")
    nets_client = init_nets(args,args.num_client)
    temp_net_global = init_nets(args,1)
    net_global = temp_net_global[0]
    num_attacker = args.mali_radio * args.num_client # 攻击者数量
    num_attacker_index = args.num_client - num_attacker # 首个攻击者所在的索引
    logger.info("The length of client nets is: "+str(len(nets_client)))
    logger.info("The length of global net is: "+str(len(temp_net_global)))
    logger.info("The number of Attackers is: "+str(num_attacker))
    logger.info("The index of the first Attack is: "+str(num_attacker_index))
    logger.info('#'*100)

    """完成模型训练过程"""
    logger.info("Training Start")
    epochs = args.epochs
    all_idxs = np.random.permutation(n_idxs)
    data_dl_train_global, data_dl_test_global, _, _ = get_dataloader(args,dataidxs=all_idxs)
    # 如果选择的是Ours防御方法需要对每个client的类权重进行初始化
    if args.defense_type == 'Ours':
        vote_weight = [[1 for i in range(args.class_evaluate_num)] for j in range(args.num_client)]
    for round in tqdm(range(epochs)):
        logger.info("+-" * 50)
        logger.info("【 Epoch "+str(round)+"】: ")
        nets_this_round_after_train = {}
        net_global.eval()
        for param in net_global.parameters():
            param.requires_grad = False
        net_global_w = net_global.state_dict()
        for net_i,net in enumerate(nets_client.values()):
            net.load_state_dict(net_global_w)
            idxs = net_dataidx_map[net_i]
            noise = args.noise * (net_i+1)
            data_dl_train_local, data_dl_test_local, _, _= get_dataloader_local(args,noise,idxs)
            if args.train_alg == 'normal':
                temp_net = local_normal_train(args,data_dl_train_local,data_dl_test_local,net,net_i)
                nets_this_round_after_train[net_i] = temp_net
        if args.poison_type == 'agr_updates' and round >= args.startattack_epoch:
            logger.info(">>> Attack start... The attack type is agr_updates attack")
            attack_net_w = poison_agr_updates_avg(args,nets_this_round_after_train,net_dataidx_map,num_attacker_index)
            for n in range(int(num_attacker_index),args.num_client):
                nets_this_round_after_train[n].load_state_dict(attack_net_w)
            logger.info(">>> Attack end...")
        if args.poison_type == 'Fang' and round >= args.startattack_epoch:
            logger.info(">>> Attack start... The attack type is Fang attack")
            attack_net_w = poison_fang_avg(args, net_global, nets_this_round_after_train, net_dataidx_map, num_attacker_index)
            for n in range(int(num_attacker_index), args.num_client):
                nets_this_round_after_train[n].load_state_dict(attack_net_w)
            logger.info(">>> Attack end...")
        if args.poison_type == 'LIE' and round >= args.startattack_epoch:
            logger.info(">>> Attack start... The attack type is LIE attack")
            attack_net_w = poison_lie(args, nets_this_round_after_train)
            for n in range(int(num_attacker_index), args.num_client):
                nets_this_round_after_train[n].load_state_dict(attack_net_w)
            logger.info(">>> Attack end...")
        if args.iid == 1 and args.defense_type == 'TDSC':
            logger.info(">>> Defense start... The Defense type is TDSC_iid")
            sub_models,sub_models_map,total_data_points = TDSC_sub_model_partition_iid(args,nets_this_round_after_train,net_dataidx_map)
            global_w = TDSC_defense_iid(args,sub_models,sub_models_map,net_global,net_dataidx_map,total_data_points,round,num_attacker_index)
            net_global.load_state_dict(global_w)
            logger.info(">>> Defense end...")
        if args.iid == 0 and args.defense_type == 'TDSC':
            logger.info(">>> Defense start... The Defense type is TDSC_noiid")
            sub_models,sub_models_map,total_data_points,client_classes,d = TDSC_sub_model_partition_noiid(args,nets_this_round_after_train,net_dataidx_map)
            global_w = TDSC_defense_noiid(args, sub_models, sub_models_map, net_global, net_dataidx_map, total_data_points, client_classes,d,round,num_attacker_index)
            net_global.load_state_dict(global_w)
            logger.info(">>> Defense end...")
        if args.defense_type == 'Ours':
            logger.info(">>> Defense start... The Defense type is Ours")
            sub_models,sub_models_map,total_data_points,client_classes = Ours_sub_model_partition(args,nets_this_round_after_train,net_dataidx_map)
            global_w, vote_weight = Our_defense(args, sub_models, sub_models_map,net_global, net_dataidx_map, total_data_points, client_classes,vote_weight,round,num_attacker_index)
            net_global.load_state_dict(global_w)
            logger.info(">>> Defense end...")
        if args.AGR == 'fedavg' and args.defense_type is None:
            global_w = local_model_aggregation(args,net_global,nets_this_round_after_train,net_dataidx_map)
            net_global.load_state_dict(global_w)
        test_acc,_ = compute_accuracy(args,net_global, data_dl_test_global)
        if args.poison_type is None:
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
        elif args.poison_type is not None and args.defense_type is None:
            logger.info('>> After Attack Global Model Test accuracy: %f' % test_acc)
        elif args.poison_type is not None and args.defense_type is not None:
            logger.info('>> After Attack and Defense Global Model Test accuracy: %f' % test_acc)


