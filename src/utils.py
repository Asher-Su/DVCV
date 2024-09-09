import os
from math import sqrt
from torch.autograd import Variable
import torch
from torchvision import transforms,datasets
from models import *
import numpy as np
import logging
from datasets import *
import torch.optim as optim
import random
import heapq

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
    自定义添加噪声类：完成对tensor的所有位置添加噪声
"""
class AddGaussianNoise(object):

    def __init__(self, noise, mean=0.):
        self.std = noise
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# 过滤测试集中标签是否为xxx
def is_label_xxx(sample,j):
    return sample[1] == j

def get_dataloader(args,dataidxs=None):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        train_ds = CIFAR10_truncated(args.datadir,dataidxs=dataidxs,train=True,transform=transform_train,download=True)
        test_ds = CIFAR10_truncated(args.datadir,transform=transform_test,download=True)
        train_dl = data.DataLoader(dataset=train_ds,batch_size=args.batch_size,drop_last=True,shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        train_ds = CIFAR100_truncated(args.datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                                     download=True)
        test_ds = CIFAR100_truncated(args.datadir, transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    elif args.dataset == 'mnist':
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        train_ds = MNIST_truncated(args.datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                                     download=True)
        test_ds = MNIST_truncated(args.datadir, transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    elif args.dataset == 'STL10':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize])
        train_ds = STL10_truncated(args.datadir, dataidxs=dataidxs, split='train', transform=transform_train,
                                     download=True)
        test_ds = STL10_truncated(args.datadir,split='test',transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds

def get_dataloader_local(args,noise,dataidxs=None):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            normalize])
        train_ds = CIFAR10_truncated(args.datadir,dataidxs=dataidxs,train=True,transform=transform_train,download=True)
        test_ds = CIFAR10_truncated(args.datadir,transform=transform_test,download=True)
        train_dl = data.DataLoader(dataset=train_ds,batch_size=args.batch_size,drop_last=True,shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            normalize])
        train_ds = CIFAR100_truncated(args.datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                                     download=True)
        test_ds = CIFAR100_truncated(args.datadir, transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    elif args.dataset == 'mnist':
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            normalize
        ])
        train_ds = MNIST_truncated(args.datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                                     download=True)
        test_ds = MNIST_truncated(args.datadir, transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    elif args.dataset == 'STL10':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            AddGaussianNoise(noise=noise),
            normalize])
        train_ds = STL10_truncated(args.datadir, dataidxs=dataidxs, split='train', transform=transform_train,
                                     download=True)
        test_ds = STL10_truncated(args.datadir,split='test',transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds

def local_model_aggregation(args,net_global,nets_this_round_after_train,net_dataidx_map):
    if args.AGR == 'fedavg':
        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.num_client)])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.num_client)]
        global_w = net_global.state_dict()
        for net_i,net in enumerate(nets_this_round_after_train.values()):
            net_para = net.state_dict()
            if net_i == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_i]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_i]
    return global_w

def compute_accuracy(args,net,data_dl_test):
    net.eval()
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    loss_collector = []
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(data_dl_test):
            if torch.cuda.is_available():
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            else:
                x, target = x.cpu(),target.cpu()
            out = net(x)
            target = target.long()
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
        if len(loss_collector) >0:
            avg_loss = sum(loss_collector) / len(loss_collector)
        else:
            avg_loss = 0
    if total > 0:
        return correct / float(total), avg_loss
    else:
        return 0.0, avg_loss


def local_normal_train(args,data_dl_train_local,data_dl_test_local,net,net_i):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,weight_decay=args.reg)
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    "本地模型训练"
    for epoch in range(args.local_ep):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(data_dl_train_local):
            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()
            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Net: %d Epoch: %d Loss: %f' % (net_i, epoch, epoch_loss))

    train_acc,_ = compute_accuracy(args, net, data_dl_train_local)
    test_acc,_ = compute_accuracy(args ,net, data_dl_test_local)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info('>> ** Training complete **')
    return net


def init_nets(args,client_num):
    nets = {net_i: None for net_i in range(client_num)}
    if args.model == 'cnn' and args.dataset == 'cifar10':
        for net_i in range(client_num):
            net = CNNCifar(args=args)
            if torch.cuda.is_available():
                net.to(args.gpu)
            else:
                net.to('cpu')
            nets[net_i] = net

    if args.model == 'cnn' and args.dataset == 'mnist':
        for net_i in range(client_num):
            net = CNNMNIST(args=args)
            if torch.cuda.is_available():
                net.to(args.gpu)
            else:
                net.to('cpu')
            nets[net_i] = net

    if args.model == 'cnn' and args.dataset == 'STL10':
        for net_i in range(client_num):
            net = CNNSTL10(args=args)
            if torch.cuda.is_available():
                net.to(args.gpu)
            else:
                net.to('cpu')
            nets[net_i] = net

    if args.model == 'resnet50' and args.dataset =='mnist':
        for net_i in range(client_num):
            net = RESNET50(args=args)
            if torch.cuda.is_available():
                net.to(args.gpu)
            else:
                net.to('cpu')
            nets[net_i] = net
    return nets

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def record_net_data(y_train,net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx],return_counts=True)  # 返回每个party对应数据集的标签并唯一排序  unq_cnt代表该client具有的数据集种类数量
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # tmp表示每个对应标签类下具体的数据集数量
        net_cls_counts[net_i] = tmp  # 将结果保存在net_cls_counts列表中(对应哪个party对应其数据相关信息tmp)
    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0  # 计算每个party对应的数据集总数
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    logger.info('mean: '+str(np.mean(data_list)))
    logger.info('std: '+str(np.std(data_list)))
    logger.info('Data Distribution is: '+str(net_cls_counts))

def partition_data(args):
    if args.dataset == 'mnist':
        datadir = args.datadir + 'mnist'
        apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(datadir,train=True,download=True,transform=apply_transform)
        test_dataset = datasets.MNIST(datadir, train=False, download=True,transform=apply_transform)
        train_data, train_label= train_dataset.data,train_dataset.targets
        test_data, test_label = test_dataset.data, test_dataset.targets
        n_idxs = train_label.shape[0]
        if args.iid == 1:
            idxs = np.random.permutation(n_idxs)
            batch_idxs = np.array_split(idxs, args.num_client)
            net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_client)}
        elif args.iid == 0:
            min_size = 0
            min_require_size = 10
            K = 10  # 代表数据集中类的数量
            N = n_idxs
            net_dataidx_map = {}
            while min_size < min_require_size:
                # 对每一个parties生成对应的一个list
                idx_batch = [[] for _ in range(args.num_client)]
                for k in range(K):
                    idx_k = np.where(train_label == k)[0]  # 取训练集标签中等于对应类的label的训练数据集的索引
                    np.random.shuffle(idx_k)  # 对该类索引进行shuffle操作
                    proportions = np.random.dirichlet(np.repeat(args.beta, args.num_client))  # 获得对于每一个party的采样率
                    proportions = np.array(
                        [p * (len(idx_j) < N / args.num_client) for p, idx_j in zip(proportions, idx_batch)])  # 重新计算采样率
                    proportions = proportions / proportions.sum()  # 归一化采样概率
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]  # 获得除最后一个party对应的本类下的idx数量之外的所有数量
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k,proportions))]  # 将idx_k按照proportions进行分割，最后生成idx_batch为每个party对应该类下的标签
                    min_size = min([len(idx_j) for idx_j in idx_batch])  # 判断每个client分得的最小标签数
            for j in range(args.num_client):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        logger.info("dataset is: " + args.dataset)
        record_net_data(train_label, net_dataidx_map)
        return (train_data, train_label,test_data, test_label,net_dataidx_map,n_idxs)

    elif args.dataset == 'cifar10' or args.dataset == 'cifar100' :
        if args.dataset == 'cifar10':
            datadir = args.datadir + 'cifar10'
        elif args.dataset == 'cifar100':
            datadir = args.datadir + 'cifar100'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(datadir,train=True,download=True,transform=apply_transform)
            test_dataset = datasets.CIFAR10(datadir, train=False, download=True,transform=apply_transform)
        elif args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(datadir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.CIFAR100(datadir, train=False, download=True, transform=apply_transform)
        train_data, train_label= train_dataset.data,train_dataset.targets
        train_data,train_label = torch.from_numpy(train_data),torch.tensor(train_label)
        test_data, test_label = test_dataset.data, test_dataset.targets
        test_data, test_label = torch.from_numpy(test_data), torch.tensor(test_label)
        n_idxs = train_label.shape[0]
        if args.iid == 1:
            idxs = np.random.permutation(n_idxs)
            batch_idxs = np.array_split(idxs, args.num_client)
            net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_client)}
        elif args.iid == 0:
            min_size = 0
            min_require_size = 10
            if args.dataset == 'cifar10':
                K = 10  # 代表数据集中类的数量
            elif args.dataset == 'cifar100':
                K =100
            N = n_idxs
            net_dataidx_map = {}
            while min_size < min_require_size:
                # 对每一个parties生成对应的一个list
                idx_batch = [[] for _ in range(args.num_client)]
                for k in range(K):
                    idx_k = np.where(train_label == k)[0]  # 取训练集标签中等于对应类的label的训练数据集的索引
                    np.random.shuffle(idx_k)  # 对该类索引进行shuffle操作
                    proportions = np.random.dirichlet(np.repeat(args.beta, args.num_client))  # 获得对于每一个party的采样率
                    proportions = np.array(
                        [p * (len(idx_j) < N / args.num_client) for p, idx_j in zip(proportions, idx_batch)])  # 重新计算采样率
                    proportions = proportions / proportions.sum()  # 归一化采样概率
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]  # 获得除最后一个party对应的本类下的idx数量之外的所有数量
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k,proportions))]  # 将idx_k按照proportions进行分割，最后生成idx_batch为每个party对应该类下的标签
                    min_size = min([len(idx_j) for idx_j in idx_batch])  # 判断每个client分得的最小标签数
            for j in range(args.num_client):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        logger.info("dataset is: " + args.dataset)
        record_net_data(train_label, net_dataidx_map)
        return (train_data, train_label,test_data, test_label,net_dataidx_map,n_idxs)

    elif args.dataset == 'STL10':
        datadir = args.datadir + 'STL10'
        apply_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.STL10(root=datadir,split='train',download=True,transform=apply_transform)
        test_dataset = datasets.STL10(root=datadir, split='test', download=True,transform=apply_transform)
        train_data, train_label= train_dataset.data,train_dataset.labels
        test_data, test_label = test_dataset.data, test_dataset.labels
        n_idxs = train_label.shape[0]
        if args.iid == 1:
            idxs = np.random.permutation(n_idxs)
            batch_idxs = np.array_split(idxs, args.num_client)
            net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_client)}
        elif args.iid == 0:
            min_size = 0
            min_require_size = 10
            K = 10  # 代表数据集中类的数量
            N = n_idxs
            net_dataidx_map = {}
            while min_size < min_require_size:
                # 对每一个parties生成对应的一个list
                idx_batch = [[] for _ in range(args.num_client)]
                for k in range(K):
                    idx_k = np.where(train_label == k)[0]  # 取训练集标签中等于对应类的label的训练数据集的索引
                    np.random.shuffle(idx_k)  # 对该类索引进行shuffle操作
                    proportions = np.random.dirichlet(np.repeat(args.beta, args.num_client))  # 获得对于每一个party的采样率
                    proportions = np.array(
                        [p * (len(idx_j) < N / args.num_client) for p, idx_j in zip(proportions, idx_batch)])  # 重新计算采样率
                    proportions = proportions / proportions.sum()  # 归一化采样概率
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]  # 获得除最后一个party对应的本类下的idx数量之外的所有数量
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k,proportions))]  # 将idx_k按照proportions进行分割，最后生成idx_batch为每个party对应该类下的标签
                    min_size = min([len(idx_j) for idx_j in idx_batch])  # 判断每个client分得的最小标签数
            for j in range(args.num_client):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        logger.info("dataset is: "+args.dataset)
        record_net_data(train_label,net_dataidx_map)
        return (train_data, train_label,test_data, test_label,net_dataidx_map,n_idxs)


"""
    TDSC中noiid条件下子模型划分
"""
def TDSC_sub_model_partition_noiid(args, nets_this_round_after_train,net_dataidx_map):
    # 计算在noiid条件下分组数量
    client_classes = [[0 for i in range(args.num_classes)] for j in range(args.num_client)] # 每个client在每个class上的数据数量是否大于门限值
    for i in range(args.num_client):
        targets = [0 for j in range(args.num_classes)]
        idx, noise = net_dataidx_map[i], args.noise * i
        data_dl_test_local, _, _, _ = get_dataloader_local(args, noise, idx)
        for batch_idx, (x,target) in enumerate(data_dl_test_local):
            for k in range(args.num_classes):
                num = torch.count_nonzero(target == k)
                targets[k]+=num
        for k in range(args.num_classes):
            if targets[k] >= args.threshold:
                client_classes[i][k] = 1
            else:
                client_classes[i][k] = 0
    column_sums = [sum(column) for column in zip(*client_classes)]
    d = min(column_sums)
    logger.info(">>> d = %d" % d)
    sub_models, sub_models_map,total_data_points =[], [i for i in range(args.num_client)],[0 for r in range(d)]
    random.shuffle(sub_models_map)
    logger.info(">>> NOIID sub-model partition: "+str(sub_models_map))
    if args.num_client % d == 0:
        nu, index = int(args.num_client / d), 0 # nu表示每个sub_model是由多少个local_model聚合而成
        for i in range(d):
            j = nu
            while j>0:
                total_data_points[i]+=len(net_dataidx_map[sub_models_map[index]])
                index +=1
                j=j-1
        index = 0
        for i in range(d):
            j = nu
            net = init_nets(args,1)[0]
            while j>0:
                net_para = net.state_dict()
                w = nets_this_round_after_train[sub_models_map[index]].state_dict()
                if j == nu:
                    for key in net_para:
                        net_para[key] = w[key] * (1/nu)
                else:
                    for key in net_para:
                        net_para[key] += w[key] * (1/nu)
                net.load_state_dict(net_para)
                j = j-1
                index = index+1
            sub_models.append(net)
    elif args.num_client % d != 0:
        nu, index = int((args.num_client-1)/(d-1)), 0
        remain = args.num_client - nu *(d - 1)
        while index < remain:
            total_data_points[0] += len(net_dataidx_map[sub_models_map[index]])
            index +=1
        index = 0
        net = init_nets(args,1)[0]
        while index < remain:
            net_para = net.state_dict()
            w = nets_this_round_after_train[sub_models_map[index]].state_dict()
            if index == 0:
                for key in net_para:
                    net_para[key] = w[key] * (1/remain)
            else:
                for key in net_para:
                    net_para[key] += w[key] * (1/remain)
            net.load_state_dict(net_para)
            index +=1
        sub_models.append(net)
        for i in range(d-1):
            j = nu
            while j>0:
                total_data_points[i+1]+=len(net_dataidx_map[sub_models_map[index]])
                index +=1
                j=j-1
        index = remain
        for i in range(d-1):
            j = nu
            net = init_nets(args,1)[0]
            while j>0:
                net_para = net.state_dict()
                w = nets_this_round_after_train[sub_models_map[index]].state_dict()
                if j == nu:
                    for key in net_para:
                        net_para[key] = w[key] * (1/nu)
                else:
                    for key in net_para:
                        net_para[key] += w[key] * (1/nu)
                net.load_state_dict(net_para)
                j = j-1
                index = index+1
            sub_models.append(net)
    return sub_models,sub_models_map,total_data_points,client_classes,d

"""
    TDSC中iid条件下的子模型划分
"""
def TDSC_sub_model_partition_iid(args, nets_this_round_after_train,net_dataidx_map):
    sub_models, sub_models_map,total_data_points =[], [i for i in range(args.num_client)],[0 for r in range(args.d)]
    random.shuffle(sub_models_map)
    logger.info(">>> IID sub-model partition: " + str(sub_models_map))
    if args.num_client % args.d == 0:
        nu, index = int(args.num_client / args.d), 0 # nu表示每个sub_model是由多少个local_model聚合而成
        for i in range(args.d):
            j = nu
            while j>0:
                total_data_points[i]+=len(net_dataidx_map[sub_models_map[index]])
                index +=1
                j=j-1
        index = 0
        for i in range(args.d):
            j = nu
            net = init_nets(args,1)[0]
            while j>0:
                net_para = net.state_dict()
                w = nets_this_round_after_train[sub_models_map[index]].state_dict()
                if j == nu:
                    for key in net_para:
                        net_para[key] = w[key] * (1/nu)
                else:
                    for key in net_para:
                        net_para[key] += w[key] * (1/nu)
                net.load_state_dict(net_para)
                j = j-1
                index = index+1
            sub_models.append(net)
    elif args.num_client % args.d != 0:
        nu, index = int((args.num_client-1)/(args.d-1)), 0
        remain = args.num_client - nu *(args.d - 1)
        while index < remain:
            total_data_points[0] += len(net_dataidx_map[sub_models_map[index]])
            index +=1
        index = 0
        net = init_nets(args,1)[0]
        while index < remain:
            net_para = net.state_dict()
            w = nets_this_round_after_train[sub_models_map[index]].state_dict()
            if index == 0:
                for key in net_para:
                    net_para[key] = w[key] * (1/remain)
            else:
                for key in net_para:
                    net_para[key] += w[key] * (1/remain)
            net.load_state_dict(net_para)
            index +=1
        sub_models.append(net)
        for i in range(args.d-1):
            j = nu
            while j>0:
                total_data_points[i+1]+=len(net_dataidx_map[sub_models_map[index]])
                index +=1
                j=j-1
        index = remain
        for i in range(args.d-1):
            j = nu
            net = init_nets(args,1)[0]
            while j>0:
                net_para = net.state_dict()
                w = nets_this_round_after_train[sub_models_map[index]].state_dict()
                if j == nu:
                    for key in net_para:
                        net_para[key] = w[key] * (1/nu)
                else:
                    for key in net_para:
                        net_para[key] += w[key] * (1/nu)
                net.load_state_dict(net_para)
                j = j-1
                index = index+1
            sub_models.append(net)
    return sub_models,sub_models_map,total_data_points

"""
    Ours防御方法的子模型划分
"""
def Ours_sub_model_partition(args, nets_this_round_after_train,net_dataidx_map):
    # 计算每个client对于哪些class拥有投票权
    client_classes = []
    for i in range(args.num_client):
        targets = [0 for j in range(args.num_classes)]
        idx, noise = net_dataidx_map[i], args.noise * i
        data_dl_test_local, _, _, _ = get_dataloader_local(args, noise, idx)
        for batch_idx, (x,target) in enumerate(data_dl_test_local):
            for k in range(args.num_classes):
                num = torch.count_nonzero(target == k)
                targets[k]+=num
        targets = torch.stack(targets)
        sort_targets= torch.argsort(targets,descending=True).tolist()
        client_classes.append(sort_targets)
    client_classes = [[row[i] for i in range(args.class_evaluate_num)] for row in client_classes]
    sub_models, sub_models_map, total_data_points = [], [i for i in range(args.num_client)], [0 for r in range(args.d)]
    random.shuffle(sub_models_map)
    logger.info(">>> Our Defense sub-model partition: " + str(sub_models_map))
    if args.num_client % args.dd == 0:
        nu, index = int(args.num_client / args.dd), 0 # nu表示每个sub_model是由多少个local_model聚合而成
        for i in range(args.dd):
            j = nu
            while j>0:
                total_data_points[i]+=len(net_dataidx_map[sub_models_map[index]])
                index +=1
                j=j-1
        index = 0
        for i in range(args.dd):
            j = nu
            net = init_nets(args,1)[0]
            while j>0:
                net_para = net.state_dict()
                w = nets_this_round_after_train[sub_models_map[index]].state_dict()
                if j == nu:
                    for key in net_para:
                        net_para[key] = w[key] * (1/nu)
                else:
                    for key in net_para:
                        net_para[key] += w[key] * (1/nu)
                net.load_state_dict(net_para)
                j = j-1
                index = index+1
            sub_models.append(net)
    elif args.num_client % args.dd != 0:
        nu, index = int((args.num_client-1)/(args.dd-1)), 0
        remain = args.num_client - nu *(args.dd - 1)
        while index < remain:
            total_data_points[0] += len(net_dataidx_map[sub_models_map[index]])
            index +=1
        index = 0
        net = init_nets(args,1)[0]
        while index < remain:
            net_para = net.state_dict()
            w = nets_this_round_after_train[sub_models_map[index]].state_dict()
            if index == 0:
                for key in net_para:
                    net_para[key] = w[key] * (1/remain)
            else:
                for key in net_para:
                    net_para[key] += w[key] * (1/remain)
            net.load_state_dict(net_para)
            index +=1
        sub_models.append(net)
        for i in range(args.dd-1):
            j = nu
            while j>0:
                total_data_points[i+1]+=len(net_dataidx_map[sub_models_map[index]])
                index +=1
                j=j-1
        index = remain
        for i in range(args.dd-1):
            j = nu
            net = init_nets(args,1)[0]
            while j>0:
                net_para = net.state_dict()
                w = nets_this_round_after_train[sub_models_map[index]].state_dict()
                if j == nu:
                    for key in net_para:
                        net_para[key] = w[key] * (1/nu)
                else:
                    for key in net_para:
                        net_para[key] += w[key] * (1/nu)
                net.load_state_dict(net_para)
                j = j-1
                index = index+1
            sub_models.append(net)
    return sub_models,sub_models_map,total_data_points,client_classes