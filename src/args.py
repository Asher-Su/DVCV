import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    """联邦学习交互过程参数设置"""
    # model设置需要训练的模型
    parser.add_argument('--model',type=str,default='cnn',help="model that need train")
    # epochs训练的总轮次
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    # local_ep每一总轮次每个client训练轮次
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    # num_client总共参与训练的client数量
    parser.add_argument('--num_client', type=int, default=10,
                        help="number of users: K")
    # local_bs本地模型训练的batch
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    # lr学习率
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    # momentum表示SGD的momentum
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    # dataset训练数据集的选取
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                            of dataset")
    # num_classes训练集数据可以分为几类
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                            of classes")
    # batch_size训练集和测试集的batch设置大小
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size of train and test')
    # AGR聚合算法设置
    parser.add_argument('--AGR', type=str, default='fedavg', help='the aggregation algorithm(fedavg,krum,Trimmed_mean,median)')
    # train_alg训练算法
    parser.add_argument('--train_alg', type=str, default='normal', help='The train algorithm(normal)')
    # optimizer训练激活器
    parser.add_argument('--optimizer', type=str, default='adam', help='The train optimizer(adam,sgd)')
    # L2正则化增强参数
    parser.add_argument('--reg', type=float, default='1e-5', help='L2 regularization strength')
    # noise表示添加noise的大小
    parser.add_argument('--noise', type=float, default=0, help='How many noise you want to add')

    """其他配置参数"""
    # gpu设置是否使用gpu
    parser.add_argument('--gpu', default='cuda:0', help="To use cuda, set \
                         to a specific GPU ID. Default set to use CPU.")
    # gpu_id设置使用gpu的id
    parser.add_argument('--gpu_id', default='0', help="choose which gpu you will use")
    # init_seed初始化种子
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    # iid设置数据集是否独立同分布
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    # beta迪利克雷分布中参数beta的设定
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')

    """debug相关参数"""
    # datadir数据集存储的路径
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    # logdir日志存储位置
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    # 设定日志文件的名字
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    """投毒相关参数"""
    parser.add_argument("--poison_type", type=str, default='agr_updates', help="the type of the attack, e.g. Fang, agr_updates,LIE")
    parser.add_argument("--startattack_epoch",type=int,default=0,help="the start attack epoch")
    parser.add_argument("--is_adaptive",type=int, default=1, help="Based on our method, the attacker does adaptive voting")
    parser.add_argument("--adapvote_fromepoch",type=int,default=0,help="From which round to start the adaptive voting")
    parser.add_argument("--mali_radio", type=float, default=0.2, help="the attacker radio")
    # agr_updates攻击类型的参数设置
    parser.add_argument("--perturb_type", type=str, default='is', help="when choose agr_updates, the perturbation type chooses from iuv,isd,is")
    parser.add_argument("--tao", type=float, default=1.0, help="when choose agr_updates, the threshold")
    # Fang攻击类型的参数设置
    parser.add_argument("--lambdaa",type=float, default=100, help="the maximum approximation params in Fang.Note that the lambda in Fang Attack is >0")

    """防御相关参数"""
    parser.add_argument("--defense_type", type=str, default='Ours', help="the type of defense, e.g. TDSC, Ours")
    # TDSC防御方法的参数设置
    parser.add_argument("--e", type=int, default=5, help="(TDSC,iid) the number of clients that evaluate one sub-model,note that it must >4")
    parser.add_argument("--m", type=int, default=5, help="The number of sub-models that a client can evaluate")
    parser.add_argument("--v", type=float, default=0.5, help="The initial penalizing coefficient")
    # iid场景下
    parser.add_argument("--d", type=int, default=5, help="the number of sub-models in current round(iid)")
    # noiid场景下
    parser.add_argument("--threshold", type=int, default=200, help="The threshold of the number of the class")

    # Ours防御方法的参数设置
    parser.add_argument("--class_evaluate_num", type=int, default=5, help="The first number of owned class samples is used for evaluation")
    parser.add_argument("--v1", type=int, default=0.5,help="Class weight adjustment parameters")
    parser.add_argument("--v2",type=int,default=0.5,help="Client weight adjustment parameters")
    parser.add_argument("--dd",type=int,default=5,help="the number of sub-models in current round(Our defense)")


    args = parser.parse_args()
    return args
