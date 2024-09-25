import numpy as np
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import argparse
from drive.utils import RMSE
from drive.FADNet import FADNet, FADNetFFA
from tqdm import tqdm
from utils import *
import datetime
from bypass_bn import disable_running_stats, enable_running_stats
from sam import SAM

from utils import load_nvidia_data, load_gazebo_data, load_carla_data, get_dataloader
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FADNet', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--decay', type=str, default='sqrt', help='decay')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--vir_clients_num', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    args = parser.parse_args()
    return args


def record_net_data_stats(y_train, class_num, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for k in range(class_num):
            if k not in tmp:
                tmp[k]=0
        net_cls_counts[net_i] = tmp
    
    party_data = {}
    for client_id, class_kv in net_cls_counts.items():
        party_data[client_id] = dict(sorted(class_kv.items()))

    logger.info('Data statistics: %s' % str(party_data))
    return party_data

def partition_data(dataset, class_num, datadir, logdir, partition, n_parties, device, beta=0.4): 
    # import pdb;pdb.set_trace()
    if dataset == 'nvidia':
        X_train, y_train, _, X_test, y_test, _ = load_nvidia_data(datadir, device)
    elif dataset == 'gazebo':
        X_train, y_train, _, X_test, y_test, _ = load_gazebo_data(datadir, device)
    elif dataset == 'carla':
        X_train, y_train, _, X_test, y_test, _ = load_carla_data(datadir, device)

    y_train = np.array(y_train)
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10

        N = y_train.shape[0]
        net_dataidx_map = {}
        # import pdb;pdb.set_trace()
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(class_num):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        net_dataidx_map_ = {i:[[] for j in range(class_num)] for i in range(n_parties)}
        for i in range(n_parties):
            for j in net_dataidx_map[i]:
                idx = y_train[j]
                net_dataidx_map_[i][idx].append(j)
    traindata_cls_counts = record_net_data_stats(y_train, class_num, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_, traindata_cls_counts)


########################################################################################################################################################
args = get_args()
mkdirs(args.logdir)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

if args.log_file_name is None:
    args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
log_path = args.log_file_name + '.log'
logging.basicConfig(
    filename=os.path.join(args.logdir, log_path),
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info(args.device)
logger.info(str(args))
seed = args.init_seed
logger.info("#" * 100)
    
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)
beta=args.beta
dataset = args.dataset
device = args.device
import time

lr = args.lr
reg = 5e-4
acc_list = []
rounds = args.comm_round
epochs = 1
real_client_num = args.n_parties
vir_clients_num = args.vir_clients_num

if dataset == 'gazebo':
    real_client_num = args.n_parties
    # vir_clients_num = 30 # 1,5,20,30
    class_num=20
    ch = 1

elif dataset == 'nvidia':
    real_client_num = args.n_parties
    # vir_clients_num = 30 # 5, 20,30
    class_num=20
    ch = 1

elif dataset == 'carla':
    real_client_num = args.n_parties
    # vir_clients_num = 30 # 5, 20,30
    class_num=20
    ch = 1

X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_, traindata_cls_counts = partition_data(
    dataset, class_num, args.datadir, 'logdir_test', 'noniid-labeldir', real_client_num, device=device, beta=beta)

# each class total num
su = [0 for _ in range(class_num)]
for x in traindata_cls_counts.values():
    for i in range(class_num):
        su[i]+=x[i]
num_per_client = int(np.sum(su)/vir_clients_num)
print(f'each class total num:{su}, \nset virtual_client_num:{args.vir_clients_num}, \nper:{num_per_client}')

dataloader_virtual_matrix = [[[] for j in range(real_client_num)] for i in range(args.vir_clients_num)]
dataloader_virtual_matrix_dl = [[] for i in range(args.vir_clients_num)]

########################################################################################################################################################
# for classification task 
# level = int(num_per_client / args.vir_clients_num)
# for virtual_th in range(args.vir_clients_num):
#     current_level = [level for i in range(class_num)]
#     for client_th in range(real_client_num):
#         for i in range(class_num):
#             temp = current_level[i]
#             sampled_data = net_dataidx_map_[client_th][i][:current_level[i]]
#             dataloader_virtual_matrix[virtual_th][client_th].extend(sampled_data)
#             current_level[i] -= len(sampled_data)
#             net_dataidx_map_[client_th][i][:temp] = []
#             # debug
#             # print("client:{0} class:{1} has {2}".format(client_th, i, len(net_dataidx_map_[client_th][i])))
# dataloader_virtual_matrix

# for regression task
level = [int(item/sum(su)*num_per_client) for item in su]

for virtual_th in range(args.vir_clients_num):
    current_level = [level[i] for i in range(class_num)]
    for client_th in range(real_client_num):
        for i in range(class_num):
            temp = current_level[i]
            sampled_data = net_dataidx_map_[client_th][i][:current_level[i]]
            dataloader_virtual_matrix[virtual_th][client_th].extend(sampled_data)
            current_level[i] -= len(sampled_data)
            net_dataidx_map_[client_th][i][:temp] = []
            # debug
            # print("client:{0} class:{1} has {2}".format(client_th, i, len(net_dataidx_map_[client_th][i])))


def compute_accuracy(model, dataloader, criterion, metric, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0

    loss_collector = []    
    with torch.no_grad():
        for batch_idx, (x, _, target) in enumerate(dataloader):
            #print("x:",x)
            if device != 'cpu':
                x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += metric[0](out, target).item()

        avg_loss = sum(loss_collector) / len(loss_collector)


    if was_training:
        model.train()

    return correct / float(total), avg_loss

########################################################################################################################################################

for virtual_client_dl_idx in range(len(dataloader_virtual_matrix)):
    for real_client_dl_idx in dataloader_virtual_matrix[virtual_client_dl_idx]:
        if real_client_dl_idx == []:
            dataloader_virtual_matrix_dl[virtual_client_dl_idx].append([])
            continue
        train_dl_local, test_dl, _, _ = get_dataloader(dataset, args.datadir, 128, 64, device, real_client_dl_idx)
        dataloader_virtual_matrix_dl[virtual_client_dl_idx].append(train_dl_local)

########################################################################################################################################################
if args.model == "FADNet":
    net = FADNet()
elif args.model == "FADNetFFA":
    net = FADNetFFA()

# optimizer = SAM(
#         net.parameters(),
#         base_optimizer=torch.optim.Adam,
#         lr=lr,
#         weight_decay=args.reg
#     )

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=reg)
# optimizer=optim.SGD([param for param in net.parameters() if param.requires_grad], lr=lr)
criterion = nn.MSELoss()
metric = [RMSE]

# virtual client
net.to(device)
for rd in tqdm(range(rounds)):
    rd_time = time.time()
    virtual_id = 0
    for virtual_client_dl in dataloader_virtual_matrix_dl:
        for epoch in range(epochs):
            ep_time = time.time()
            for train_dl_local in virtual_client_dl:
                if train_dl_local == []:
                    continue
                for batch_idx, (x, _, target) in enumerate(train_dl_local):
                    x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)
                    if True:
                        optimizer.zero_grad()
                        x.requires_grad = False
                        target.requires_grad = False

                        _, _, out = net(x)
                        loss = criterion(out.to(torch.float32), target)

                        loss.backward()
                        optimizer.step()
                    else:
                        enable_running_stats(net)
                        _,_,out = net(x)
                        loss = criterion(out.to(torch.float32), target)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        # second forward-backward step
                        disable_running_stats(net)
                        _,_,out = net(x)
                        loss2 = criterion(out.to(torch.float32), target)
                        loss2.backward()
                        optimizer.second_step(zero_grad=True)
                    
            print("round::{0}, virtual_client::{1}, epoch::{2}, loss::{3}, elps_time::{4}.".format(rd,virtual_id,epoch,loss.item(),time.time()-ep_time))
            logger.info("round::{0}, virtual_client::{1}, epoch::{2}, loss::{3}, elps_time::{4}.".format(rd,virtual_id,epoch,loss.item(),time.time()-ep_time))
        virtual_id += 1
    acc, _ = compute_accuracy(net, test_dl, criterion, metric, device=device)
    acc_list.append(acc)
    print("round::{0} finished, test_acc::{1}, elpsed time::{2}".format(rd,acc,time.time()-rd_time))
    logger.info('>> Global Model Test accuracy: %f' % acc)

print(f"{dataset}_{beta} acc: {acc_list}")