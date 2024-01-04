import paddle
import numpy as np
import paddle.nn as nn
from paddle.vision.datasets import Cifar10 as CIFAR10
from paddle.vision.datasets import Cifar100 as CIFAR100
from paddle.vision import transforms
from paddle import optimizer as optim
import paddle.nn.functional as F

import os
import logging
import matplotlib.pyplot as plt
from paddle.io import DataLoader

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Swish(nn.Layer):  # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * F.sigmoid(input)


class ConvNet(nn.Layer):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier1 = nn.Linear(num_feat, 256)
        self.classifier2 = nn.Linear(256, num_classes)
        self.flatten = paddle.nn.Flatten()

    def forward(self, x):
        h = self.features(x)
        
        _out = self.flatten(h)
        x = self.classifier1(_out)
        y = self.classifier2(x)
        return h, x, y

    def embed(self, x):
        _out = self.features(x)
        _out = self.flatten(_out)
        return _out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU()
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2D(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2D(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2D(shape_feat[0])
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0])
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0])
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2D(in_channels=in_channels, out_channels=net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


def load_cifar10_data(datadir):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # data prep for test set
    transform_test = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10(data_file=os.path.join(datadir, 'cifar-10-python.tar.gz'), mode='train', transform=transform_train)
    cifar10_test_ds = CIFAR10(data_file=os.path.join(datadir, 'cifar-10-python.tar.gz'), mode='test', transform=transform_test)

    Train_data = cifar10_train_ds.data
    Test_data = cifar10_test_ds.data
    X_train,y_train = np.array([x[0] for x in Train_data]), np.array([x[1] for x in Train_data])
    X_test,y_test = np.array([x[0] for x in Test_data]), np.array([x[1] for x in Test_data])
    return (cifar10_train_ds,cifar10_test_ds,X_train,y_train,X_test,y_test)

def load_cifar100_data(datadir):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    # data prep for test set
    transform_test = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100(data_file=os.path.join(datadir, 'cifar-100-python.tar.gz'), mode='train', transform=transform_train)
    cifar100_test_ds = CIFAR100(data_file=os.path.join(datadir, 'cifar-100-python.tar.gz'), mode='test', transform=transform_test)

    Train_data = cifar100_train_ds.data
    Test_data = cifar100_test_ds.data

    X_train,y_train = np.array([x[0] for x in Train_data]), np.array([x[1] for x in Train_data])
    X_test,y_test = np.array([x[0] for x in Test_data]), np.array([x[1] for x in Test_data])

    return (cifar100_train_ds,cifar100_test_ds,X_train,y_train,X_test,y_test)

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, is_test=False):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY','cifar100'):
        if dataset == 'cifar10':
            train_ds,_,_,_,_,_ = load_cifar10_data(datadir)
            if dataidxs is not None:
                train_ds = [train_ds[i] for i in dataidxs]
            if is_test:
                _,test_ds,_,_,_,_ = load_cifar10_data(datadir)
        elif dataset == 'cifar100':
            train_ds,_,_,_,_,_ = load_cifar100_data(datadir)
            if dataidxs is not None:
                train_ds = [train_ds[i] for i in dataidxs]
            if is_test:
                _,test_ds,_,_,_,_ = load_cifar100_data(datadir)
        else:
            train_ds, test_ds = None, None

        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        if is_test:
            test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
            return train_dl,test_dl
        else:
            return train_dl

def compute_accuracy(model, dataloader):
    model.eval()
    accuracies = []
    losses = []
    for batch_id, (x_data,y_data) in enumerate(dataloader()):

        y_data = paddle.to_tensor(y_data)
        y_data = paddle.unsqueeze(y_data, 1)

        _,_,logits = model(x_data)
        loss = F.cross_entropy(logits, y_data)
        acc = paddle.metric.accuracy(logits, y_data)
        accuracies.append(acc.numpy())
        losses.append(loss.numpy())

    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)

    return avg_acc

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        _,_,X_train,y_train,X_test,y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        _,_,X_train,y_train,X_test,y_test = load_cifar100_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        N = len(y_train)
        #np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    net_dataidx_map_ = {i:[[] for j in range(10)] for i in range(10)}
    for i in range(n_parties):
        for j in net_dataidx_map[i]:
            idx = y_train[j]
            net_dataidx_map_[i][idx].append(j)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_, traindata_cls_counts)




seed = 0
np.random.seed(seed)
vir_clients_num = 10
real_client_num = 10

X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_, traindata_cls_counts = partition_data('cifar10', 'data/data152754', 'logdir_test', 'noniid-labeldir', real_client_num, beta=0.5)


dataloader_virtual_matrix = [[[] for j in range(real_client_num)] for i in range(vir_clients_num)]
dataloader_virtual_matrix_dl = [[] for i in range(vir_clients_num)]
## this is the core codes
class_num=10
for virtual_th in range(vir_clients_num):
    current_level = [500 for i in range(class_num)]
    for client_th in range(real_client_num):
        for i in range(class_num):
            temp = current_level[i]
            dataloader_virtual_matrix[virtual_th][client_th].extend(net_dataidx_map_[client_th][i][:current_level[i]])
            current_level[i] -= len(net_dataidx_map_[client_th][i][:current_level[i]])
            net_dataidx_map_[client_th][i][:temp] = []

for virtual_client_dl_idx in range(len(dataloader_virtual_matrix)):
    for real_client_dl_idx in dataloader_virtual_matrix[virtual_client_dl_idx]:
        if real_client_dl_idx == []:
            dataloader_virtual_matrix_dl[virtual_client_dl_idx].append([])
            continue
        train_dl_local= get_dataloader('cifar10', 'data/data152754', 64, 32, real_client_dl_idx, is_test=False)
        dataloader_virtual_matrix_dl[virtual_client_dl_idx].append(train_dl_local)


args_optimizer = 'sgd'
lr = 0.01
reg = 5e-4
net = ConvNet(3, 10, 128, 3, 'relu', 'instancenorm', 'avgpooling', (32, 32))


_,test_dl = get_dataloader('cifar10', 'data/data152754', 64, 32, is_test=True)

acc_list = []
rounds = 20
epochs = 10

if args_optimizer == 'adam':
    optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=reg)
else:
    optimizer = optim.SGD(parameters=net.parameters(), learning_rate=lr, weight_decay=reg)


for rd in range(rounds):
    virtual_id = 0
    for virtual_client_dl in dataloader_virtual_matrix_dl:
        for epoch in range(epochs):
            for train_dl_local in virtual_client_dl:
                if train_dl_local == []:
                    continue
                for batch_idx, (x, target) in enumerate(train_dl_local()):
                    _, _, out = net(x)
                    loss = F.cross_entropy(out, target)

                    loss.backward()
                    optimizer.step()
                    optimizer.clear_grad()
            print("round::{0}, virtual_client::{1}, epoch::{2}, loss::{3}.".format(rd,virtual_id,epoch,loss.item()))
        virtual_id += 1

    acc = compute_accuracy(net, test_dl)
    acc_list.append(acc)

print(acc_list)

plt.plot(range(rounds),acc_list)
plt.savefig("./virtual_clients.png",dpi=330)







