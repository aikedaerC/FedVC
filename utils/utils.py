import os
import logging
import pickle
import paddle.nn.functional as F
import numpy as np
import paddle
from paddle.vision import transforms
import pickle as pkl
from paddle.io import Dataset

from sklearn.metrics import confusion_matrix
from paddle.io import DataLoader

import paddle.nn as nn

import random
from paddle.vision.datasets import Cifar10 as CIFAR10
from paddle.vision.datasets import Cifar100 as CIFAR100
from paddle.vision.datasets import MNIST


from time import sleep

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


# class CIFAR10_DIST(Dataset):

#     def __init__(self, root, client_num, experiment):
#         self.root = root
#         self.client_num = client_num
#         self.experiment = experiment
#         self.mean = [0.4914, 0.4822, 0.4465]
#         self.std = [0.2023, 0.1994, 0.2010]
#         self.train_ds_list = []
#         self.train_ds_label_list = []
#         normalize = transforms.Normalize(self.mean,self.std)
#         transform_train = transforms.Compose([
#             transforms.ToTensor()])

#         self.data = self._constract()
#         self.target = self.train_ds_label_list

#         self.transform = transform_train
#         self.target_transform = None


#     def _constract(self):
#         for j in range(self.client_num):
#             if self.client_num !=1:
#                 with open(os.path.join(self.root, "client_" + str(self.client_id + 1), "exp_" + str(self.experiment + 1),
#                                  "res_DM_CIFAR10_ConvNet_10ipc.pt"),"rb") as f:
#                     res = pkl.load(f)
#             else:
#                 with open(os.path.join(self.root, "data.pkl"),"rb") as f:
#                     res = pkl.load(f)
#             # how many data
#             if self.client_num !=1:
#                 num_data = len(res['data'][0][0])
#             else:
#                 num_data = 100
#             for i in range(num_data):
#                 if self.client_num != 1:
#                     if (res['data'][0][0][i] == torch.zeros((3, 32, 32)))[0][0][0] == True:
#                         continue
#                     image_syn_vis = res['data'][0][0][i]
#                 else:
#                     image_syn_vis = res[0][i]
#                 # for ch in range(3):
#                 #     image_syn_vis[:, ch] = image_syn_vis[:, ch] * self.std[ch] + self.mean[ch]
#                 # image_syn_vis[image_syn_vis < 0] = 0.0
#                 # image_syn_vis[image_syn_vis > 1] = 1.0
#                 self.train_ds_list.append(np.array(image_syn_vis))
#                 if self.client_num != 1:
#                     self.train_ds_label_list.append(res['data'][0][1][i])
#                 else:
#                     self.train_ds_label_list.append(res[1][i])

#         train_ds = np.vstack(self.train_ds_list).reshape((-1, 3, 32, 32))
#         # train_ds = train_ds.transpose((0, 2, 3, 1))

#         return train_ds

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.target[index]
#         # img = Image.fromarray(img)
#         # print("cifar10 img:", img)
#         # print("cifar10 target:", target)

#         # if self.transform is not None:
#         #     img = self.transform(img)
#         #
#         # if self.target_transform is not None:
#         #     target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.data)



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

def load_mnist_data():

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # data prep for test set
    transform_test = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST(mode='train', transform=transform_train)
    mnist_test_ds = MNIST(mode='test', transform=transform_test)

    X_train,y_train = np.array([x[0] for x in mnist_train_ds]), np.array([x[1][0] for x in mnist_train_ds])
    X_test,y_test = np.array([x[0] for x in mnist_test_ds]), np.array([x[1][0] for x in mnist_test_ds])
    
    return (mnist_train_ds,mnist_test_ds,X_train,y_train,X_test,y_test)

def load_cifar100_data(datadir):
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform_train = transforms.Compose([
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

    cifar100_train_ds = CIFAR100(data_file=os.path.join(datadir, 'cifar-100-python.tar.gz'), mode='train', transform=transform_train)
    cifar100_test_ds = CIFAR100(data_file=os.path.join(datadir, 'cifar-100-python.tar.gz'), mode='test', transform=transform_test)

    Train_data = cifar100_train_ds.data
    Test_data = cifar100_test_ds.data

    X_train,y_train = np.array([x[0] for x in Train_data]), np.array([x[1] for x in Train_data])
    X_test,y_test = np.array([x[0] for x in Test_data]), np.array([x[1] for x in Test_data])

    return (cifar100_train_ds,cifar100_test_ds,X_train,y_train,X_test,y_test)

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
        K = 10
    elif dataset == 'cifar100':
        _,_,X_train,y_train,X_test,y_test = load_cifar100_data(datadir)
        K = 100
    elif dataset == 'mnist':
        _,_,X_train,y_train,X_test,y_test = load_mnist_data()
        K = 10
    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

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

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def compute_accuracy(model, dataloader,dataset=None):

    # true_labels_list, pred_labels_list = np.array([]), np.array([])

    # evaluate model after one epoch
    model.eval()
    accuracies = []
    losses = []
    for batch_id, (x_data,y_data) in enumerate(dataloader()):

        y_data = paddle.to_tensor(y_data)
        if dataset != 'mnist':
            y_data = paddle.unsqueeze(y_data, 1)

        _,_,logits = model(x_data)
        loss = F.cross_entropy(logits, y_data)
        acc = paddle.metric.accuracy(logits, y_data)
        accuracies.append(acc.numpy())
        losses.append(loss.numpy())

    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    # print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))

    # if get_confusion_matrix:
    #     conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    #
    # if get_confusion_matrix:
    #     return correct/float(total), conf_matrix

    return avg_acc

#
# def save_model(model, model_index, args):
#     logger.info("saving local model-{}".format(model_index))
#     with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
#         torch.save(model.state_dict(), f_)
#     return
#
# def load_model(model, model_index, device="cpu"):
#     #
#     with open("trained_local_model"+str(model_index), "rb") as f_:
#         model.load_state_dict(torch.load(f_))
#     model.to(device)
#     return model


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY','cifar100'):
        if dataset == 'cifar10':
            train_ds,_,_,_,_,_ = load_cifar10_data(datadir)
            if dataidxs is not None:
                train_ds = [train_ds[i] for i in dataidxs]
            _,test_ds,_,_,_,_ = load_cifar10_data(datadir)
        elif dataset == 'cifar100':
            train_ds,_,_,_,_,_ = load_cifar100_data(datadir)
            if dataidxs is not None:
                train_ds = [train_ds[i] for i in dataidxs]
            _,test_ds,_,_,_,_ = load_cifar100_data(datadir)
        elif dataset == 'mnist':
            train_ds,_,_,_,_,_ = load_mnist_data()
            if dataidxs is not None:
                train_ds = [train_ds[i] for i in dataidxs]
            _,test_ds,_,_,_,_ = load_mnist_data()
        else:
            train_ds, test_ds = None, None


        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds

# ci = paddle.vision.datasets.Cifar10("cifar-10-python.tar.gz",mode='train',transform=paddle.vision.transforms.ToTensor())
# cid = paddle.io.DataLoader(ci,batch_size=22)

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


# def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
#     """
#     Sample random noise vector for training.
#
#     INPUT
#     --------
#     n_dis_c : Number of discrete latent code.
#     dis_c_dim : Dimension of discrete latent code.
#     n_con_c : Number of continuous latent code.
#     n_z : Dimension of iicompressible noise.
#     batch_size : Batch Size
#     device : GPU/CPU
#     """
#
#     z = torch.randn(batch_size, n_z, 1, 1, device=device)
#     idx = np.zeros((n_dis_c, batch_size))
#     if(n_dis_c != 0):
#         dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)
#
#         c_tmp = np.array(choice)
#
#         for i in range(n_dis_c):
#             idx[i] = np.random.randint(len(choice), size=batch_size)
#             for j in range(batch_size):
#                 idx[i][j] = c_tmp[int(idx[i][j])]
#
#             dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0
#
#         dis_c = dis_c.view(batch_size, -1, 1, 1)
#
#     if(n_con_c != 0):
#         # Random uniform between -1 and 1.
#         con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1
#
#     noise = z
#     if(n_dis_c != 0):
#         noise = torch.cat((z, dis_c), dim=1)
#     if(n_con_c != 0):
#         noise = torch.cat((noise, con_c), dim=1)
#
#     return noise, idx


def stop_epoch(time=3):
    try:
        print('can break now')
        for i in range(time):
            sleep(1)
        print('wait for next epoch')
        return False
    except KeyboardInterrupt:
        return True


# def compute_loss_accuracy(net, data_loader, criterion, device):
#     net.eval()
#     correct = 0
#     total_loss = 0.
#
#     with torch.no_grad():
#         for batch_idx, (inputs, labels) in enumerate(data_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = net(inputs)
#             total_loss += criterion(outputs, labels).item()
#             _, pred = outputs.max(1)
#             correct += pred.eq(labels).sum().item()
#
#     return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)
