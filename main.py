import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random

from tqdm import tqdm


from bypass_bn import disable_running_stats, enable_running_stats
from sam import SAM
from model import *
from drive.utils import RMSE, get_optimizer
from drive.FADNet import DrivingNet, FADNet, FADNetFFA
from utils import *


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
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    else: 
        n_classes = 20
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)

            elif args.model=="FADNet":
                net = FADNet()

            elif args.model=="FADNetFFA":
                net = FADNetFFA()

            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net_fedsam(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    # net = nn.DataParallel(net)
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    
    cnt = 0
    # optimizer = get_optimizer(args_optimizer, net, lr)
    optimizer = SAM(
            net.parameters(),
            base_optimizer=torch.optim.Adam,
            lr=lr,
            weight_decay=args.reg
        )
    criterion = nn.MSELoss()
    metric = [RMSE]
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, _, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)

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

            # optimizer.zero_grad()
            # x.requires_grad = False
            # target.requires_grad = False
            # # target = target.long()

            # _,_,out = net(x)
            # loss = criterion(out.to(torch.float32), target)

            # loss.backward()
            # optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc, _ = compute_accuracy(net, train_dataloader, criterion, metric, device=device)
        #     test_acc, _ = compute_accuracy(net, test_dataloader, criterion, metric, device=device)

        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)
    if False:
        train_acc, _ = compute_accuracy(net, train_dataloader, criterion, metric, device=device) 
        test_acc, _ = compute_accuracy(net, test_dataloader, criterion, metric, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        train_acc, test_acc = 0., 0.
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedavg(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    # net = nn.DataParallel(net)
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    cnt = 0
    optimizer = get_optimizer(args_optimizer, net, lr)

    criterion = nn.MSELoss()
    metric = [RMSE]
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, _, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            # target = target.long()

            _,_,out = net(x)
            loss = criterion(out.to(torch.float32), target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc, _ = compute_accuracy(net, train_dataloader, criterion, metric, device=device)
        #     test_acc, _ = compute_accuracy(net, test_dataloader, criterion, metric, device=device)

        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)
    if False:
        train_acc, _ = compute_accuracy(net, train_dataloader, criterion, metric, device=device) 
        test_acc, _ = compute_accuracy(net, test_dataloader, criterion, metric, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        train_acc, test_acc = 0., 0.
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    # net = nn.DataParallel(net)
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    cnt = 0
    global_weight_collector = list(global_net.to(device).parameters())
    optimizer = get_optimizer(args_optimizer, net, lr)
    criterion = nn.MSELoss()
    metric = [RMSE]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, _, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            # target = target.long()

            _,_,out = net(x)
            loss = criterion(out.to(torch.float32), target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    if False:
        train_acc, _ = compute_accuracy(net, train_dataloader, criterion, metric, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        train_acc, test_acc = 0., 0.
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    # net = nn.DataParallel(net)
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    def old():
        return
        train_acc, _ = compute_accuracy(net, train_dataloader, criterion, metric, device=device)

        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        if args_optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        elif args_optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args_optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                                weight_decay=args.reg)

        criterion = nn.CrossEntropyLoss().to(device)
        # global_net.to(device)

    optimizer = get_optimizer(args_optimizer, net, lr)
    criterion = nn.MSELoss()
    metric = [RMSE]
    
    for previous_net in previous_nets:
        previous_net.to(device)
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    
    for epoch in range(epochs): 
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, _, target) in enumerate(train_dataloader):
            # import pdb;pdb.set_trace()
            x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            # target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net.to(device)
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros_like(logits).to(device).to(torch.float32) #.long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out.to(torch.float32), target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    if False:
        train_acc, _ = compute_accuracy(net, train_dataloader, criterion, metric, device=device)
        test_acc, _ = compute_accuracy(net, test_dataloader, criterion, metric, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        train_acc, test_acc = 0., 0.
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # if args_optimizer == 'adam':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    # elif args_optimizer == 'amsgrad':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
    #                            amsgrad=True)
    # elif args_optimizer == 'sgd':
    #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(args_optimizer, net, lr)
    criterion = nn.MSELoss()
    metric = [RMSE]
    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, _, target) in enumerate(tmp):
                x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                # target = target.long()
                _,_,out = net(x)
                loss = criterion(out.to(torch.float32), target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    if False:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else: 
        train_acc, test_acc = 0., 0.
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para


def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.MSELoss()
    metric = [RMSE]
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, _, target) in enumerate(tmp):
                x, target = x.to(device), target.unsqueeze(-1).to(device).to(torch.float32)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                # target = target.long()

                _,_,out = net(x)
                loss = criterion(out.to(torch.float32), target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
    if False:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        train_acc, test_acc, = 0., 0.

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.to(device)
    if server_c:
        server_c.to(device)
        server_c_collector = list(server_c.to(device).parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, device = device, dataidxs=dataidxs)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,  device = device)
        n_epoch = args.epochs

        if args.alg == 'fedavg':
            trainacc, testacc = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
        
        elif args.alg == 'fedfa':
            trainacc, testacc = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
        
        elif args.alg == 'fedsam':
            trainacc, testacc = train_net_fedsam(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
              
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device)

        elif args.alg == 'local_training':
            trainacc, testacc = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        # logger.info("net %d final test acc %f" % (net_id, testacc))
        # local testacc and trainacc is useless now, maybe useful latter
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))

    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        # noise_level = args.noise
        # if net_id == args.n_parties - 1:
        #     noise_level = 0

        # if args.noise_type == 'space':
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        # else:
        #     noise_level = args.noise / (args.n_parties - 1) * net_id
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)

        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, device = device, dataidxs=dataidxs)

        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        # logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        # noise_level = args.noise
        # if net_id == args.n_parties - 1:
        #     noise_level = 0

        # if args.noise_type == 'space':
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        # else:
        #     noise_level = args.noise / (args.n_parties - 1) * net_id
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        train_dl_local, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, device = device, dataidxs=dataidxs)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        # logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    # if args.log_file_name is None:
    #     argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    # else:
    #     argument_path = args.log_file_name + '.json'
    # with open(os.path.join(args.logdir, argument_path), 'w') as f:
    #     json.dump(str(args), f)

    device = torch.device(args.device)
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
    logger.info(device)
    logger.info(str(args))
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, device=device, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               train_bs = args.batch_size,
                                                                               test_bs = 32,
                                                                               device = device)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device=device)

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device=device)
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    criterion = nn.MSELoss()
    metric = [RMSE]

    def check_for_nan(state_dict):
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                print("exist nan in net param")
                return True
        return False

    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device=device)
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in tqdm(range(n_comm_rounds)):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, criterion, metric, device=device)
            test_acc, _ = compute_accuracy(global_model, test_dl, criterion, metric, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)


            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')

    elif args.alg == 'fedavg':
        for round in tqdm(range(n_comm_rounds)):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, criterion, metric, device=device)
            test_acc, _ = compute_accuracy(global_model, test_dl, criterion, metric, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            # torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            # torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
              
    
    elif args.alg == 'fedsam':
        for round in tqdm(range(n_comm_rounds)):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, criterion, metric, device=device)
            test_acc, _ = compute_accuracy(global_model, test_dl, criterion, metric, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            # mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            # torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            # torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'fedfa':
        for round in tqdm(range(n_comm_rounds)):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            clients = list(nets_this_round.values())
            # import pdb;pdb.set_trace()

            for key in global_model.state_dict().keys():
                tmp = torch.zeros_like(global_model.state_dict()[key]).float().to(device)
                for client_idx in range(len(fed_avg_freqs)):
                    tmp += fed_avg_freqs[client_idx] * clients[client_idx].to(device).state_dict()[key]
                global_model.state_dict()[key].data.copy_(tmp)
                if 'running_var_mean_bmic' in key or 'running_var_std_bmic' in key:
                    tmp = []
                    for client_idx in range(len(fed_avg_freqs)): 
                        tmp.append(clients[client_idx].state_dict()[key.replace('running_var_', 'running_')])
                    tmp = torch.stack(tmp)
                    var = torch.var(tmp)
                    global_model.state_dict()[key].data.copy_(var)
            # import pdb;pdb.set_trace()
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            test_acc, _ = compute_accuracy(global_model, test_dl, criterion, metric, device=device)

            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            global_model.to('cpu')
    
    elif args.alg == 'fedprox':

        for round in tqdm(range(n_comm_rounds)):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, criterion, metric, device=device)
            test_acc, _ = compute_accuracy(global_model, test_dl, criterion, metric, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')
    
    elif args.alg == 'scaffold':
        # logger.info("Initializing nets")
        # nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device=device)
        # global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device=device)
        # global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device=device)
        c_globals, _, _ = init_nets(args.net_config, 1, args, device=device)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)


        for round in tqdm(range(args.comm_round)):
            logger.info("in comm round:" + str(round))

            # arr = np.arange(args.n_parties)
            # np.random.shuffle(arr)
            # selected = arr[:int(args.n_parties * args.sample)]
            selected = party_list_rounds[round]
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                exist = check_for_nan(net_para)
                if exist:
                    continue
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)
            # import pdb;pdb.set_trace()

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.to(device)
            # train_acc, _ = compute_accuracy(global_model, train_dl_global, criterion, metric, device=device)
            
            test_acc, test_loss = compute_accuracy(global_model, test_dl, criterion, metric, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test loss: %f' % test_loss)

    elif args.alg == 'fednova':
        # logger.info("Initializing nets")
        # nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        # global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        # global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in tqdm(range(args.comm_round)):
            logger.info("in comm round:" + str(round))

            # arr = np.arange(args.n_parties)
            # np.random.shuffle(arr)
            # selected = arr[:int(args.n_parties * args.sample)]
            selected = party_list_rounds[round]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl, device=device)
            total_n = sum(n_list)
            #print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                exist = check_for_nan(d_para)
                if exist:
                    print("local model nan")
                    continue
                for key in d_para:
                    #if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    #else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n


            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                #print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            exist = check_for_nan(updated_model)
            global_model.load_state_dict(updated_model)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.to(device)
            # import pdb;pdb.set_trace()
            # train_acc, _ = compute_accuracy(global_model, train_dl_global, criterion, metric, device=device)
            # import pdb;pdb.set_trace()
            test_acc, test_loss = compute_accuracy(global_model, test_dl, criterion, metric, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test loss: %f' % test_loss)
            

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, epoch_size=len(train_ds_global), device=device)
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')

