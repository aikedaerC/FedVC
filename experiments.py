import logging
import os
import paddle
import paddle.nn as nn
import numpy as np
from paddle import optimizer as optim
import argparse
import copy

import paddle.vision as transforms
from paddle.io import Dataset
import pickle as pkl


import datetime
from tensorboardX import SummaryWriter

from model.model import ConvNet

import paddle.nn.functional as F


from utils.utils import compute_accuracy, get_dataloader, partition_data, mkdirs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ConvNet', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=120, help='number of local epochs 1')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='moon',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False,
                        help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=5e-4, help="L2 regularization strength, weight decay")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=5, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0.9, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',
                        help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--save_dir', type=str, default='model_save', help='model saved after training')

    ## meta learning
    # parser.add_argument('--meta_net_hidden_size', type=int, default=100)
    # parser.add_argument('--meta_net_num_layers', type=int, default=1)

    # parser.add_argument('--dampening', type=float, default=0.)
    # parser.add_argument('--nesterov', type=bool, default=False)
    # parser.add_argument('--meta_lr', type=float, default=1e-5)
    # parser.add_argument('--meta_weight_decay', type=float, default=0.)

    # parser.add_argument('--num_meta', type=int, default=1000)
    # parser.add_argument('--imbalanced_factor', type=int, default=None)
    # parser.add_argument('--corruption_type', type=str, default=None)
    # parser.add_argument('--corruption_ratio', type=float, default=0.)

    # parser.add_argument('--meta_interval', type=int, default=1)
    # parser.add_argument('--paint_interval', type=int, default=20)

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    _nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        if args.model == 'ConvNet':
            if args.dataset == 'mnist':
                _net = ConvNet(1, 10, 128, 3, 'relu', 'instancenorm', 'avgpooling', (28, 28))
            elif args.dataset == 'cifar100':
                _net = ConvNet(3, 100, 128, 3, 'relu', 'instancenorm', 'avgpooling', (32, 32))
            elif args.dataset == 'cifar10':
                _net = ConvNet(3, 10, 128, 3, 'relu', 'instancenorm', 'avgpooling', (32, 32))
        else:
            print("not supported yet")
            exit(1)
        _nets[net_i] = _net

    model_meta_data = []
    _layer_type = []
    for (k, v) in _nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        _layer_type.append(k)
    return _nets, model_meta_data, _layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer):
    # logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)
    else:
        optimizer = optim.SGD(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)

    cnt = 0
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader()):
            _, _, out = net(x)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    test_acc = compute_accuracy(net, test_dataloader, dataset=args.dataset)
    logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc

def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu):
    # logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)
    else:
        optimizer = optim.SGD(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader()):

            _, _, out = net(x)
            loss = F.cross_entropy(out, target)
            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * paddle.norm((param - global_weight_collector[param_index]),p=2) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc = compute_accuracy(net, test_dataloader, dataset=args.dataset)
    logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                   args_optimizer, mu, temperature, args, round):
    # logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)
    else:
        optimizer = optim.SGD(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)

    cnt = 0
    cos = nn.CosineSimilarity(axis=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader()):

            _, pro1, out_y = net(x)
            _, pro2, _ = global_net(x)

            if args.loss == 'l2norm':
                loss2 = mu * np.mean(np.linalg.norm(pro2 - pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape((-1, 1))

                for previous_net in previous_nets:
                    _, pro3, _ = previous_net(x)

                    nega = cos(pro1, pro3)
                    logits = np.concatenate((logits, nega.reshape((-1, 1))), axis=1)
                logits /= temperature
                logits = paddle.to_tensor(logits,dtype='float32',stop_gradient=False)
                labels = paddle.to_tensor(np.zeros((logits.shape[0],1)),dtype='int32',stop_gradient=False)

                loss2 = mu * F.cross_entropy(logits, labels, soft_label=False, axis=-1, weight=None, reduction='mean')

            if args.loss == 'only_contrastive':
                loss_ = loss2
            else:
                loss1 = F.cross_entropy(out_y, target)
                loss_ = loss1 + loss2

            loss_.backward()
            optimizer.step()
            optimizer.clear_grad()

            cnt += 1
            epoch_loss_collector.append(loss_.item())
            epoch_loss2_collector.append(loss2.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Total Loss: %.4f Contrastive Loss: %.4f' % (epoch, epoch_loss,epoch_loss2))


    test_acc = compute_accuracy(net, test_dataloader, dataset=args.dataset)
    logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader():
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, test_dl=None):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer)
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=None):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu)

        avg_acc += testacc
    avg_acc /= len(selected)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model=None, prev_model_pool=None,
                         round=None):
    avg_acc = 0.0
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, 
                                           test_dl, n_epoch, args.lr, args.optimizer, args.mu, args.temperature, args, round)

        avg_acc += testacc

    avg_acc /= len(selected)
    nets_list = list(nets.values())
    return nets_list


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    # paddle.set_device("cpu")
    # mkdirs(args.modeldir)
    # if args.log_file_name is None:
    #     argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    # else:
    #     argument_path = args.log_file_name + '.json'
    # with open(os.path.join(args.logdir, argument_path), 'w') as f:
    #     json.dump(str(args), f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logs_path = os.path.join(args.logdir, args.dataset, "beta" + str(args.beta), "seed_" + str(args.init_seed), args.alg, "mu_" + str(args.mu) ,"global")
    os.makedirs(logs_path, exist_ok=True)  # exist_ok=True to override the already exists dir.
    writer = SummaryWriter(logs_path)

    # experiment_name = args.alg + "_" + args.partition + (
    #     str(args.beta) if args.partition == 'noniid-labeldir' else "") + "_" + "mu" + str(args.mu) + "_" +args.dataset + "_" + "parties" +"_"+str(args.n_parties)
    #
    # logger.info(experiment_name)
    # wandb_log_dir = os.path.join(args.logdir, experiment_name)
    # if not os.path.exists('{}'.format(wandb_log_dir)):
    #     os.makedirs('{}'.format(wandb_log_dir))
    # wandb.init(entity='aikedaer', project="Summer_Last",
    #            group=args.partition + (str(args.beta) if args.partition == 'noniid-labeldir' else ""),
    #            job_type=args.alg, dir=wandb_log_dir)
    # wandb.run.name = experiment_name
    # wandb.run.save()
    # wandb.config.update(args)
    logger.info(args)
    logger.info("#" * 100)
    seed = args.init_seed

    np.random.seed(seed)

    logger.info("Partitioning data")

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)

    print("len train_ds_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            train_dl_local, test_dl, train_ds_local, test_ds = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, 32,
                                                                                              dataidxs)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds)


    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.set_state_dict(global_para)

        # wandb.watch(global_model)
        for round in range(args.comm_round):
            # wandb_dict = {}
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].set_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].set_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl=test_dl_global)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.set_state_dict(global_para)

            # train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc = compute_accuracy(global_model, test_dl_global, dataset=args.dataset)

            # wandb_dict[args.alg + "train_acc"] = train_acc
            # wandb_dict[args.alg + 'test_acc'] = test_acc
            #
            # wandb.log(wandb_dict)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.warning('>> Global Model Test accuracy: %f' % test_acc)

            # writer.add_scalar("Train_Acc", train_acc, round)
            writer.add_scalar("Test_Acc", test_acc, round)

    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.set_state_dict(global_para)
        # wandb.watch(global_model)
        for round in range(args.comm_round):

            # wandb_dict = {}
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].set_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].set_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=test_dl_global)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.set_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            test_acc = compute_accuracy(global_model, test_dl_global, dataset=args.dataset)

            # wandb_dict[args.alg + "train_acc"] = train_acc
            # wandb_dict[args.alg + 'test_acc'] = test_acc
            #
            # wandb.log(wandb_dict)

            logger.warning('>> Global Model Test accuracy: %f' % test_acc)
            writer.add_scalar("Test_Acc", test_acc, round)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.set_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        # wandb.watch(global_model)
        for round in range(args.comm_round):
            # wandb_dict = {}
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].set_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].set_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=test_dl_global,
                                 global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.set_state_dict(global_para)

            test_acc = compute_accuracy(global_model, test_dl_global, dataset=args.dataset)

            # wandb_dict[args.alg + "train_acc"] = train_acc
            # wandb_dict[args.alg + 'test_acc'] = test_acc
            #
            # wandb.log()
            logger.warning('>> Global Model Test accuracy: %f' % test_acc)
            writer.add_scalar("Test_Acc", test_acc, round)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets


