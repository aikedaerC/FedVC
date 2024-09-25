import numpy as np

import torch
import torch.nn as nn


def RMSE(y_preds, y_trues):
    with torch.no_grad():
        return torch.sqrt(torch.mean((y_preds - y_trues)**2))

import torch.optim as optim
import numpy as np
# import torch_optimizer as upgrade_optim


def get_optimizer(optimizer_name, net, lr_initial=1e-3):
    """

    :param optimizer_name:
    :param net:
    :param lr_initial:
    :return:
    """
    if optimizer_name == "adam":
        return optim.Adam([param for param in net.parameters() if param.requires_grad], lr=lr_initial, weight_decay=1e-5)

    elif optimizer_name == "sgd":
        return optim.SGD([param for param in net.parameters() if param.requires_grad], lr=lr_initial)

    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, epoch_size):
    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1/np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cyclic":
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=0.1)

    elif scheduler_name == "custom":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30*int(epoch_size), gamma=0.1)
    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")



def get_model(name, model, device, epoch_size, optimizer_name="adam", lr_scheduler="custom",
              initial_lr=1e-3, seed=1234):
    """
    Load Model object corresponding to the experiment
    :param name: experiment name; possible are: "driving" in name such as driving_carla, driving_gazebo
    :param device:
    :param epoch_size:
    :param optimizer_name: optimizer name, for now only "adam" is possible
    :param lr_scheduler:
    :param initial_lr:
    :param seed:
    :return: Model object
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if "driving" in name:
        criterion = nn.MSELoss()
        metric = [RMSE]
        return DrivingNet(model, criterion, metric, device, optimizer_name, lr_scheduler, initial_lr, epoch_size)
    else:
        raise NotImplementedError
