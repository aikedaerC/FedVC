import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
# import torchvision.models as models
# from resnetcifar import ResNet18_cifar10, ResNet50_cifar10

# import torch.nn.init as init



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



class LambdaLayer(nn.Layer):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2D(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HiddenLayer(nn.Layer):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Layer):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        # x = F.sigmoid(x)
        # x = F.gelu(x)
        x = F.sigmoid_focal_loss(x)
        return x
