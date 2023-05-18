import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import time
import os


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, BasicBlock):
            weight_init(m)
        elif isinstance(m, Bottleneck):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.MaxPool2d):
            pass
        elif isinstance(m, nn.Flatten):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, LambdaLayer):
            pass
        else:
            m.initialize()

def load_pretrained(model, path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class DropoutShapley(nn.Module):
    def __init__(self, p_mode=0.5, inplace=False, mode='train', sample_set=0.05, fixed_len=True):
        super(DropoutShapley, self).__init__()
        self.p_mode = p_mode
        self.inplace = inplace
        self.mode = mode
        self.sample_set = sample_set
        self.fixed_len = fixed_len

    def forward(self, x, S_rate=[0.0, 1.0], cls_loss=True):
        p = self.generate_p()
        self.p = p
        if(self.mode == 'train'):
            if(self.training):
                return F.dropout(x, p=p, training=self.training, inplace=self.inplace)
            else:
                p = 0.5
                return F.dropout(x, p=p, training=self.training, inplace=self.inplace)
        elif(self.mode == 'test'):
            if(self.training):
                x_shape = x.shape
                x = torch.flatten(x, 1)
                if(cls_loss):
                    mask = torch.rand_like(x).to(x.device)
                    mask = (mask - p).sign()
                    mask = (mask + 1) / 2
                    return (x * mask.detach()).reshape(x_shape)

                if(self.fixed_len):
                    if(type(self.sample_set) == float):
                        sample_num     = min(max(1, int(x.shape[-1] * self.sample_set)), int(x.shape[-1] / 2 - 1))
                    elif(len(self.sample_set) == 2):
                        temp = torch.rand(1)[0] * (self.sample_set[1] - self.sample_set[0]) + self.sample_set[0]
                        sample_num = min(max(1, int(x.shape[-1] * temp)), int(x.shape[-1] / 2 - 1))
                    pos = np.random.choice(x.shape[-1], sample_num * 2, replace=False)
                    i = pos[:sample_num]
                    j = pos[sample_num:]
                else:
                    sample_num = min(max(1, int(x.shape[-1] * torch.rand(1)[0] * 0.05)), x.shape[1]-1)
                    pos = np.random.choice(x.shape[-1], sample_num, replace=False)
                    split = max(1, int(torch.rand(1)[0] * sample_num))
                    i = pos[:split]
                    j = pos[split:]

                maskImg_ij = []
                maskImg_i  = []
                maskImg_j  = []
                maskImg_   = []
                for count in range(10):
                    p_ = self.generate_p(S_rate)
                    mask_shapley = torch.rand_like(x).to(x.device)
                    mask_shapley = (mask_shapley - p_).sign()
                    mask_shapley = (mask_shapley + 1) / 2
                    mask_ij = deepcopy(mask_shapley)
                    mask_i  = deepcopy(mask_shapley)
                    mask_j  = deepcopy(mask_shapley)
                    mask_   = deepcopy(mask_shapley)
                    mask_ij[:,i] = 1
                    mask_ij[:,j] = 1
                    mask_i[:,i]  = 1
                    mask_i[:,j]  = 0
                    mask_j[:,i]  = 0
                    mask_j[:,j]  = 1
                    mask_[:,i]   = 0
                    mask_[:,j]   = 0
                    #x = x * mask.detach()
                    #return (x * mask_ij.detach()).reshape(x_shape), (x * mask_i.detach()).reshape(x_shape), (x * mask_j.detach()).reshape(x_shape), (x * mask_.detach()).reshape(x_shape)
                    maskImg_ij.append((x * mask_ij.detach()).reshape(x_shape))
                    maskImg_i.append((x * mask_i.detach()).reshape(x_shape))
                    maskImg_j.append((x * mask_j.detach()).reshape(x_shape))
                    maskImg_.append((x * mask_.detach()).reshape(x_shape))
                return torch.cat(maskImg_ij[:], 0), torch.cat(maskImg_i[:], 0), torch.cat(maskImg_j[:], 0), torch.cat(maskImg_[:], 0)
            else:
                if(type(self.p_mode) == float):
                    return x * (1 - self.p_mode)
                elif(len(self.p_mode) == 2):
                    return x * (1 - ((self.p_mode[1] + self.p_mode[0]) / 2))

    def generate_p(self, p=None):
        if(p is None):
            p = self.p_mode
        if(type(p) == float):
            return p
        elif(len(p) == 2):
            range_len = p[1] - p[0]
            return range_len * torch.rand(1)[0] + p[0]

    def initialize(self):
        pass


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True):
        super(AlexNet, self).__init__()
        if(set_name == 'CIFAR10'):
            out_size = 4
        elif(set_name == 'tinyImageNet' or set_name == 'tinyImageNet_corrupt'):
            out_size = 8
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(256 * out_size * out_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)
        '''
        self.linears = nn.Sequential(
#             DropoutShapley(p_mode, inplace=False, mode=mode),
            nn.Linear(256*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        '''
        weight_init(self)

    def forward(self, x, rate, S_rate=[0.0, 1.0], dropout_layer=8):
        x_cls = self.features[:dropout_layer](x)
        # x = torch.flatten(x, 1)
        t1 = time.time()
        if rate > 0.0:
            if (self.training):
                x_inter = self.features[:dropout_layer](x)
                x_cls = self.dropout(x_cls, S_rate, cls_loss=True)
                x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
                # return self.features[(dropout_layer+3):](x), self.features[(dropout_layer+3):](x_ij), self.features[(dropout_layer+3):](x_i), self.features[(dropout_layer+3):](x_j), self.features[(dropout_layer+3):](x_)
                cls_output = self.features[dropout_layer:](x_cls)
                inter_x_, inter_x_i, inter_x_j, inter_x_ij = self.features[dropout_layer:](x_), self.features[dropout_layer:](x_i), self.features[dropout_layer:](x_j), self.features[dropout_layer:](x_ij)
                return cls_output, inter_x_ij, inter_x_i, inter_x_j, inter_x_
            else:
                x = self.dropout(x_cls)
                t2 = time.time()
                # print(t1 - t0, t2 - t1)
                return self.features[dropout_layer:](x)
        else:
            x_cls = self.dropout(x_cls)
            x = self.features[dropout_layer:](x_cls)
            if self.training:
                return x, torch.zeros((x.size(0) * 10, x.size(1)), device=x.device, requires_grad=True), torch.zeros(
                    (x.size(0) * 10, x.size(1)), device=x.device), torch.zeros((x.size(0) * 10, x.size(1)),
                                                                               device=x.device), torch.zeros(
                    (x.size(0) * 10, x.size(1)), device=x.device)
            else:
                return x


class our_AlexNet(nn.Module):
    def __init__(self, num_classes=40, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True):
        super(our_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)
        weight_init(self)

    def forward(self, x, S_rate=[0.0, 1.0], dropout_layer=6):
        # x = self.features(x)
        x_cls = self.features[:dropout_layer](x)
        # x = torch.flatten(x, 1)
        if (self.training):
            x_inter = self.features[:dropout_layer](x)
            x_cls = self.dropout(x_cls, S_rate, cls_loss=True)
            x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
            # return self.features[(dropout_layer+2):](x), self.features[(dropout_layer+2):](x_ij), self.features[(dropout_layer+2):](x_i), self.features[(dropout_layer+2):](x_j), self.features[(dropout_layer+2):](x_)
            cls_output = self.features[dropout_layer:](x_cls)
            # self.eval()
            inter_x_ij = self.features[dropout_layer:](x_ij)
            inter_x_i = self.features[dropout_layer:](x_i)
            inter_x_j = self.features[dropout_layer:](x_j)
            inter_x_ = self.features[dropout_layer:](x_)
            # self.train()
            return cls_output, inter_x_ij, inter_x_i, inter_x_j, inter_x_
        else:
            x_cls= self.dropout(x_cls)
            x = self.features[dropout_layer:](x_cls)
            return x


class AlexNet_large(nn.Module):
    def __init__(self, num_classes=40, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True):
        super(AlexNet_large, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)
        weight_init(self)

    def forward(self, x, S_rate=[0.0, 1.0], dropout_layer=6):
        # x = self.features(x)
        x_cls = self.features[:dropout_layer](x)
        # x = torch.flatten(x, 1)
        if (self.training):
            x_inter = self.features[:dropout_layer](x)
            x_cls = self.dropout(x_cls, S_rate, cls_loss=True)
            x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
            x_cls = self.features[dropout_layer:](x_cls)
            x_cls = self.avgpool(x_cls)
            x_cls = torch.flatten(x_cls, 1)
            x_cls = self.classifier(x_cls)
            x_ij, x_i, x_j, x_ = self.features[dropout_layer:](x_ij), self.features[dropout_layer:](
                x_i), self.features[dropout_layer:](x_j), self.features[dropout_layer:](x_)
            x_ij, x_i, x_j, x_ = self.avgpool(x_ij), self.avgpool(x_i), self.avgpool(x_j), self.avgpool(x_)
            x_ij, x_i, x_j, x_ = torch.flatten(x_ij, 1), torch.flatten(x_i, 1), torch.flatten(x_j, 1), torch.flatten(x_, 1)
            x_ij, x_i, x_j, x_ = self.classifier(x_ij), self.classifier(x_i), self.classifier(x_j), self.classifier(x_)
            return x_cls, x_ij, x_i, x_j, x_
        else:
            x = self.dropout(x_cls)
            x = self.features[dropout_layer:](x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


class Origin_AlexNet(nn.Module):

    def __init__(self, num_classes=40, p_mode=0.5, mode="train", set_name="CelebA", sample_set=0.05, fixed_len=True):
        super(Origin_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)

    def forward(self, x, S_rate=[0.0, 1.0], dropout_layer=0):
        x_cls = self.features(x)
        x_cls = self.avgpool(x_cls)
        x_cls = torch.flatten(x_cls, 1)
        if (self.training):
            x_inter = self.features(x)
            x_inter = self.avgpool(x_inter)
            x_inter = torch.flatten(x_inter, 1)
            x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
            x_cls = self.dropout(x_cls, S_rate=S_rate, cls_loss=True)
            x_ij, x_i, x_j, x_ = self.classifier(x_ij), self.classifier(x_i), self.classifier(x_j), self.classifier(x_)
            x_cls = self.classifier(x_cls)
            return x_cls, x_ij, x_i, x_j, x_
        else:
            x = self.dropout(x_cls, S_rate=S_rate, cls_loss=True)
            x = self.classifier(x)
            return x


class LeNet(nn.Module):

    def __init__(self, num_classes=1, p_mode=0.5, mode='train', set_name='CIFAR10'):
        super(LeNet, self).__init__()
        if(set_name == 'CIFAR10'):
            imchannel = 3
            outsize=5
        elif(set_name == 'MNIST'):
            imchannel = 1
            outsize=4
        self.features = nn.Sequential(
            nn.Conv2d(imchannel, 6, kernel_size=5),# 28*28 / 24*24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),# 14*14 / 12*12
            nn.Conv2d(6, 16, kernel_size=5),# 10*10 / 8*8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),# 5*5 * 4*4
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode)
        self.linears = nn.Sequential(
            nn.Linear(16 * outsize * outsize, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )
        weight_init(self)

    def forward(self, x):
        x = self.features(x)
        self.mid_feature = torch.flatten(x, 1)
        if(self.training):
            x, x_ij, x_i, x_j, x_ = self.dropout(x, pos)
            return self.linears(x), self.linears(x_ij), self.linears(x_i), self.linears(x_j), self.linears(x_)
        else:
            x = self.dropout(x, pos)
            return self.linears(x)

class NetNet(nn.Module):

    def __init__(self, num_classes=10, p_mode=0.5, mode='train', set_name='CIFAR10'):
        super(NetNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 784),# 28*28 / 24*24
            nn.ReLU(inplace=True),
            nn.Linear(784, 784),# 28*28 / 24*24
            nn.ReLU(inplace=True),
        )
        self.linears = nn.Sequential(
            DropoutShapley(p_mode, inplace=False, mode=mode),
            nn.Linear(784, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100,  1),
        )
        weight_init(self)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.features(x)
        self.monitor = x
        x = self.linears(x)
        return x

def NetNet_(p_mode, pretrained=True, mode='test', set_name='MNIST', sample_set=0.05, fixed_len=True):
    net = NetNet(num_classes=1, p_mode=p_mode, mode=mode, set_name=set_name)
    return net

def LeNet_(p_mode, pretrained=True, mode='test', set_name='CIFAR10'):
    net = LeNet(num_classes=1, p_mode=p_mode, mode=mode, set_name=set_name)
    return net

def AlexNet_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=6):
    net = AlexNet(num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    if(pretrained):
        return load_pretrained(net, 'pretrained/alexnet-owt-4df8aa71.pth')
    else:
        return net

def AlexNet_large_(p_mode, pretrained=False, mode='test', set_name='CelebA', sample_set=0.05, fixed_len=True):
    net = our_AlexNet(num_classes=1, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    if(pretrained):
        return load_pretrained(net, 'pretrained/alexnet-owt-4df8aa71.pth')
    else:
        return net

def Origin_AlexNet_(p_mode, pretrained=False, mode='train', set_name='CelebA', sample_set=0.05, fixed_len=True):
    net = Origin_AlexNet(num_classes=1, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set,
                        fixed_len=fixed_len)
    if (pretrained):
        return load_pretrained(net, 'pretrained/alexnet-owt-4df8aa71.pth')
    else:
        return net

class VGG(nn.Module):

    def __init__(self, features, num_classes=1, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)
        # if init_weights:
        self._initialize_weights()

    def forward(self, x, S_rate=[0.0, 1.0], dropout_layer=15):
        x_cls = self.features[:dropout_layer](x)
        if (self.training):
            x_inter = self.features[:dropout_layer](x)
            x_cls = self.dropout(x_cls, S_rate, cls_loss=True)
            x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
            x_cls = self.features[dropout_layer:](x_cls)
            x_cls = self.avgpool(x_cls)
            x_cls = torch.flatten(x_cls, 1)
            x_cls = self.classifier(x_cls)
            x_ij, x_i, x_j, x_ = self.features[dropout_layer:](x_ij), self.features[dropout_layer:](
                x_i), self.features[dropout_layer:](x_j), self.features[dropout_layer:](x_)
            x_ij, x_i, x_j, x_ = self.avgpool(x_ij), self.avgpool(x_i), self.avgpool(x_j), self.avgpool(x_)
            x_ij, x_i, x_j, x_ = torch.flatten(x_ij, 1), torch.flatten(x_i, 1), torch.flatten(x_j, 1), torch.flatten(x_,
                                                                                                                     1)
            x_ij, x_i, x_j, x_ = self.classifier(x_ij), self.classifier(x_i), self.classifier(x_j), self.classifier(x_)
            return x_cls, x_ij, x_i, x_j, x_
        else:
            x = self.dropout(x_cls)
            x = self.features[dropout_layer:](x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class our_VGG(nn.Module):

    def __init__(self, cfg, num_classes=1, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=0.0, S_rate=[0.0, 1.0], dropout_layer=5):
        super(our_VGG, self).__init__()
        conv_layers = self._make_layers(cfg)
        if (set_name == 'tinyImageNet'):
            out_size = 2
        elif (set_name == 'CelebA' or set_name == "corrupt"):
            out_size = 7
        elif (set_name == 'CIFAR10'):
            out_size = 1

        # self.features = feature
        # length = len(feature)
        # self.features.add_module(str(length), nn.AdaptiveAvgPool2d((7, 7)))
        # self.features.add_module(str(length + 1), nn.Flatten())
        # self.features.add_module(str(length + 2), nn.Linear(512 * 7 * 7, 4096))
        # self.features.add_module(str(length + 3), nn.ReLU(True))
        # self.features.add_module(str(length + 4), nn.Linear(4096, 4096))
        # self.features.add_module(str(length + 5), nn.ReLU(True))
        # self.features.add_module(str(length + 6), nn.Linear(4096, num_classes))
        self.features = nn.Sequential((*conv_layers),
                                      nn.AdaptiveAvgPool2d((out_size, out_size)),
                                      nn.Flatten(),
                                      nn.Linear(512 * out_size * out_size, 4096),
                                      nn.ReLU(True),
                                      nn.Linear(4096, 4096),
                                      nn.ReLU(True),
                                      nn.Linear(4096, num_classes),
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)
        # if init_weights:
        self._initialize_weights()

    def forward(self, x, rate, S_rate=[0.0, 1.0], dropout_layer=15):
        x_cls = self.features[:dropout_layer](x)
        if rate > 0.0:
            if (self.training):
                x_inter = self.features[:dropout_layer](x)
                x_cls = self.dropout(x_cls, S_rate, cls_loss=True)
                x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
                # return self.features[(dropout_layer+2):](x), self.features[(dropout_layer+2):](x_ij), self.features[(dropout_layer+2):](x_i), self.features[(dropout_layer+2):](x_j), self.features[(dropout_layer+2):](x_)
                cls_output = self.features[dropout_layer:](x_cls)
                self.eval()
                inter_x_ij = self.features[dropout_layer:](x_ij)
                inter_x_i = self.features[dropout_layer:](x_i)
                inter_x_j = self.features[dropout_layer:](x_j)
                inter_x_ = self.features[dropout_layer:](x_)
                self.train()
                return cls_output, inter_x_ij, inter_x_i, inter_x_j, inter_x_
            else:
                x_cls = self.dropout(x_cls)
                x = self.features[dropout_layer:](x_cls)
                return x
        else:
            x_cls = self.dropout(x_cls)
            x = self.features[dropout_layer:](x_cls)
            if self.training:
                return x, torch.zeros((x.size(0) * 10, x.size(1)), device=x.device, requires_grad=True), torch.zeros((x.size(0) * 10, x.size(1)), device=x.device), torch.zeros((x.size(0) * 10, x.size(1)), device=x.device), torch.zeros((x.size(0) * 10, x.size(1)), device=x.device)
            else:
                return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return layers


class Origin_VGG(nn.Module):

    def __init__(self, features, num_classes=1, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True):
        super(Origin_VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)
        # if init_weights:
        self._initialize_weights()

    def forward(self, x, S_rate=[0.0, 1.0], dropout_layer=15):
        x_cls = self.features(x)
        x_cls = self.avgpool(x_cls)
        x_cls = torch.flatten(x_cls, 1)
        if (self.training):
            x_inter = self.features(x)
            x_inter = self.avgpool(x_inter)
            x_inter = torch.flatten(x_inter, 1)
            x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
            x_cls = self.dropout(x_cls, S_rate=S_rate, cls_loss=True)
            x_ij, x_i, x_j, x_ = self.classifier(x_ij), self.classifier(x_i), self.classifier(x_j), self.classifier(x_)
            x_cls = self.classifier(x_cls)
            return x_cls, x_ij, x_i, x_j, x_
        else:
            x = self.dropout(x_cls, S_rate=S_rate, cls_loss=True)
            x = self.classifier(x)
            return x
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

'''
def _vgg(arch, cfg, batch_norm, pretrained, progress, p_mode, mode='test', num_classes=10, set_name='tinyImageNet', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1, seed=2, bs=64):
    # if pretrained:
    #     kwargs['init_weights'] = False
    model = our_VGG(cfgs[cfg], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    if(pretrained):
        # epochs = [800, 500, 300, 240, 200, 160, 130, 100, 70, 50, 25, 15, 8, 5, 2]
        epoch = 299
        path_exist = False
        for f in sorted(os.listdir(f'./results_dropout/{arch}{set_name}_0.1_epoch_300_bs_{bs}_lr_0.01/')):
            f_list = f.split('_')
            if((str(rate)==f_list[0]) and (str(p_mode)==f_list[1]) and (str(fixed_len)==f_list[2]) and (str(sample_set)==f_list[3]) and (str(S_rate)==f_list[4]) and (str(dropout_layer)==f_list[5]) and (str(f_list[6])=='%04d'%(epoch+1)) and (str(seed)==(f_list[-1].split('.'))[0])):
                filename = f
                path_exist = True
                print(filename)
                break
        # if(path_exist):
        #     break

        return load_pretrained(model, os.path.join(f'./results_dropout/{arch}{set_name}_0.1_epoch_300_bs_{bs}_lr_0.01/', filename)), (epoch+1)
    else:
        return model
'''
    # if pretrained:
    #     raise RuntimeError("Only use unpretrained model for initial training")
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    # return model

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = our_VGG(cfgs[cfg], **kwargs)
    if pretrained:
        raise RuntimeError("Only use unpretrained model for initial training")
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def _origin_vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = Origin_VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        raise RuntimeError("Only use unpretrained model for initial training")
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def vgg11_(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)



def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)



def vgg13_(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)



def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)



def vgg16_(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_', 'D', False, pretrained, progress, **kwargs)


def origin_vgg16_(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _origin_vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)



def vgg19_(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def origin_vgg19_(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _origin_vgg('vgg19', 'E', False, pretrained, progress, **kwargs)



def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)

## ============ResNet========================================================================
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )

    def forward(self, x):
        return nn.Softplus()(self.residual_function(x) + self.shortcut(x))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, option='B'):
        super(Bottleneck, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet_CIFAR10(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True):
        super(ResNet_CIFAR10, self).__init__()
        self.in_planes = 16

        layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            (*layer1),
            (*layer2),
            (*layer3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes))

        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)

#        weight_init(self)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option='A'))
            self.in_planes = planes * block.expansion

        return layers

    def forward(self, x, rate, S_rate=[0.0, 1.0], dropout_layer=1):
        x_cls = self.features[:(dropout_layer + 3)](x)
        # x = torch.flatten(x, 1)
        t1 = time.time()
        if rate > 0.0:
            if (self.training):
                x_inter = self.features[:(dropout_layer + 3)](x)
                x_cls = self.dropout(x_cls, S_rate, cls_loss=True)
                x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
                # return self.features[(dropout_layer+3):](x), self.features[(dropout_layer+3):](x_ij), self.features[(dropout_layer+3):](x_i), self.features[(dropout_layer+3):](x_j), self.features[(dropout_layer+3):](x_)
                cls_output = self.features[(dropout_layer + 3):](x_cls)
                self.eval()
                inter_x_, inter_x_i, inter_x_j, inter_x_ij = self.features[(dropout_layer + 3):](x_), self.features[(
                                                                                                                                dropout_layer + 3):](
                    x_i), self.features[(dropout_layer + 3):](x_j), self.features[(dropout_layer + 3):](x_ij)
                self.train()
                return cls_output, inter_x_ij, inter_x_i, inter_x_j, inter_x_
            else:
                x = self.dropout(x_cls)
                t2 = time.time()
                # print(t1 - t0, t2 - t1)
                return self.features[(dropout_layer + 3):](x)
        else:
            x_cls = self.dropout(x_cls)
            x = self.features[(dropout_layer + 3):](x_cls)
            if self.training:
                return x, torch.zeros((x.size(0) * 10, x.size(1)), device=x.device, requires_grad=True), torch.zeros(
                    (x.size(0) * 10, x.size(1)), device=x.device), torch.zeros((x.size(0) * 10, x.size(1)),
                                                                               device=x.device), torch.zeros(
                    (x.size(0) * 10, x.size(1)), device=x.device)
            else:
                return x

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10, p_mode=0.5, mode='train', set_name='CIFAR10', sample_set=0.05, fixed_len=True):
        super(ResNet, self).__init__()
        self.in_channels = 64

        conv2_x = self._make_layer(block, 64, num_block[0], 1)
        conv3_x = self._make_layer(block, 128, num_block[1], 2)
        conv4_x = self._make_layer(block, 256, num_block[2], 2)
        conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            (*conv2_x),
            (*conv3_x),
            (*conv4_x),
            (*conv5_x),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes))

        self.dropout = DropoutShapley(p_mode, inplace=False, mode=mode, sample_set=sample_set, fixed_len=fixed_len)

        weight_init(self)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, option='B'))
            self.in_channels = out_channels * block.expansion

        return layers

    def forward(self, x, rate, S_rate=[0.0, 1.0], dropout_layer=1):
        x_cls = self.features[:(dropout_layer+3)](x)
        #x = torch.flatten(x, 1)
        t1 = time.time()
        if rate > 0.0:
            if(self.training):
                x_inter = self.features[:(dropout_layer + 3)](x)
                x_cls = self.dropout(x_cls, S_rate, cls_loss=True)
                x_ij, x_i, x_j, x_ = self.dropout(x_inter, S_rate, cls_loss=False)
                #return self.features[(dropout_layer+3):](x), self.features[(dropout_layer+3):](x_ij), self.features[(dropout_layer+3):](x_i), self.features[(dropout_layer+3):](x_j), self.features[(dropout_layer+3):](x_)
                cls_output = self.features[(dropout_layer+3):](x_cls)
                self.eval()
                inter_x_, inter_x_i, inter_x_j, inter_x_ij = self.features[(dropout_layer+3):](x_), self.features[(dropout_layer+3):](x_i), self.features[(dropout_layer+3):](x_j), self.features[(dropout_layer+3):](x_ij)
                self.train()
                return cls_output, inter_x_ij, inter_x_i, inter_x_j, inter_x_
            else:
                x = self.dropout(x_cls)
                t2 = time.time()
                # print(t1 - t0, t2 - t1)
                return self.features[(dropout_layer+3):](x)
        else:
            x_cls = self.dropout(x_cls)
            x = self.features[(dropout_layer+3):](x_cls)
            if self.training:
                return x, torch.zeros((x.size(0) * 10, x.size(1)), device=x.device, requires_grad=True), torch.zeros((x.size(0) * 10, x.size(1)), device=x.device), torch.zeros((x.size(0) * 10, x.size(1)), device=x.device), torch.zeros((x.size(0) * 10, x.size(1)), device=x.device)
            else:
                return x

def ResNet18_(p_mode, pretrained=True, mode='test', num_classes=1, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet(block=BasicBlock, num_block=[2, 2, 2, 2], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet34_(p_mode, pretrained=True, mode='test', num_classes=1, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet(block=BasicBlock, num_block=[3, 4, 6, 3], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet50_(p_mode, pretrained=True, mode='test', num_classes=1, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet(block=Bottleneck, num_block=[3, 4, 6, 3], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet101_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet(block=Bottleneck, num_block=[3, 4, 23, 3], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet152_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet(block=Bottleneck, num_block=[3, 8, 36, 3], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet20_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet_CIFAR10(block=BasicBlock, num_blocks=[3, 3, 3], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet32_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet_CIFAR10(block=BasicBlock, num_blocks=[5, 5, 5], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet44_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet_CIFAR10(block=BasicBlock, num_blocks=[7, 7, 7], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet56_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet_CIFAR10(block=BasicBlock, num_blocks=[9, 9, 9], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net

def ResNet110_(p_mode, pretrained=True, mode='test', num_classes=10, set_name='CIFAR10', sample_set=0.05, fixed_len=True, rate=5, S_rate=[0.0, 1.0], dropout_layer=1):
    net = ResNet_CIFAR10(block=BasicBlock, num_blocks=[18, 18, 18], num_classes=num_classes, p_mode=p_mode, mode=mode, set_name=set_name, sample_set=sample_set, fixed_len=fixed_len)
    return net
