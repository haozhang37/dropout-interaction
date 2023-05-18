from trainer_cifar10 import *
import numpy as np
import argparse
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='Run tracker.')
parser.add_argument('net_name', type=str, default='ResNet56')
parser.add_argument('--p_info', type=str, default='0.0')# 0.5/0.0-1.0
parser.add_argument('--epochs', type=int, default=300)# 1000
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--set_name', type=str, default='CIFAR10')# CIFAR10/MNIST/tinyImageNet/CelebA
parser.add_argument('--num_classes', type=int, default=10)# 10 for CIFAR10/MNIST  10 for tinyImageNet  40 for CelebA
parser.add_argument('--pretrained', type=int, default=0)
parser.add_argument('--mode', type=str, default='test')

parser.add_argument('--pos_info', type=str, default='pos_100.npy')# None/pos.npy
parser.add_argument('--val_mode', type=str, default='trainset')# testset/trainset
parser.add_argument('--pos_pair', type=int, default=500)# 10000(200*399=79800)
parser.add_argument('--rate', type=float, default=1.0)
parser.add_argument('--target_rate', type=float, default=1.0)
parser.add_argument('--sample_set', type=str, default='0.05')
parser.add_argument('--S_rate', type=str, default='0.0,1.0')
parser.add_argument('--fixed_len', type=int, default=1)
parser.add_argument('--dropout_layer', type=int, default=3)
parser.add_argument('--sample_number', type=int, default=500)
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--seed_interaction', type=int, default=2)
parser.add_argument('--fraction', type=float, default=0.1)
parser.add_argument('--fraction_test', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument("--softmax", type=int, default=0)

conf = parser.parse_args()
conf.pretrained = bool(conf.pretrained)
#conf.grad = bool(conf.grad)
conf.fixed_len = bool(conf.fixed_len)
conf.set_seed = bool(conf.set_seed)
try:
    conf.p_info = float(conf.p_info)
except:
    conf.p_info = list(np.float64(conf.p_info.split(',')))

try:
    conf.sample_set = float(conf.sample_set)
except:
    conf.sample_set = list(np.float64(conf.sample_set.split(',')))

try:
    conf.S_rate = float(conf.S_rate)
except:
    conf.S_rate = list(np.float64(conf.S_rate.split(',')))

print(conf)
Trainer = trainer(conf)
Trainer.train()
# Trainer.compute_interaction(softmax=False)
# Trainer.compute_interaction(mode='banzhaf', softmax=False)
# for subseed in [1, 2, 3, 4, 5]:   # compute instability
#     Trainer.compute_interaction(softmax=False, mode="shapley", subseed=subseed)

# Trainer.compute_dataset_normalize()
# Trainer.compute_error_diff()
# Trainer.interval_interaction()

#Trainer.compute_input_interaction(mode='banzhaf')
