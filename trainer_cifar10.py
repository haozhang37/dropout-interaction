import importlib
import time
from tqdm import tqdm
import os
import math
from copy import deepcopy
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import MNISTuser, CIFAR10Subset, tinyImageNetSubset, tinyImageNetSubset_corrupt
from loss import EasyLoss, GenerateLoss

def set_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class trainer():
    def __init__(self, conf):
        self.set_seed = conf.set_seed
        self.seed = conf.seed
        self.seed_interaction = conf.seed_interaction
        if(self.set_seed):
            set_seed_pytorch(self.seed)

        self.set_name    = conf.set_name
        self.num_classes = conf.num_classes
        self.val_mode    = conf.val_mode
        self.pos_info    = conf.pos_info
        self.pos_pair    = conf.pos_pair
        self.sample_set  = conf.sample_set
        self.S_rate      = conf.S_rate
        self.fixed_len   = conf.fixed_len
        self.fraction    = conf.fraction
        self.fraction_test = conf.fraction_test
        #self.transform   = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if(conf.set_name == 'CIFAR10'):
            self.transform   = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transform_val   = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #self.trainset    = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=True, download=False, transform=self.transform)
            self.trainset    = CIFAR10Subset(root='./data/CIFAR10', train=True, download=True, fraction=self.fraction, transform=self.transform)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=conf.bs, shuffle=True, num_workers=1)
            self.testset     = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=self.transform_val)
            #self.testset     = CIFAR10Subset(root='./data/CIFAR10', train=False, download=False, fraction_test=self.fraction_test, transform=self.transform_val)
            self.testloader  = torch.utils.data.DataLoader(self.testset, batch_size=conf.bs, shuffle=False, num_workers=1)
        elif(conf.set_name == 'MNIST'):
            self.transform   = transforms.Compose([transforms.ToTensor()])
            self.trainset    = MNISTuser(root='data/MNIST', train=True, download=False, transform=self.transform)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=conf.bs, shuffle=True, num_workers=1)
            self.testset     = MNISTuser(root='data/MNIST', train=False, download=False, transform=self.transform)
            self.testloader  = torch.utils.data.DataLoader(self.testset, batch_size=conf.bs, shuffle=False, num_workers=1)
        elif(conf.set_name == 'tinyImageNet'):
            self.transform   = transforms.Compose([transforms.RandomCrop((64,64), padding=(8, 8, 8, 8), padding_mode='reflect'), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #self.transform   = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transform_val   = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dir = './data/tiny-imagenet-200/train/'
            val_dir   = './data/tiny-imagenet-200/val/'
            #self.trainset    = torchvision.datasets.ImageFolder(train_dir, transform=self.transform)
            self.trainset    = tinyImageNetSubset(root=train_dir, state='train', fraction=self.fraction, transform=self.transform)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=conf.bs, shuffle=True, num_workers=1)
            #self.testset     = torchvision.datasets.ImageFolder(val_dir, transform=self.transform_val)
            self.testset     = tinyImageNetSubset(root=val_dir, state='test', fraction=self.fraction_test, transform=self.transform_val)
            self.testloader  = torch.utils.data.DataLoader(self.testset, batch_size=conf.bs, shuffle=False, num_workers=1)
        elif(conf.set_name == 'tinyImageNet_corrupt'):
            self.transform   = transforms.Compose([transforms.RandomCrop((64,64), padding=(8, 8, 8, 8), padding_mode='reflect'), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #self.transform   = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transform_val   = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dir = './data/tiny-imagenet-200/train/'
            val_dir   = './data/tiny-imagenet-200/val/'
            #self.trainset    = torchvision.datasets.ImageFolder(train_dir, transform=self.transform)
            self.trainset    = tinyImageNetSubset_corrupt(root=train_dir, state='train', fraction=self.fraction, seed=self.seed, transform=self.transform)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=conf.bs, shuffle=True, num_workers=1)
            #self.testset     = torchvision.datasets.ImageFolder(val_dir, transform=self.transform_val)
            self.testset     = tinyImageNetSubset_corrupt(root=val_dir, state='test', fraction=self.fraction_test, seed=self.seed, transform=self.transform_val)
            self.testloader  = torch.utils.data.DataLoader(self.testset, batch_size=conf.bs, shuffle=False, num_workers=1)

        if(self.val_mode=='trainset'):
            self.valloader   = torch.utils.data.DataLoader(self.trainset, batch_size=1, shuffle=False, num_workers=1)
        elif(self.val_mode=='testset'):
            self.valloader   = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=1)

        self.gpu         = conf.gpu
        self.epochs      = conf.epochs
        self.net_name    = conf.net_name
        self.p_info      = conf.p_info
        self.pretrained  = conf.pretrained
        self.mode        = conf.mode
        self.rate        = conf.rate
        self.dropout_layer = conf.dropout_layer
        self.sample_number = conf.sample_number
        self.target_rate = conf.target_rate
        if(self.pretrained):
            self.lr = 0.0001
        else:
            self.lr = conf.lr
        self.bs = conf.bs
        lib = importlib.import_module('lib')
        net_build = getattr(lib, self.net_name)
        net = net_build(p_mode=self.p_info, pretrained=self.pretrained, mode=self.mode, num_classes=self.num_classes, set_name=self.set_name, sample_set=self.sample_set, fixed_len=self.fixed_len, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
        #net = net_build()
        self.pretrained_epoch = 0
        if(self.pretrained):
            self.pretrained_epoch = net[1]
            net = net[0]
        self.net = net.cuda(self.gpu)
        # self.softplus = conf.softplus
        # if(self.softplus):
        #     replace_relu_to_softplus(self.net)
        self.softmax = conf.softmax
        self.criterion = GenerateLoss(r=self.rate, tr=self.target_rate, softmax=self.softmax)
        #self.criterion = nn.CrossEntropyLoss()
#         self.criterion   = nn.CrossEntropyLoss()
#         self.optimizer   = optim.Adam(net.parameters(), lr=0.0001)
        self.optimizer   = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.result = 0
        self.result_train = 0

        self.grad = [[], [], [], [], [], [], [], [], [], []]
        self.grad_epochs = []
        if not self.softmax:
            self.sub_folder = f"{self.net_name}{self.set_name}_{self.fraction}_epoch_{self.epochs}_bs_{self.bs}_lr_{self.lr}"
        else:
            self.sub_folder = f"{self.net_name}{self.set_name}_{self.fraction}_epoch_{self.epochs}_bs_{self.bs}_lr_{self.lr}_softmax"

    def train(self, resume=None, inter_batch=5):
        losses_check = []
        grad_epoch = []
        log_lr_st = math.log10(self.lr)
        lr_epoch = torch.logspace(log_lr_st, log_lr_st - 2, steps=self.epochs)
        count = 1
        # print(time.asctime(time.localtime(time.time())))
        for epoch in range(self.pretrained_epoch, self.epochs):
            set_seed_pytorch(self.seed + epoch)
            self.result_train = 0
            losses_epoch = []
            self.net.train()
            correct_train = 0
            total_train = 0
            interloss_avg = 0
            for i, data in enumerate(tqdm(self.trainloader)):
                if('corrupt' in self.set_name):
                    inputs, labels, image_path = data
                else:
                    inputs, labels = data
                inputs, labels = Variable(inputs).cuda(self.gpu), Variable(labels).cuda(self.gpu)
                outputs = self.net(inputs, rate=self.rate, S_rate = self.S_rate, dropout_layer = self.dropout_layer)
                # print(self.is_celeba)
                cls_loss, inter_loss = self.criterion(outputs, labels)
                #print(cls_loss.item(), inter_loss.item())
                if(epoch <= 4):
                    loss = cls_loss
                    inter_loss.backward()
                else:
                    # interloss_avg = interloss_avg + inter_loss
                    # if(i % 5 == 4):
                    loss = cls_loss + inter_loss
                    # else:
                    #     loss = cls_loss
                #print(cls_loss.item())
                self.optimizer.zero_grad()
                predicted_train = (torch.argmax(outputs[0], dim=1)).detach().cpu().numpy()
                total_train += labels.size(0)
                correct_train += (predicted_train == labels.data.detach().cpu().numpy()).sum()
                # if(torch.isnan(loss)):
                #     print('loss is nan')
                if(not torch.isnan(loss)):
                    # with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                    # if i % 5 == 4:
                    self.optimizer.step()
                    # interloss_avg = 0
                    #grad_epoch.append(self.grad_statistics())
                losses_epoch.append([self.criterion.targetlossvalue.detach().cpu(), self.criterion.interlossvalue.detach().cpu()])
                #print(self.criterion.targetlossvalue.detach().cpu().item(), self.criterion.interlossvalue.detach().cpu().item())
                #print(outputs[0])
                #losses_epoch.append([loss.detach().cpu()])
            #print(total_train)
            self.result_train = 100 * correct_train / total_train

            if((not self.pretrained) and (count <= (self.epochs - 1))):
                for param_group in self.optimizer.param_groups:
                    #print(param_group['lr'])
                    param_group['lr'] = lr_epoch[count]
                    #print(param_group['lr'])
                    #print(lr_epoch[count])
                    count += 1


            loss_test = []
            criterion_test = nn.CrossEntropyLoss()
            if(epoch % 10 == 9 or epoch < 9):
                self.net.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i, data in enumerate(tqdm(self.testloader)):
                        if('corrupt' in self.set_name):
                            images, labels, image_path = data
                        else:
                            images, labels = data
                        #images, labels = data
                        outputs = self.net(Variable(images).cuda(self.gpu), rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                        targetloss_test = criterion_test(outputs, labels.cuda(self.gpu))
                        loss_test.append(targetloss_test.detach().cpu())
                        predicted = outputs.argmax(dim=1).data.detach().cpu().numpy()
                        total += labels.size(0)
                        correct += (predicted == labels.data.numpy()).sum()
                    #print(total)
                    self.result = 100 * correct / total
                self.net.train()

                temp = np.mean(losses_epoch, axis=0)
                tmp_test = np.mean(loss_test)
                # epoch, target loss, inter loss, result, target loss on testset, testloss-trainloss, grad
                epoch_statics = [epoch, temp[0], temp[1], self.result_train, self.result, tmp_test, (tmp_test - temp[0])] #, np.mean(grad_epoch)]
                grad_epoch = []
                losses_check.append(epoch_statics)
                if(not os.path.exists(f'information/{self.sub_folder}')):
                    os.makedirs(f'information/{self.sub_folder}')
                self.draw_curves(np.array(losses_check))
                np.save(f'information/{self.sub_folder}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_%04d_{self.seed}.npy' % (epoch+1), np.array(losses_check))

            if (epoch in [2, 5, 8, 15, 25, 50, 99, 100, 160, 199, 200, 299, 300, 499]) or (epoch % 10 == 9):
                self.checkpoint(epoch+1)

        # print(time.asctime(time.localtime(time.time())))

    def draw_curves(self, loss_check):
        # plt.figure()
        train_loss = loss_check[:, 1]
        test_loss = loss_check[:, 5]
        train_acc = loss_check[:, 3]
        test_acc = loss_check[:, 4]

        plt.figure(figsize=(5, 10))
        plt.subplot(211)
        plt.plot(loss_check[:, 0], train_loss, color="b")
        plt.plot(loss_check[:, 0], test_loss, color="r")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.subplot(212)
        plt.plot(loss_check[:, 0], train_acc, color="b")
        plt.plot(loss_check[:, 0], test_acc, color="r")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.savefig(f"./information/{self.sub_folder}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_{self.seed}.png")
        plt.close()


    def checkpoint(self, epochs):
        if(not os.path.exists(f'results/{self.sub_folder}')):
            os.makedirs(f'results/{self.sub_folder}')
        torch.save(self.net.state_dict(), f'results/{self.sub_folder}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_%04d_{self.result_train}_{self.result}_{self.seed}.pth' % epochs)

    def compute_error_diff(self):
        #self.net.load_state_dict(torch.load(weight))
        epochs = [2, 5, 8, 15, 25, 50, 99, 100, 160, 199]
        # epochs = [2, 5, 8, 15, 25, 50, 100, 160, 200, 300, 400, 430, 460, 499]
        error_diff = []
        loss_diff  = []
        criterion_test = nn.CrossEntropyLoss()
        for epoch in epochs:
            for f in sorted(os.listdir(f'./results/{self.sub_folder}/')):
                f_list = f.split('_')
                if((str(self.rate)==f_list[0]) and (str(self.p_info)==f_list[1]) and (str(self.fixed_len)==f_list[2]) and (str(self.sample_set)==f_list[3]) and (str(self.S_rate)==f_list[4]) and (str(self.dropout_layer)==f_list[5]) and (str(f_list[6])=='%04d'%(epoch+1)) and (str(self.seed)==(f_list[-1].split('.'))[0])):
                    file_name = f
                    #print(file_name)
                    break
            self.net.load_state_dict(torch.load(os.path.join(f'./results/{self.sub_folder}/', file_name), map_location=torch.device('cpu')))

            correct_train  = 0
            total_train  = 0
            loss_train = []
            with torch.no_grad():
                self.net.train()
                for i, data in enumerate(tqdm(self.trainloader)):
                    images, labels = data
                    images, labels = Variable(images).cuda(self.gpu), Variable(labels).cuda(self.gpu)
                    outputs = self.net(images, rate=self.rate, dropout_layer=self.dropout_layer)
                    targetloss_train = criterion_test(outputs[0], labels)
                    loss_train.append(targetloss_train.detach().cpu())
                    predicted_train = outputs[0].argmax(dim=1).detach().cpu().numpy()
                    total_train += labels.size(0)
                    correct_train += (predicted_train == labels.detach().cpu().numpy()).sum()
                result_train = 100 * correct_train / total_train
                loss_epoch_train = np.mean(loss_train)

                correct  = 0
                total  = 0
                loss_test = []
                self.net.eval()
                for i, data in enumerate(tqdm(self.testloader)):
                    images, labels = data
                    images, labels = Variable(images).cuda(self.gpu), Variable(labels).cuda(self.gpu)
                    outputs = self.net(images, rate=self.rate, dropout_layer=self.dropout_layer)
                    targetloss_test = criterion_test(outputs, labels)
                    loss_test.append(targetloss_test.detach().cpu())
                    predicted = outputs.argmax(dim=1).detach().cpu().numpy()
                    total += labels.size(0)
                    correct += (predicted == labels.detach().cpu().numpy()).sum()
                result = 100 * correct / total
                loss_epoch_test = np.mean(loss_test)

                #error_diff.append([epoch, result, result_train, np.abs(result - result_train)])
                loss_diff.append([epoch, loss_epoch_train, loss_epoch_test, result_train, result, (loss_epoch_test - loss_epoch_train)])
                #print(np.abs(result-result_train))
                #print(np.abs(loss_epoch_test - loss_epoch_train))
        if(not os.path.exists(f'./error_diff/{self.sub_folder}')):
            os.makedirs(f'./error_diff/{self.sub_folder}')
        #np.save(f'./error_diff/{self.net_name}{self.set_name}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_errorDiff_.npy', np.array(error_diff))
        np.save(f'./error_diff/{self.sub_folder}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_{self.seed}.npy', np.array(loss_diff))

    def interval_interaction(self, single=True, mode='shapley'):
        epochs = [5, 8, 15, 25, 50, 100, 160, 200, 299]
        interaction_epoch = []
        for epoch in epochs:
            for f in sorted(os.listdir(f'./results/{self.sub_folder}/')):
                f_list = f.split('_')
                if((str(self.rate)==f_list[0]) and (str(self.p_info)==f_list[1]) and (str(self.fixed_len)==f_list[2]) and (str(self.sample_set)==f_list[3]) and (str(self.S_rate)==f_list[4]) and (str(self.dropout_layer)==f_list[5]) and (str(f_list[6])=='%04d'%(epoch+1)) and (str(self.seed)==(f_list[-1].split('.'))[0])):
                    file_name = f
                    #print(file_name)
                    break
            self.net.load_state_dict(torch.load(os.path.join(f'./results/{self.sub_folder}/', file_name), map_location=torch.device('cpu')))
            interval = np.array([0.0, 0.2])
            interaction_interval = [epoch]
            for count in range(5):
                interaction = self.get_interaction(sample_rate=interval)
                interaction_interval.append(interaction)
                interval += 0.2
            interaction_epoch.append(interaction_interval)
            #print(interaction_interval)
            if(not os.path.exists(f'./interaction_interval/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}')):
                os.makedirs(f'./interaction_interval/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}')
            np.save(f'./interaction_interval/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_{self.seed}.npy', np.array(interaction_epoch))

    def compute_interaction(self, loss_interaction=False, mode='shapley', subseed=0):
        #epochs = [2, 5, 8, 15, 25, 50, 100, 160, 200, 240, 300, 500]
        epochs = [2, 5, 8, 15, 25, 50, 100, 160, 200, 299]
        # epochs = [299]
        interaction_epoch = []
        for epoch in epochs:
            for f in sorted(os.listdir(f'./results/{self.sub_folder}/')):
                f_list = f.split('_')
                if((str(self.rate)==f_list[0]) and (str(self.p_info)==f_list[1]) and (str(self.fixed_len)==f_list[2]) and (str(self.sample_set)==f_list[3]) and (str(self.S_rate)==f_list[4]) and (str(self.dropout_layer)==f_list[5]) and (str(f_list[6])=='%04d'%(epoch+1)) and (str(self.seed)==(f_list[-1].split('.'))[0])):
                    file_name = f
                    #print(file_name)
                    break
            self.net.load_state_dict(torch.load(os.path.join(f'./results/{self.sub_folder}/', file_name), map_location=torch.device('cpu')))

            # interaction_value = self.get_input_interaction_grid_instability(mode=mode, subseed=subseed)   # use this function when compute instability
            # interaction_value, interaction_value_norm = self.get_corrupt_input_interaction_grid_tinyimagenet(mode=mode)   # use this function when compute interaction of corrupt tinyimagenet
            # interaction_value, interaction_value_norm = self.get_corrupt_input_interaction_grid_tinyimagenet(mode=mode, corrupt=True)
            interaction_value, interaction_value_norm, dataset_norm = self.get_interaction(mode=mode)   # use this function when compute interaction between features before the dropout/interaction loss layer

            interaction_epoch.append([epoch, interaction_value, interaction_value_norm, dataset_norm])
            # interaction_epoch.append([epoch, interaction_value, interaction_value_norm])
            #print(f'no corrupt: {interaction_epoch}')

            if not softmax:
                if(not os.path.exists(f'./interaction/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}')):
                    os.makedirs(f'./interaction/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}')
                np.save(f'./interaction/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_{mode}_{self.seed_interaction}_dataset_norm.npy', np.array(interaction_epoch))
            else:
                if(not os.path.exists(f'./softmax_interaction/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}')):
                    os.makedirs(f'./softmax_interaction/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}')
                np.save(f'./softmax_interaction/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_posPair_{self.pos_pair}_sampleNumber_{self.sample_number}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_{mode}_{self.seed_interaction}_dataset_norm.npy', np.array(interaction_epoch))


    def compute_dataset_normalize(self):
        #epochs = [2, 5, 8, 15, 25, 50, 100, 160, 200, 240, 300, 500]
        epochs = [2, 5, 8, 15, 25, 50, 99, 100, 160, 199]
        normalize_epoch = []
        for epoch in epochs:
            for f in sorted(os.listdir(f'./results/{self.sub_folder}/')):
                f_list = f.split('_')
                if((str(self.rate)==f_list[0]) and (str(self.p_info)==f_list[1]) and (str(self.fixed_len)==f_list[2]) and (str(self.sample_set)==f_list[3]) and (str(self.S_rate)==f_list[4]) and (str(self.dropout_layer)==f_list[5]) and (str(f_list[6])=='%04d'%(epoch+1)) and (str(self.seed)==(f_list[-1].split('.'))[0])):
                    file_name = f
                    #print(file_name)
                    break
            self.net.load_state_dict(torch.load(os.path.join(f'./results/{self.sub_folder}/', file_name), map_location=torch.device('cpu')))
            # interaction_value, interaction_value_norm = self.get_input_interaction_grid(loss_interaction=loss_interaction, mode=mode)
            with torch.no_grad():
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                self.net.eval()
                dataset_normalized = 0
                for i, (img, lbl) in enumerate(tqdm(self.valloader)):
                    img = img.cuda(self.gpu)
                    output = self.net.features(img)
                    # print(output.shape)
                    y_lbl = output[:, lbl].clone()
                    y_avg = (torch.sum(output, 1) - output[:, lbl]) / (output.size(1) - 1)
                    img_normalized = torch.abs(y_lbl - y_avg)
                    dataset_normalized += img_normalized.item()

                dataset_normalized = dataset_normalized / len(self.valloader)
            normalize_epoch.append([epoch, dataset_normalized])
            if (not os.path.exists(
                    f'./normalize_dataset/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}')):
                os.makedirs(
                    f'./normalize_dataset/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}')
            np.save(
                f'./normalize_dataset/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}/{self.rate}_{self.p_info}_{self.fixed_len}_{self.sample_set}_{self.S_rate}_{self.dropout_layer}_{self.seed}.npy',
                np.array(normalize_epoch))


    def get_corrupt_input_interaction_grid_tinyimagenet(self, size_grid=4, mode='shapley', sample=500, img_num=95, corrupt=False, sample_rate=None, pos_array_sample=None):
        with torch.no_grad():
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            import time
            #print(time.asctime(time.localtime(time.time())))
            self.net.eval()
            corrupt_imgs = np.load(f'tinyimagenet_corrupt_25_2.npy', allow_pickle=True).item()

            # use dataset to normalize
            '''
            dataset_normalized = 0
            count_total = 0
            for i, data in enumerate(self.valloader):
                img, lbl, img_path = data
                #if(i < 5):
                #    print(img_path[0])
                if((not corrupt) and (img_path[0] in corrupt_imgs)):
                    continue
                elif(corrupt and (not (img_path[0] in corrupt_imgs))):
                    continue

                count_total = count_total + 1
                img = img.cuda(self.gpu)
                output = self.net(img, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                # print(output.shape)
                y_lbl = output[:, lbl].clone()
                # output[:, lbl] = 0
                y_avg = (torch.sum(output, 1) - output[:, lbl]) / (output.size(1) - 1)
                img_normalized = torch.abs(y_lbl - y_avg)
                dataset_normalized += img_normalized.item()

            if(corrupt):
                len_dataset = len(corrupt_imgs)
            else:
                len_dataset = len(self.trainset) - len(corrupt_imgs)
            dataset_normalized = dataset_normalized / len_dataset
            print(f'corrupt: {corrupt}, {count_total}, {dataset_normalized}')
            '''
            #return 0,0,0
            #corrupt_imgs = np.load(f'tinyimagenet_corrupt_25_2.npy', allow_pickle=True).item()
            if(not corrupt):
                img_count = 95
            else:
                img_count = 5

            counter = 0
            interaction_avg = 0
            interaction_avg_norm = 0
            criterion_interaction = nn.CrossEntropyLoss()
            selection = [[-2,-2], [-2,-1], [-2,0], [-2,1], [-2,2], [-1,-2], [-1,-1], [-1,0], [-1,1], [-1,2], [0,-2], [0,-1], [0,1], [0,2], [1,-2], [1,-1], [1,0], [1,1], [1,2], [2,-2], [2,-1], [2,0], [2,1], [2,2]]
            for i, data in enumerate(self.valloader):
                img, lbl, img_path = data
                #print(i)
                if((not corrupt) and (img_path[0] in corrupt_imgs)):
                    continue
                elif(corrupt and (not (img_path[0] in corrupt_imgs))):
                    continue

                #print(lbl, img_path, img_path[0])
                interaction_single_avg = 0
                counter = counter + 1
                print(f'img_count: {img_count}')
                print(f'counter: {counter}')
                #interaction_single_avg_false = 0
                img = img.cuda(self.gpu)
                lbl = lbl.cuda(self.gpu)
                img_shape = img.shape
                #print(img_shape)
                size_map = img_shape[2] // size_grid

                # use single image to normalize
                y_complete = self.net(img, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                y_lbl = y_complete[:, lbl].clone()
                y_avg = (torch.sum(y_complete, 1) - y_complete[:, lbl]) / (y_complete.size(1) - 1)
                img_normalized = (torch.abs(y_lbl - y_avg)).item()

                #img = torch.flatten(img, 2)
                #print(img.shape)
                # pos_info = f'{self.net_name}{self.set_name}_{self.fraction}_' + str(img.shape[2]) + '.npy'
                # sel = []
                for row in [0, 5, 10, 15]:
                    for col in [0, 5, 10, 15]:
                        sel = []
                        #print(row, col)
                        for pos in selection:
                            if (row+pos[0]>=0) and (row+pos[0]<=(size_map-1)) and (col+pos[1]>=0) and (col+pos[1]<=(size_map-1)):
                                sel.append(pos)
                        #print(sel)
                        sample_pos = random.sample(sel, 5)
                        for j in range(len(sample_pos)):
                            # print(sample_pos[j], row, col)
                            t1 = time.time()

                            if (sample_rate is None):
                                if mode == 'shapley':
                                        #r = random.random()
                                        r = torch.rand(sample, 1, 1, 1).to(img.device)
                                else:
                                    r = 0.5
                            else:
                                #r = random.uniform(sample_rate[0], sample_rate[1])
                                r = (sample_rate[1] - sample_rate[0]) * torch.rand(sample, 1, 1, 1).to(img.device) + sample_rate[0]
                            mask = torch.rand(sample, 1, size_map, size_map).to(img.device)
                            mask = (mask - r).sign()
                            mask = (mask + 1) / 2
                            mask[:, :, row, col] = 0
                            mask[:, :, row + sample_pos[j][0], col + sample_pos[j][1]] = 0
                            mask_ = deepcopy(mask)
                            mask[:, :, row, col] = 1
                            mask_i = deepcopy(mask)
                            mask[:, :, row + sample_pos[j][0], col + sample_pos[j][1]] = 1
                            mask_ij = deepcopy(mask)
                            mask[:, :, row, col] = 0
                            mask_j = deepcopy(mask)

                            mask_ = torch.nn.functional.interpolate(mask_, (img.size(2), img.size(3)), mode="nearest")
                            mask_i = torch.nn.functional.interpolate(mask_i, (img.size(2), img.size(3)), mode="nearest")
                            mask_j = torch.nn.functional.interpolate(mask_j, (img.size(2), img.size(3)), mode="nearest")
                            mask_ij = torch.nn.functional.interpolate(mask_ij, (img.size(2), img.size(3)), mode="nearest")

                            img = img.expand(sample, -1, -1, -1)
                            maskImg0 = img.detach() * mask_
                            maskImg_i0 = img.detach() * mask_i
                            maskImg_j0 = img.detach() * mask_j
                            maskImg_ij0 = img.detach() * mask_ij

                            #print(count)
                            #print(maskImg0.shape)
                            t2 = time.time()
                            y0    = self.net(maskImg0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                            y_i0  = self.net(maskImg_i0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                            y_j0  = self.net(maskImg_j0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                            y_ij0 = self.net(maskImg_ij0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                            t3 = time.time()

                            if self.softmax:
                                y0, y_i0, y_j0, y_ij0 = nn.functional.softmax(y0, 1), nn.functional.softmax(y_i0, 1), nn.functional.softmax(y_j0, 1), nn.functional.softmax(y_ij0, 1)
                                y0, y_i0, y_j0, y_ij0 = torch.log(y0), torch.log(y_i0), torch.log(y_j0), torch.log(y_ij0)

                            interaction = torch.abs(torch.mean(y_ij0[:, lbl] + y0[:, lbl] - y_i0[:, lbl] - y_j0[:, lbl]))
                            interaction_single_avg += interaction.item()
                            # print(t2 - t1, t3 - t2)
                        # interaction = (interaction0 + interaction1 + interaction2 + interaction3 + interaction4) / 5

                        # interaction_single_avg += interaction.item()

                interaction_single_avg = interaction_single_avg / (4 * 4 * 5)
                interaction_single_avg_norm = interaction_single_avg / img_normalized
                interaction_avg += interaction_single_avg
                interaction_avg_norm += interaction_single_avg_norm
                #print(interaction_single_avg)
                #print(time.asctime(time.localtime(time.time())))
                if counter >= img_count:
                    break
            interaction_avg = interaction_avg / img_count
            #interaction_avg_norm = interaction_avg / dataset_normalized
            interaction_avg_norm = interaction_avg_norm / img_count
            self.net.train()
            return interaction_avg, interaction_avg_norm


    def get_input_interaction_grid_instability(self, size_grid=2, mode='shapley', sample=5000, img_num=10, sample_rate=None, pos_array_sample=None, subseed=0):
        with torch.no_grad():
            set_seed_pytorch(self.seed)
            import time
            #print(time.asctime(time.localtime(time.time())))
            self.net.eval()
            interaction_avg = 0
            #interaction_avg_false = 0
            criterion_interaction = nn.CrossEntropyLoss()
            selection = [[-2,-2], [-2,-1], [-2,0], [-2,1], [-2,2], [-1,-2], [-1,-1], [-1,0], [-1,1], [-1,2], [0,-2], [0,-1], [0,1], [0,2], [1,-2], [1,-1], [1,0], [1,1], [1,2], [2,-2], [2,-1], [2,0], [2,1], [2,2]]
            for i, (img, lbl) in enumerate(self.valloader):
                print(i)
                interaction_single_avg = 0
                all_interactions = []
                #interaction_single_avg_false = 0
                img = img.cuda(self.gpu)
                lbl = lbl.cuda(self.gpu)
                img_shape = img.shape
                #print(img_shape)
                size_map = img_shape[2] // size_grid
                pos_info = f'input_{self.net_name}{self.set_name}_{self.fraction}_' + str(img.shape[2]) + '.npy'
                if (self.pos_info is not None):
                    if (not os.path.exists(pos_info)):
                        pos_array = []
                        for row in [0, 5, 10, 15]:  # range(size_map):
                            for col in [0, 5, 10, 15]:  # range(size_map):
                                sel = []
                                for pos in selection:
                                    if (row + pos[0] >= 0) and (row + pos[0] <= (size_map - 1)) and (
                                            col + pos[1] >= 0) and (col + pos[1] <= (size_map - 1)):
                                        sel.append(pos)
                                sample_pos = random.sample(sel, 5)
                                for pos in sample_pos:
                                    pos_array.append([row, col, row + pos[0], col + pos[1]])
                        pos_array = np.array(pos_array)
                        pos_array = np.unique(pos_array, axis=0)
                        np.save(pos_info, pos_array)
                    else:
                        pos_array = np.load(pos_info)
                for pos in pos_array:
                    print(subseed, pos)
                    set_seed_pytorch(subseed + pos.sum() + i)
                    if (sample_rate is None):
                        if mode == 'shapley':
                                #r = random.random()
                                r = torch.rand(sample, 1, 1, 1).to(img.device)
                        else:
                            r = 0.5
                    else:
                        r = (sample_rate[1] - sample_rate[0]) * torch.rand(sample, 1, 1, 1).to(img.device) + sample_rate[0]
                    set_seed_pytorch(subseed + pos.sum())
                    mask = torch.rand(sample, 1, size_map, size_map).to(img.device)
                    mask = (mask - r).sign()
                    mask = (mask + 1) / 2
                    mask[:, :, pos[0], pos[1]] = 0
                    mask[:, :, pos[2], pos[3]] = 0
                    mask_ = deepcopy(mask)
                    mask[:, :, pos[0], pos[1]] = 1
                    mask_i = deepcopy(mask)
                    mask[:, :, pos[2], pos[3]] = 1
                    mask_ij = deepcopy(mask)
                    mask[:, :, pos[0], pos[1]] = 0
                    mask_j = deepcopy(mask)

                    mask_ = torch.nn.functional.interpolate(mask_, (img.size(2), img.size(3)), mode="nearest")
                    mask_i = torch.nn.functional.interpolate(mask_i, (img.size(2), img.size(3)), mode="nearest")
                    mask_j = torch.nn.functional.interpolate(mask_j, (img.size(2), img.size(3)), mode="nearest")
                    mask_ij = torch.nn.functional.interpolate(mask_ij, (img.size(2), img.size(3)), mode="nearest")

                    img = img.expand(sample, -1, -1, -1)
                    maskImg0 = img.detach() * mask_
                    maskImg_i0 = img.detach() * mask_i
                    maskImg_j0 = img.detach() * mask_j
                    maskImg_ij0 = img.detach() * mask_ij

                    y0    = self.net(maskImg0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                    y_i0  = self.net(maskImg_i0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                    y_j0  = self.net(maskImg_j0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                    y_ij0 = self.net(maskImg_ij0, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)

                    all_interaction = (y_ij0[:, lbl] + y0[:, lbl] - y_i0[:, lbl] - y_j0[:, lbl]).view(-1).cpu().numpy()
                    all_interactions.append(all_interaction)
                    interaction = torch.abs(torch.mean(y_ij0[:, lbl] + y0[:, lbl] - y_i0[:, lbl] - y_j0[:, lbl]))
                    interaction_single_avg += interaction.item()
                    # print(t2 - t1, t3 - t2)
                        # interaction = (interaction0 + interaction1 + interaction2 + interaction3 + interaction4) / 5

                        # interaction_single_avg += interaction.item()

                interaction_single_avg = interaction_single_avg / (4 * 4 * 5)
                interaction_avg += interaction_single_avg
                if(not os.path.exists(f"input_interaction_dropout_instabiliy/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_grid/")):
                    os.makedirs(f"input_interaction_dropout_instabiliy/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_grid/")
                np.save(f"input_interaction_dropout_instability/{self.net_name}{self.set_name}_{self.val_mode}_{self.fraction}_grid/img_{i}_subseed_{subseed}_instability.npy", np.array(all_interactions))
                if i >= img_num - 1:
                    break
            interaction_avg = interaction_avg / img_num
        self.net.train()
        return interaction_avg


    def get_input_interaction(self, mode='shapley', sample=500, img_num=10, sample_rate=None, pos_array_sample=None):
        with torch.no_grad():
            #import time
            #print(time.asctime(time.localtime(time.time())))
            random.seed(self.seed_interaction)
            np.random.seed(self.seed_interaction)
            torch.manual_seed(self.seed_interaction)
            torch.cuda.manual_seed_all(self.seed_interaction)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            self.net.eval()

            dataset_normalized = 0
            selection = [[-2,-2], [-2,-1], [-2,0], [-2,1], [-2,2], [-1,-2], [-1,-1], [-1,0], [-1,1], [-1,2], [0,-2], [0,-1], [0,1], [0,2], [1,-2], [1,-1], [1,0], [1,1], [1,2], [2,-2], [2,-1], [2,0], [2,1], [2,2]]
            for i, (img, lbl) in enumerate(self.valloader):
                img = img.cuda(self.gpu)
                lbl = lbl.cuda(self.gpu)
                output = self.net(img, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)

                # ****** use loss to compute normalization ******
                # img_none = img * 0
                # output_none = self.net(img_none, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                # loss_img = criterion_interaction(output, lbl)
                # loss_none = criterion_interaction(output_none, lbl)

                #print(output.shape)
                y_lbl = output[:, lbl].clone()
                #output[:, lbl] = 0
                y_avg = (torch.sum(output, 1) - output[:, lbl]) / (output.size(1) - 1)
                img_normalized = torch.abs(y_lbl - y_avg)
                # img_normalized = torch.abs(loss_img - loss_none)
                dataset_normalized += img_normalized.item()

            dataset_normalized = dataset_normalized / len(self.valloader)

            interaction_avg = 0
            interaction_avg_false = 0
            criterion_interaction = nn.CrossEntropyLoss()
            for i, (img, lbl) in enumerate(self.valloader):
                #print(i)
                interaction_single_avg = 0
                interaction_single_avg_false = 0
                img = img.cuda(self.gpu)
                lbl = lbl.cuda(self.gpu)
                img_shape = img.shape
                img = torch.flatten(img, 2)
                #print(img.shape)
                pos_info = f'{self.net_name}{self.set_name}_{self.pos_pair}_' + str(img.shape[2]) + '.npy'
                if(self.pos_info is not None):
                    if(not os.path.exists(pos_info)):
                        # sample_count = 0
                        pos_array = []
                        pos_choice = np.random.choice(img.shape[2], self.pos_pair, replace=False)
                        for pos in pos_choice:
                            sel = []
                            row = pos // img_shape[2]
                            col = pos % img_shape[2]
                            for shift in selection:
                                if (row+shift[0]>=0) and (row+shift[0]<=(img_shape[2]-1)) and (col+shift[1]>=0) and (col+shift[1]<=(img_shape[2]-1)):
                                    sel.append(shift)
                            #print(sel)
                            sample_shift = random.sample(sel, 1)
                            pos1 = (row + sample_shift[0][0]) * img_shape[2] + (col + sample_shift[0][1])
                            pos_array.append([pos, pos1])

                        # while(1):
                        #     pos1, pos2 = random.sample(range(img.shape[2]), 2)
                        #     if((np.abs(pos1//img_shape[2]-pos2//img_shape[2]) > 2) or (np.abs(pos1%img_shape[2]-pos2%img_shape[2]) > 2)):
                        #         continue
                        #     pos_array.append([pos1, pos2])
                        #     sample_count += 1
                        #     if(sample_count == 10):
                        #         break

                        # for _ in range(self.pos_pair):
                        #     pos_array.append(random.sample(range(img.shape[2]), 2))
                        pos_array = np.array(pos_array)
                        pos_array = np.unique(pos_array, axis=0)
                        np.save(pos_info, pos_array)
                    else:
                        pos_array = np.load(pos_info)
                else:
                    #pos_array = []
                    #for _ in range(self.pos_pair):
                    #    pos_array.append(random.sample(range(img.shape[2]), 2))
                    #pos_array = np.array(pos_array)
                    #pos_array = np.unique(pos_array, axis=0)
                    pos_array = pos_array_sample
                for (p1, p2) in pos_array:
                    maskImg_i  = []
                    maskImg_j  = []
                    maskImg_ij = []
                    maskImg    = []
                    for count in range(self.sample_number):
                        if(sample_rate is None):
                            if mode == 'shapley':
                                r = random.random()
                            else:
                                r = 0.5
                        else:
                            r = random.uniform(sample_rate[0], sample_rate[1])

                        mask = torch.rand_like(img[:, 0, :]).to(img.device)
                        mask = (mask - r).sign()
                        mask = (mask + 1) / 2
                        mask[:, p1] = 0
                        mask[:, p2] = 0
                        maskImg.append((mask * img.detach()).reshape(img_shape))
                        mask[:, p1] = 1
                        maskImg_i.append((mask * img.detach()).reshape(img_shape))
                        mask[:, p2] = 1
                        maskImg_ij.append((mask * img.detach()).reshape(img_shape))
                        mask[:, p1] = 0
                        maskImg_j.append((mask * img.detach()).reshape(img_shape))
                    maskImg    = torch.cat((maskImg[:]),    0).cuda(self.gpu)
                    maskImg_i  = torch.cat((maskImg_i[:]),  0).cuda(self.gpu)
                    maskImg_j  = torch.cat((maskImg_j[:]),  0).cuda(self.gpu)
                    maskImg_ij = torch.cat((maskImg_ij[:]), 0).cuda(self.gpu)
                    #print(maskImg.shape)
                    y    = self.net(maskImg, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                    y_i  = self.net(maskImg_i, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                    y_j  = self.net(maskImg_j, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                    y_ij = self.net(maskImg_ij, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
                    if(loss_interaction):
                        lbl_ = lbl.expand(y.shape[0])
                        interaction = torch.abs(torch.mean(criterion_interaction(y_ij, lbl_) + criterion_interaction(y, lbl_) - criterion_interaction(y_i, lbl_) - criterion_interaction(y_j, lbl_)))
                    else:
                        if self.softmax:
                            y, y_i, y_j, y_ij = nn.functional.softmax(y, 1), nn.functional.softmax(y_i, 1), nn.functional.softmax(y_j, 1), nn.functional.softmax(y_ij, 1)
                            y, y_i, y_j, y_ij = torch.log(y), torch.log(y_i), torch.log(y_j), torch.log(y_ij)

                        interaction = torch.abs(torch.mean(y_ij[:, lbl] + y[:, lbl] - y_i[:, lbl] - y_j[:, lbl]))
                    #interaction_false = torch.mean(torch.abs(y_ij[:, lbl] + y[:, lbl] - y_i[:, lbl] - y_j[:, lbl]))
                    #interaction_false = torch.mean(y_ij[:, lbl] + y[:, lbl] - y_i[:, lbl] - y_j[:, lbl])
                    interaction_single_avg += interaction.item()
                    #interaction_single_avg_false += interaction_false.item()
                #print(maskImg.shape)
                interaction_single_avg = interaction_single_avg / self.pos_pair
                #interaction_single_avg_false = interaction_single_avg_false / self.pos_pair
                interaction_avg += interaction_single_avg
                #interaction_avg_false += interaction_single_avg_false
                #print(time.asctime(time.localtime(time.time())))
                if i >= img_num - 1:
                    break
            interaction_avg = interaction_avg / img_num
            interaction_avg_norm = interaction_avg / dataset_normalized
            #interaction_avg_false = interaction_avg_false / img_num
            self.net.train()
            #print(interaction_avg)
            return interaction_avg, interaction_avg_norm, dataset_normalized

    def get_interaction(self, loss_interaction=True, mode='shapley', img_num=10, sample_rate=None):
        with torch.no_grad():
            random.seed(self.seed_interaction)
            np.random.seed(self.seed_interaction)
            torch.manual_seed(self.seed_interaction)
            torch.cuda.manual_seed_all(self.seed_interaction)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            self.net.eval()
            interaction_epoch = []
            criterion_interaction = nn.CrossEntropyLoss()
            dataset_normalized = 0
            for i, (img, lbl) in enumerate(self.valloader):
               img = img.cuda(self.gpu)
               lbl = lbl.cuda(self.gpu)
               output = self.net(img, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)

               # ****** use loss to compute normalization ******
               # img_none = img * 0
               # output_none = self.net(img_none, rate=self.rate, S_rate=self.S_rate, dropout_layer=self.dropout_layer)
               # loss_img = criterion_interaction(output, lbl)
               # loss_none = criterion_interaction(output_none, lbl)

               #print(output.shape)
               y_lbl = output[:, lbl].clone()
               #output[:, lbl] = 0
               y_avg = (torch.sum(output, 1) - output[:, lbl]) / (output.size(1) - 1)
               img_normalized = torch.abs(y_lbl - y_avg)
               # img_normalized = torch.abs(loss_img - loss_none)
               dataset_normalized += img_normalized.item()

            dataset_normalized = dataset_normalized / len(self.valloader)

            interaction_avg = 0
            interaction_avg_norm = 0
            for i, (img, lbl) in enumerate(self.valloader):
                #print(i)
                interaction_single_avg = 0
                interaction_single_avg_norm = 0
                img = img.cuda(self.gpu)
                #print(img.shape)
                #feature = torch.flatten(self.net.features(img), 1)
                feature = self.net.features[:self.dropout_layer](img)
                feature = self.net.dropout(feature, S_rate=self.S_rate)
                #print(feature.shape)
                feature_shape = feature.shape
                feature = torch.flatten(feature, 1)
                #feature_shape = feature.shape
                #feature = torch.flatten(feature, 1)
                pos_info = f'{self.net_name}{self.set_name}_{self.pos_pair}_' + str(feature.shape[1]) + '.npy'
                if(self.pos_info is not None):
                    if(not os.path.exists(pos_info)):
                        pos_array = []
                        for _ in range(self.pos_pair):
                            pos_array.append(random.sample(range(feature.shape[1]), 2))
                        pos_array = np.array(pos_array)
                        pos_array = np.unique(pos_array, axis=0)
                        np.save(pos_info, pos_array)
                    else:
                        pos_array = np.load(pos_info)
                else:
                    pos_array = []
                    for _ in range(self.pos_pair):
                        pos_array.append(random.sample(range(feature.shape[1]), 2))
                    pos_array = np.array(pos_array)
                    pos_array = np.unique(pos_array, axis=0)

                #y_complete = self.net.features[self.dropout_layer:](feature.detach().reshape(feature_shape))
                #y_none = self.net.features[self.dropout_layer:]((0 * feature.detach()).reshape(feature_shape))
                for (p1, p2) in pos_array:
                    maskImg_i  = []
                    maskImg_j  = []
                    maskImg_ij = []
                    maskImg    = []
                    for count in range(self.sample_number):
                        if(sample_rate is None):
                            if mode == 'shapley':
                                r = random.random()
                            else:
                                r = 0.5
                        else:
                            # r = random.uniform(sample_rate[0], sample_rate[1])
                            r = 1 - sample_rate[0] - (sample_rate[1] - sample_rate[0]) * torch.rand(1)[0]

                        mask = torch.rand_like(feature).to(img.device)
                        mask = (mask - r).sign()
                        mask = (mask + 1) / 2
                        mask[:, p1] = 0
                        mask[:, p2] = 0
                        maskImg.append((mask * feature.detach()).reshape(feature_shape))
                        mask[:, p1] = 1
                        maskImg_i.append((mask * feature.detach()).reshape(feature_shape))
                        mask[:, p2] = 1
                        maskImg_ij.append((mask * feature.detach()).reshape(feature_shape))
                        mask[:, p1] = 0
                        maskImg_j.append((mask * feature.detach()).reshape(feature_shape))
                    maskImg    = torch.cat((maskImg[:]),    0).cuda(self.gpu)
                    maskImg_i  = torch.cat((maskImg_i[:]),  0).cuda(self.gpu)
                    maskImg_j  = torch.cat((maskImg_j[:]),  0).cuda(self.gpu)
                    maskImg_ij = torch.cat((maskImg_ij[:]), 0).cuda(self.gpu)
                    #print(maskImg.shape)
                    y    = self.net.features[self.dropout_layer:](maskImg)
                    y_i  = self.net.features[self.dropout_layer:](maskImg_i)
                    y_j  = self.net.features[self.dropout_layer:](maskImg_j)
                    y_ij = self.net.features[self.dropout_layer:](maskImg_ij)
                    if(loss_interaction):
                        lbl_ = (lbl.expand(y.shape[0])).cuda(self.gpu)
                        interaction = torch.abs(torch.mean(criterion_interaction(y_ij, lbl_) + criterion_interaction(y, lbl_) - criterion_interaction(y_i, lbl_) - criterion_interaction(y_j, lbl_)))
                    else:
                        if self.softmax:
                            y, y_i, y_j, y_ij = F.softmax(y, 1), F.softmax(y_i, 1), F.softmax(y_j, 1), F.softmax(y_ij, 1)
                            y, y_i, y_j, y_ij = torch.log(y), torch.log(y_i), torch.log(y_j), torch.log(y_ij)
                        interaction = torch.abs(torch.mean(y_ij[:, lbl] + y[:, lbl] - y_i[:, lbl] - y_j[:, lbl]))

                    interaction_single_avg += interaction.item()

                interaction_single_avg = interaction_single_avg / self.pos_pair
                # interaction_single_avg_norm = interaction_single_avg / (torch.abs(y_complete[:, lbl] - y_none[:, lbl]))
                interaction_single_avg_norm = interaction_single_avg / dataset_normalized
                #print(interaction_single_avg, interaction_single_avg_norm)
                ## interaction_epoch.append([interaction_single_avg, interaction_single_avg_norm, dataset_normalized])
                interaction_avg += interaction_single_avg
                interaction_avg_norm += interaction_single_avg_norm
                if i >= img_num - 1:
                    break
            interaction_avg = interaction_avg / img_num
            interaction_avg_norm = interaction_avg_norm / img_num
            self.net.train()
            return interaction_avg, interaction_avg_norm, dataset_normalized
            ## return interaction_epoch








