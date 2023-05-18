import os
import torch
import torchvision
import random
import codecs
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import VisionDataset, download_url, download_and_extract_archive, extract_archive, \
    makedir_exist_ok, verify_str_arg, read_label_file, read_image_file, read_sn3_pascalvincent_tensor, \
    open_maybe_compressed_file, get_int

def set_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MNISTuser(VisionDataset):
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    def __init__(self, root, train=True, fraction=0.1, corrupt_rate=0.05, seed=0, download=False, transform=None, target_transform=None):
        super(MNISTuser, self).__init__(root,transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data, targets = torch.load(os.path.join(self.processed_folder, data_file))

        data = data.numpy()
        targets = targets.numpy()
        #self.targets = targets
        set_seed_pytorch(seed)
        if(train):
            num_every_class = int(data.shape[0] * fraction / 10)
            num_corrupt = int(num_every_class * corrupt_rate)
            for i in range(10):
                self.data.append((data[targets == i])[:num_every_class])
                for count in range(num_every_class):
                    if(count < (num_every_class - num_corrupt)):
                        self.targets.append([i, 1])
                    else:
                        lbl = np.random.choice((list(range(i)) + list(range((i+1),10))))
                        self.targets.append([lbl, 0])
                #self.targets.append((targets[targets == i])[:num_every_class])
                #print(train_targets[train_targets == i].shape)
            self.data = np.concatenate(self.data[:], 0)
            #self.targets = np.concatenate(self.targets[:], 0)
            set_seed_pytorch(seed)
            fraction_shuffle_train = f'./MNIST_fraction_shuffle_train_{self.fraction}_{seed}.npy'
            if(not os.path.exists(fraction_shuffle_train)):
                idx = list(range(len(self.targets)))
                np.random.shuffle(idx)
                np.save(fraction_shuffle_train, np.array(idx))
            else:
                idx = np.load(fraction_shuffle_train)
            self.data = self.data[idx]
            #self.targets = self.targets[idx]
            self.targets = [self.targets[i] for i in idx]
        else:
            self.data = data
            self.targets = targets

        #self.data = data[np.logical_or(np.array(targets) == 0, np.array(targets) == 1)]
        #self.targets = ((targets[np.logical_or(np.array(targets) == 0, np.array(targets) == 1)]) - 0.5) * 2

    def __getitem__(self, index):
        if(self.train):
            img, target = self.data[index], int((self.targets[index])[0])
        else:
            img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class CIFAR10Subset(Dataset):
    def __init__(self, root, train=True, download=False, fraction=0.01, fraction_test=0.01, transform=None, target_transform=None):

        self.data = []
        self.targets = []
        self.fraction = fraction
        self.fraction_test = fraction_test
        self.transform = transform
        self.target_transform = target_transform

        if train:
            trainset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
            train_targets = np.array(trainset.targets)
            num_every_class = int(trainset.data.shape[0] * self.fraction / 10)
            for i in range(10):
                self.data.append((trainset.data[train_targets == i])[:num_every_class])
                self.targets.append((train_targets[train_targets == i])[:num_every_class])
                #print(train_targets[train_targets == i].shape)
            self.data = np.concatenate(self.data[:], 0)
            self.targets = np.concatenate(self.targets[:], 0)
            fraction_shuffle_train = f'./fraction_shuffle_train_{self.fraction}.npy'
            if(not os.path.exists(fraction_shuffle_train)):
                idx = list(range(len(self.targets)))
                np.random.shuffle(idx)
                np.save(fraction_shuffle_train, np.array(idx))
            else:
                idx = np.load(fraction_shuffle_train)
            self.data = self.data[idx]
            self.targets = self.targets[idx]
        else:
            testset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
            test_targets = np.array(testset.targets)
            num_every_class = int(testset.data.shape[0] * self.fraction_test / 10)
            for i in range(10):
                self.data.append((testset.data[test_targets == i])[:num_every_class])
                self.targets.append((test_targets[test_targets == i])[:num_every_class])
                #print(test_targets[test_targets == i].shape)
            self.data = np.concatenate(self.data[:], 0)
            self.targets = np.concatenate(self.targets[:], 0)
            fraction_shuffle_test = f'./fraction_shuffle_test_{self.fraction_test}.npy'
            if(not os.path.exists(fraction_shuffle_test)):
                idx = list(range(len(self.targets)))
                np.random.shuffle(idx)
                np.save(fraction_shuffle_test, np.array(idx))
            else:
                idx = np.load(fraction_shuffle_test)
            self.data = self.data[idx]
            self.targets = self.targets[idx]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class tinyImageNetSubset(Dataset):
    def __init__(self, root, state='train', fraction=0.1, seed=0, transform=None, target_transform=None):

        self.data = []
        self.transform = transform
        self.target_transform = target_transform

        samples = torchvision.datasets.ImageFolder(root=root)
        #print(len(samples.imgs))
        num_every_class = int(len(samples.imgs) * fraction / 200)
        #print(num_every_class)
        for i in range(10):
            count = 0
            for sample in samples.imgs:
                if(sample[1] == i):
                    self.data.append(sample)
                    count = count + 1
                    if(count == num_every_class):
                        break

        set_seed_pytorch(seed)
        if(state == 'train'):
            fraction_shuffle = f'./tinyimagenet_fraction_shuffle_{state}_{num_every_class}.npy'
            if(not os.path.exists(fraction_shuffle)):
                idx = list(range(len(self.data)))
                np.random.shuffle(idx)
                np.save(fraction_shuffle, np.array(idx))
            else:
                idx = np.load(fraction_shuffle)

            self.data = [self.data[j] for j in idx]

    def __getitem__(self, index):
        img_path, target = self.data[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class tinyImageNetSubset_corrupt(Dataset):
    def __init__(self, root, state='train', fraction=0.1, corrupt_rate=0.05, seed=0, transform=None, target_transform=None):

        self.data = []
        self.transform = transform
        self.target_transform = target_transform

        samples = torchvision.datasets.ImageFolder(root=root)
        #print(len(samples.imgs))
        num_every_class = int(len(samples.imgs) * fraction / 200)
        num_corrupt = int(num_every_class * corrupt_rate)
        #print(num_every_class)
        corrupt_label = f'./tinyimagenet_corrupt_{num_corrupt}_{seed}.npy'
        if(os.path.exists(corrupt_label)):
            file_exist = True
            corrupt_img = np.load(corrupt_label, allow_pickle=True).item()
        else:
            file_exist = False
            corrupt_img = {}
        set_seed_pytorch(seed)
        for i in range(10):
            count = 0
            for sample in samples.imgs:
                if(sample[1] == i):
                    count = count + 1
                    sample_ = list(sample)
                    if((state == 'train') and (count > (num_every_class - num_corrupt))):
                        if file_exist:
                            sample_[1] = (corrupt_img[sample_[0]])[1]
                        else:
                            sample_[1] = np.random.choice((list(range(i)) + list(range((i+1),10))))
                            corrupt_img[sample_[0]] = [i, sample_[1]]
                    self.data.append(sample_)
                    #count = count + 1
                    if(count == num_every_class):
                        break
        if((not file_exist) and state == 'train'):
            np.save(corrupt_label, corrupt_img)

        set_seed_pytorch(seed)
        if(state == 'train'):
            fraction_shuffle = f'./tinyimagenet_fraction_shuffle_{state}_{num_every_class}_{seed}.npy'
            if(not os.path.exists(fraction_shuffle)):
                idx = list(range(len(self.data)))
                np.random.shuffle(idx)
                np.save(fraction_shuffle, np.array(idx))
            else:
                idx = np.load(fraction_shuffle)

            self.data = [self.data[j] for j in idx]

    def __getitem__(self, index):
        img_path, target = self.data[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img_path

    def __len__(self):
        return len(self.data)
