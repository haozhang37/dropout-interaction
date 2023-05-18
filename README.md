# Pytorch Implementation for 'Theoretical Understanding of Dropout and Its Alternative Regularization'

## Requriments:

* Python >= 3.6
* Pytorch >= 1.3

##  Train Networks on CIFAR10(10% of total dataset):
python train_cifar10.py [--p_info, rate of dropout] [--epochs, epoch number of training] [--gpu, gpu id to be used]
                        [--set_name, name of dataset] [num_classes, number of classes] [--pretrained, pretrained or not]
                        [--mode, please always set this argument 'test'] [--pos_info, if specified, save the sampled pairs of features] [--val_mode, use trainset or testset to compute interaction(all experiments in the paper use trainset)]
                        [--pos_pair, number of sampled pairs of features, e.g., 500] [--rate, weight of interaction loss] [--target_rate, weight of classification loss]
                        [--sample_set, the ratio of sample set when train the networks, e.g., 0.05] [S_rate, sample interval when train the networks] [--fixed_len, sample set with the same length or not]
                        [--dropout_layer, the position of applying dropout/interaction loss] [--set_seed, set seed or not] [--seed, seed of random number used in the training]
                        [--seed_interaction, seed of random number used in the computation of interaction] [--fraction, ratio of dataset used to train networks, e.g., 0.1] [--fraction_test, ratio of dataset used to test the model, please always keep this value be 1]
                        [--lr, learning rate] [--bs, batch size] [--softmax, use softmax when train ResNets]
                        [network, AlexNet_/vgg11_/vgg16_/vgg19_]

## Example:
Train AlexNet on CIFAR10(10% of total dataset):
python train_cifar10.py AlexNet_ --p_info 0.0 --epochs 300 --gpu 0 --set_name CIFAR10 --num_classes 10 --pretrained 0 --mode test --pos_info pos_500.npy --val_mode trainset --pos_pair 500 --rate 10.0 --target_rate 1.0 --sample_set 0.05 --S_rate 0.0,1.0 --fixed_len 1 --dropout_layer 6 --set_seed 1 --seed 2 --seed_interaction 2 --fraction 0.1 --fraction_test 1 --lr 0.01 --bs 128 --softmax 0

##  Compute interacion between features before the dropout/interaction loss after training:
Use Trainer.compute_interaction() in train_cifar10.py, and use the function get_interaction() in compute_interaction() in trainer_cifar10.py
Args of compute_interaction():
loss_interaction, bool, use loss of network output to compute interaction or not. By default, the function will use output of network to compute the interaction, not loss
mode, str, 'shapley' or 'banzhaf', denotes shapley interaction and banzhaf interaction, respectively
subseed, int, seed of random number when compute instability
Args of get_interaction():
loss_interaction, bool, same with the argument of compute_interaction()
mode, str, same with the argument of compute_interaction()
img_num, int, the number of image used to compute interaction. Default:10
sample_rate, list, sample interval. If specified, the function will sample drop rate of every activation unit from the specified interval. Default:None

