import torch
import torch.nn as nn

class EasyLoss(nn.Module):
    def __init__(self):
        super(EasyLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.view(-1)
        return torch.mean( - pred * target)


class InterL2Loss(nn.Module):
    def __init__(self, softmax=False):
        super(InterL2Loss, self).__init__()
        self.softmax = softmax

    def forward(self, outputs, target, is_celeba):
        [y_ij, y_i, y_j, y_] = outputs
        bs = y_ij.shape[0]
#         print(y_ij[:,target].shape)
#         return torch.mean((y_ij[:,target.item()] + y_[:,target.item()] - y_i[:,target.item()] - y_j[:,target.item()]) ** 2)
        if not is_celeba:
            temp = torch.tensor(range(bs))
            target_ = target.repeat(10)
            # print(temp, target_, y_ij.size(), y_i.size(), y_j.size(), y_.size())
            if self.softmax:
                y_ij, y_i, y_j, y_ = torch.nn.functional.softmax(y_ij, 1), torch.nn.functional.softmax(y_i, 1), torch.nn.functional.softmax(y_j, 1), torch.nn.functional.softmax(y_, 1)
                delta_f = torch.log(y_ij[temp, target_]) + torch.log(y_[temp, target_]) - torch.log(y_i[temp, target_]) - torch.log(y_j[temp, target_])
            else:
                delta_f = y_ij[temp, target_] + y_[temp, target_] - y_i[temp, target_] - y_j[temp, target_]
                # delta_f = torch.nn.functional.cross_entropy(y_ij, target_) + torch.nn.functional.cross_entropy(y_, target_) - torch.nn.functional.cross_entropy(y_i, target_) - torch.nn.functional.cross_entropy(y_j, target_)
            # if delta_f > self.max_delta_f:
            #     self.max_delta_f = delta_f.detach()
            # coe = 2 * torch.nn.functional.sigmoid(self.alpha / self.max_delta_f * delta_f.detach()) - 1
            return torch.mean(delta_f * delta_f)
        else:
            temp = torch.tensor(range(bs))
            return torch.mean(torch.abs(y_ij[temp, 0] + y_[temp, 0] - y_i[temp, 0] - y_j[temp, 0]))


class GenerateLoss(nn.Module):
    def __init__(self, r=1, tr=1, is_celeba=False, use_interloss=True, softmax=False):
        super(GenerateLoss, self).__init__()
        self.is_celeba = is_celeba
        if is_celeba:
            self.targetloss = nn.BCEWithLogitsLoss()
        else:
            self.targetloss = nn.CrossEntropyLoss()
        self.interloss  = InterL2Loss(softmax=softmax)
        self.r          = r
        self.tr         = tr
        self.use_interloss = use_interloss

    def forward(self, outputs, target, use_interloss=True):
        [y, y_ij, y_i, y_j, y_] = outputs
        loss_target    = self.targetloss(y, target)
        loss_inter     = self.interloss(outputs[1:], target, self.is_celeba)
        self.targetlossvalue = loss_target
        self.interlossvalue = loss_inter
        if(use_interloss):
            #return self.tr * loss_target + self.r * loss_inter
            return loss_target, self.r * loss_inter
        else:
            return loss_target
