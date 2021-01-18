import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class HMMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, gamma=1.1, ldam=False):
        super(HMMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (0.5 / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.max_m = max_m
        self.gamma = gamma
        self.ldam = ldam

    def weight(self, freq_bias, target, args):

        index = torch.zeros_like(freq_bias, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        # plus 1 affects top-1 acc.
        cls_num_list = (index_float.sum(0).data.cpu() + 1)

        beta = args.beta

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)

        return per_cls_weights

    def obj_margins(self, rm_obj_dists, labels, index_float, max_m):

        #obj_dists_ = F.softmax(rm_obj_dists, dim=1)
        obj_neg_labels = 1.0 - index_float
        obj_neg_dists = rm_obj_dists * obj_neg_labels

        min_pos_prob = rm_obj_dists[:, labels.data.cpu().numpy()[0]].data
        max_neg_prob = obj_neg_dists.max(1)[0].data

        # estimate the margin between dists and gt labels
        batch_m_fg = torch.max(
            min_pos_prob - max_neg_prob,
            torch.zeros_like(min_pos_prob))[:,None]

        mask_fg = (batch_m_fg > 0).float()
        batch_fg = torch.exp(-batch_m_fg - max_m * self.gamma) * mask_fg

        batch_m_bg = torch.max(
            max_neg_prob - min_pos_prob,
            torch.zeros_like(max_neg_prob))[:,None]

        mask_ng = (batch_m_bg > 0).float()
        batch_ng = torch.exp(-batch_m_bg - max_m) * mask_ng
        batch_m = batch_ng + batch_fg
        return batch_m.data

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))

        # 1.0 - [0.5] => [0.0 ~ 0.5]
        if self.ldam :
            max_m = self.max_m - batch_m
        else:
            max_m = self.max_m

        with torch.no_grad():
            batch_hmm = self.obj_margins(x, target, index_float, max_m)

        x_m = x - batch_hmm

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
