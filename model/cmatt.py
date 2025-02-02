__author__ = "Jie Lei, Jiapeng Li"
# Original code by Jie Lei
# Updated intentQA support and CaVIR model by Jiapeng Li on 2023-03-15

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CMAtten(nn.Module):
    
    def __init__(self):
        super(CMAtten, self).__init__()
       

    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        s = torch.bmm(s1, s2.transpose(1, 2))

        # import ipdb; ipdb.set_trace()
        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            s_mask[i][:l_1, :l_2] = 0

        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data, -float("inf"))
        return s

    @classmethod
    def get_u_tile(cls, s, s2):
        """
        attended vectors of s2 for each word in s1,
        signify which words in s2 are most relevant to words in s1
        """
        a_weight = F.softmax(s, dim=2)  # [B, l1, l2]
        # remove nan from softmax on -inf
        # print(a_weight.shape, s2.shape)
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)
        # [B, l1, l2] * [B, l2, D] -> [B, l1, D]
        u_tile = torch.bmm(a_weight, s2)
        return u_tile, a_weight


    def forward(self, s1, l1, s2, l2):
        s = self.similarity(s1, l1, s2, l2)
        u_tile, a_weight = self.get_u_tile(s, s2)
        
        return u_tile, a_weight


class CMAtten_Align(nn.Module):

    def __init__(self):
        super(CMAtten_Align, self).__init__()

    def similarity(self, s1, s2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        s = torch.bmm(s1, s2.transpose(1, 2))

        return s

    @classmethod
    def get_u_tile(cls, s, s2):
        """
        attended vectors of s2 for each word in s1,
        signify which words in s2 are most relevant to words in s1
        """
        a_weight = F.softmax(s, dim=2)  # [B, l1, l2]
        # remove nan from softmax on -inf
        # print(a_weight.shape, s2.shape)
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)
        a_weight_like = torch.zeros_like(a_weight)
        arg_list = torch.argmax(a_weight, dim=2)
        for i in range(len(arg_list[:])):
            for j in range(len(arg_list[i])):
                arg = arg_list[i, j]
                a_weight_like[i, j, arg] = 1
            # print(a_weight_like[i])
        # [B, l1, l2] * [B, l2, D] -> [B, l1, D]
        u_tile = torch.bmm(a_weight_like, s2)

        return u_tile,a_weight_like



    def forward(self, s1, s2):
        s = self.similarity(s1, s2)
        u_tile, a_weight = self.get_u_tile(s, s2)

        return u_tile, a_weight


class CM_grounding(nn.Module):

    def __init__(self):
        super(CM_grounding, self).__init__()

    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        s = torch.bmm(s1, s2.transpose(1, 2))

        # import ipdb; ipdb.set_trace()
        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            s_mask[i][:l_1, :l_2] = 0

        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data, -float("inf"))
        return s

    @classmethod
    def get_u_tile(cls, s, s2):
        """
        attended vectors of s2 for each word in s1,
        signify which words in s2 are most relevant to words in s1
        """
        a_weight = torch.mean(s, dim=1)
        a_weight = F.softmax(a_weight, dim=1)  # [B, l1, L2]
        # remove nan from softmax on -inf
        # print(a_weight.shape, s2.shape)
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)
        # [B, l1] * [B, l1, D] -> [B, l1, D]
        a_weight = a_weight.unsqueeze(-1)
        # print(a_weight.shape, s2.shape)
        # print(a_weight.shape, s1.shape)
        u_tile = a_weight * s2
        return u_tile, a_weight

    def forward(self, s1, l1, s2, l2):
        s = self.similarity(s1, l1, s2, l2)
        u_tile, a_weight = self.get_u_tile(s, s2)

        return u_tile, a_weight
