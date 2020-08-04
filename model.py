# -*- coding: utf-8 -*-
# @Time : 2020/7/22 13:54
# @Author : cos0sin0
# @Email : cos0sin0@qq.com
import torch
import torch.nn as nn
import resnet
import decoder
from loss import L1BalanceCELoss


class DBNet(nn.Module):
    def __init__(self):
        super(DBNet, self).__init__()
        self.backbone = resnet.resnet18()
        self.decoder = decoder.seg_detector()
        self.multiloss = L1BalanceCELoss()

    def forward(self,x):
        feature_map = self.backbone(x)
        result = self.decoder(feature_map)
        return result

    def loss(self,pred,tag):
        loss = self.multiloss(pred,tag)
        return loss

