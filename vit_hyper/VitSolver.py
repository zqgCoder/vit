# -*- coding: utf-8 -*-

"""
@File    : VitSolver.py
@Description:
@Author  : zqgCcoder
@Time    : 2023/2/18 18:28
"""
import sys

import torch.nn
from tqdm import tqdm

import model_vit


class HyperSolver(object):
    """
    先创建一个模型
    """

    def __init__(self, config, device):
        self.config = config
        self.epochs = config.epochs
        self.device = device

        # 超参数网络
        # self.model_hyper = model_vit.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper = model_vit.HyperNet(16, 96, 192, 96, 48, 24, 12, 6).cuda()
        self.model_hyper.train(True)  # 看train方法如何设定
        self.model_hyper.vit.train(True)  # 超网hyperNet中有一个vit模型

        # 损失
        self.bce_loss = torch.nn.BCELoss().cuda()
        self.l1_loss = torch.nn.L1Loss().cuda()

        # vit网络参数，可能还需要设置梯度
        # self.model_vit = model_vit.vit_base_patch16_224_in21k(num_classes=2, has_logits=False).cuda()
        ckpt = torch.load('vit_base_patch16_224.pth')
        if 'head.bias' in list(ckpt.keys()):
            ckpt.pop('head.bias')
        ckpt.pop('head.weight')
        self.model_hyper.vit.load_state_dict(ckpt, strict=False)

        # 设置学习的超参数，vit和超网
        backbone_params = list(map(id, self.model_hyper.vit.parameters()))

        self.hyper_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        # self.lr
        self.lr = self.config.lr
        self.lrratio = self.config.lr_ratio  # 超网的学习率
        self.weight_decay = self.config.weight_decay
        paras = [{'params': self.hyper_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.vit.parameters(), 'lr': self.lr}]

        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)  # 优化器

    def train(self, data_loader, epoch):
        acc_loss = torch.zeros(1).to(self.device)
        accu_num = torch.zeros(1).to(self.device)
        threshold = 0.5
        self.optimizer.zero_grad()  # 梯度清零

        sample_num = 0
        data_loader = tqdm(data_loader, file=sys.stdout)

        for step, data in enumerate(data_loader):
            images, labels = data
            if len(labels) != 8:
                print('batch size is not 8' * 10)
                continue
            sample_num += images.shape[0]
            images = torch.as_tensor(images.cuda())
            labels = torch.as_tensor(labels.cuda())

            # 生成目标网络的w & b
            paras = self.model_hyper(images)

            # Build the target network
            model_target = model_vit.TargetNet(paras).cuda()
            for param in model_target.parameters():
                param.requires_grad = False
            # 目标网络传入输入并且进行prediction
            pred = model_target(paras['target_in_vec'])
            # print(pred.shape)
            # 目前采用二分类，需要进一步修改
            # pred_classes = torch.max(pred, dim=1)[1]
            # accu_num += torch.eq(pred_classes, labels).sum()
            accu_num += torch.gt(pred, threshold).sum()
            # loss = self.bce_loss(pred, labels) # crossentropyloss
            loss = self.bce_loss(pred, labels.to(torch.float))  # bce
            loss.requires_grad_(True)
            # print('loss', loss)
            loss.backward()
            acc_loss += loss.detach()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   acc_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return acc_loss.item() / (step + 1), accu_num.item() / sample_num

    @torch.no_grad()
    def evaluate(self, data_loader, epoch):
        loss_function = self.bce_loss

        # model.eval()
        self.model_hyper.train(False)
        self.model_hyper.vit.train(False)
        # self.model_vit.train(False)

        accu_num = torch.zeros(1).to(self.device)  # 累计预测正确的样本数
        accu_loss = torch.zeros(1).to(self.device)  # 累计损失
        threshold = 0.5

        sample_num = 0
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            if len(labels) != 8:
                print('test: batch size is not 8' * 10)
                continue
            sample_num += images.shape[0]

            paras = self.model_hyper(images.to(self.device))
            model_target = model_vit.TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            # It is assumed that greater than the threshold is true
            # pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.gt(pred, threshold).sum()

            loss = loss_function(pred, labels.to(self.device).to(torch.float))
            accu_loss += loss

            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num
