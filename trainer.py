# -*- coding: utf-8 -*-
# @Time : 2020/7/22 13:39
# @Author : cos0sin0
# @Email : cos0sin0@qq.com
from model import DBNet
from dataloader import ImageDataset,SroieOcrDataset
import numpy as np
import torch
import cv2
from torch import optim
from representer import map2box
from visual import show_boxes

class DBNetTrainer():
    def __init__(self,data_dir,batch_size,epoch,lr=0.001,weight_decay=1e-4,checkpoint=None):
        self.dbnet = DBNet()
        self.dataset = SroieOcrDataset(data_dir)
        self.batch_size = batch_size
        self.epoch = epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dbnet.to(self.device)
        self.optimizer = optim.Adam(params=self.dbnet.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        self.dbnet.train()
        self.checkpoint = checkpoint


    def run(self,batch_percentl=0.1):
        if self.checkpoint is not None:
            print("loading model from checkpoint")
            self.dbnet.load_state_dict(torch.load(self.checkpoint))
        print('total train num:',len(self.dataset))
        batch_num = len(self.dataset)//self.batch_size+1
        min_loss = 2
        for ep in range(self.epoch):
            indices = list(range(len(self.dataset)))
            np.random.shuffle(indices)
            losses = 0
            batch_interval = int(batch_num*batch_percentl)
            for i in range(batch_num):
                loss,metrics = self.batch_run(indices,i,batch_interval)
                losses+=loss
            mean_loss = losses/batch_num
            print("current epoch {}\ncurrent loss:{}".format(ep,mean_loss))

            if mean_loss < min_loss:
                print('save model ...')
                min_loss = mean_loss
                torch.save(self.dbnet.state_dict(), 'model/model_epoch_{}_loss_{:.4f}.pth'.format(ep, mean_loss))

            if ep%2==0 and ep>=0:
                self.validate(indices[0:1])


    def batch_run(self,indices,i,batch_interval):
        batch = indices[i * self.batch_size:(i + 1) * self.batch_size]
        if len(batch) == 0:
            return 0,None
        imgs, gts, masks, thresh_maps, thresh_masks = [], [], [], [], []
        for id in batch:
            img, gt, mask, thresh_map, thresh_mask = self.dataset[id]
            imgs.append(img)
            gts.append(gt)
            masks.append(mask)
            thresh_maps.append(thresh_map)
            thresh_masks.append(thresh_mask)

        imgs = torch.Tensor(imgs).to(self.device)
        # print('imgs shape:',imgs.shape)
        gts = torch.Tensor(gts).to(self.device)
        masks = torch.Tensor(masks).to(self.device)
        thresh_masks = torch.Tensor(thresh_masks).to(self.device)
        thresh_maps = torch.Tensor(thresh_maps).to(self.device)
        pred = self.dbnet(imgs)
        tags = {'gt': gts, 'mask': masks, 'thresh_map': thresh_maps, 'thresh_mask': thresh_masks}
        loss ,metrics= self.dbnet.loss(pred, tags)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        if i%batch_interval == 0:
            print("current batch {} loss {},metrics {}".format(i,loss,metrics))
        return loss,metrics


    def validate(self,subindices):
        self.dbnet.eval()
        for id in subindices:

            img, gt, mask, thresh_map, thresh_mask = self.dataset[id]
            height,width = img.shape[-2:]
            timg = torch.Tensor([img]).to(self.device)
            result = self.dbnet(timg)
            binary = result['binary']
            t_binary = result['binary'].cpu().detach().numpy()[0][0]
            # t_binary = t_binary < 0.3
            # boxes,socres = map2box(np.array(gt),width,height)
            boxes,socres = map2box(binary,width,height)
            if len(boxes)>=0:
                t_binary = t_binary > 0.3
                t_binary = t_binary * 255
                cv2.imwrite('log/bin.jpg', t_binary)
                image = np.transpose(img, (1, 2, 0))*255
                show_boxes(image,boxes)
        self.dbnet.train()


if __name__ == '__main__':
    dbnt = DBNetTrainer('data/sroie/ocr',1,1200,checkpoint='model/model_epoch_53_loss_0.9771.pth'
                                                           '')
    dbnt.run()
