import torch.nn as nn
import torch
import numpy as np
import utils.lovasz_losses as L
from torch.nn import functional as F
from torch.nn import Parameter
from .loss import OhemCrossEntropy2d
from dataset.target_generation import generate_edge


# class ConsistencyLoss(nn.Module):
#     def __init__(self, ignore_index=255):
#         super(ConsistencyLoss, self).__init__()
#         self.ignore_index=ignore_index

#     def forward(self, parsing, edge, label):
#         parsing_pre = torch.argmax(parsing, dim=1)
#         parsing_pre[label==self.ignore_index]=self.ignore_index
#         generated_edge = generate_edge(parsing_pre)
#         edge_pre = torch.argmax(edge, dim=1)
#         v_generate_edge = generated_edge[label!=255]
#         v_edge_pre = edge_pre[label!=255]
#         v_edge_pre = v_edge_pre.type(torch.cuda.FloatTensor)
#         positive_union = (v_generate_edge==1)&(v_edge_pre==1) # only the positive values count
#         return F.smooth_l1_loss(v_generate_edge[positive_union].squeeze(0), v_edge_pre[positive_union].squeeze(0))

class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.ConsEdge = ConsistencyLoss(ignore_index=ignore_index)
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)  
        # self.l2Loss = torch.nn.MSELoss(reduction='mean')
        self.l2loss = torch.nn.MSELoss()
           
    def parsing_loss(self, preds, target, hwgt):
        h, w = target[0].size(1), target[0].size(2)
        b = hwgt[0].size(0)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0

        # loss for parsing
        pws = [0.4, 1, 1, 1]
        preds_parsing = preds[0]
        ind = 0
        tmpLoss = 0
        real_loss = 0


        # parsing_target = target[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                if ind >= 1:
                    parsing_target = torch.cat([target[0], target[0]], dim=0)
                else:
                    parsing_target = target[0]
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                # fea_visual(scale_pred)
                tmpLoss = self.criterion(scale_pred, parsing_target)
                scale_pred = F.softmax(scale_pred, dim=1)
                tmpLoss += L.lovasz_softmax(scale_pred, parsing_target, ignore=self.ignore_index)
                tmpLoss *= pws[ind]
                loss += tmpLoss
                ind += 1
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])
            # scale_pred = F.softmax( scale_pred, dim = 1 )
            # loss += L.lovasz_softmax( scale_pred, target[0], ignore = self.ignore_index )
        real_loss += loss

        set1 = [1, 2, 4, 13]
        set2 = [3, 5, 7, 10, 11, 12, 14,15]
        set3 = [6, 8, 9, 16, 17, 18,19]

        set = [set1, set2, set3]
        target_t = target[0].clone()
        count = 1.
        for index in set:
            for value in index:
                target_t[target[0] == value] = count
            count += 1.

        aff_ = preds[-1]
        scale_pred = F.interpolate(input=aff_, size=(h, w),
                                   mode='bilinear', align_corners=True)
        aff_loss = self.criterion(scale_pred, target_t)
        loss += aff_loss * 3

        aff = preds[-2]
        target_t = target_t.unsqueeze(1).repeat(1, b, 1, 1).view(b * b, h, w)
        scale_pred = F.interpolate(input=aff, size=(h, w),
                                   mode='bilinear', align_corners=True)
        aff_loss = self.criterion(scale_pred, target_t)
        loss += aff_loss * 3

        # loss for edge
        tmpLoss = 0
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                tmpLoss += F.cross_entropy(scale_pred, target[1],
                                           weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            tmpLoss += F.cross_entropy(scale_pred, target[1],
                                       weights.cuda(), ignore_index=self.ignore_index)
        loss += tmpLoss
        real_loss += tmpLoss
        # loss for height and width attention
        #  loss for hwattention
        hwLoss = 0
        preds_hw = preds[2]
        i = 0
        if len(preds_hw) == 4:
            pw = [0.4, 0.4, 0.6, 0.6]

        if len(preds_hw) == 2:
            pw = [0.4, 0.6]

        if isinstance(preds_hw, list):
            for pred_hw in preds_hw:
                hgt = hwgt[0]
                wgt = hwgt[1]
                n, c, h = hgt.size()
                w = wgt.size()[2]
                hpred = pred_hw[0]  # fea_h...
                wpred = pred_hw[1]  # fea_w...
                if i == (len(preds_hw) - 1):
                    hgt = torch.cat([hwgt[0], hwgt[0]], dim=0)
                    wgt = torch.cat([hwgt[1], hwgt[1]], dim=0)

                else:
                    if hpred.size(0) == n ** 2:
                        hgt = hgt.unsqueeze(1).repeat(1, n, 1, 1).view(n ** 2, c, h)
                    if wpred.size(0) == n ** 2:
                        wgt = wgt.unsqueeze(1).repeat(1, n, 1, 1).view(n ** 2, c, w)
                scale_hpred = hpred.unsqueeze(3)  # n,c,h,1
                scale_hpred = F.interpolate(input=scale_hpred, size=(h, 1), mode='bilinear', align_corners=True)
                scale_hpred = scale_hpred.squeeze(3)  # n,c,h
                # hgt = hgt[:,1:,:]
                # scale_hpred=scale_hpred[:,1:,:]
                hloss = torch.mean((hgt - scale_hpred) * (hgt - scale_hpred))

                scale_wpred = wpred.unsqueeze(2)  # n,c,1,w
                scale_wpred = F.interpolate(input=scale_wpred, size=(1, w), mode='bilinear', align_corners=True)
                scale_wpred = scale_wpred.squeeze(2)  # n,c,w
                # wgt=wgt[:,1:,:]
                # scale_wpred = scale_wpred[:,1:,:]
                wloss = torch.mean((wgt - scale_wpred) * (wgt - scale_wpred))
                hwLoss += (hloss + wloss) * 40 * pw[i]
                i += 1
        loss += hwLoss
        real_loss += hwLoss
        print(
            'real_loss = {}'.format(real_loss.data.cpu().numpy()))
        return loss

    def forward(self, preds, target, hwgt ):
          
        loss = self.parsing_loss(preds, target, hwgt  ) 
        return loss
    