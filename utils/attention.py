import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import functools
# from visual import visual_offset
from einops import rearrange
# from utils.feature_visual import fea_visual
# from visual import visual_on_image

from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import BatchNorm1d as BatchNorm1d


def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


def conv1d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


class Deformable_Batchformer(nn.Module):
    def __init__(self, in_channel, n_head, forward_dim, hidden_dim, size=[473 // 4, 473 // 4]):
        super(Deformable_Batchformer, self).__init__()
        h, w = size[0], size[1]
        self.F = nn.Conv2d(in_channel, hidden_dim, 1)
        self.key = nn.Conv2d(in_channel, in_channel, 1)
        self.value = nn.Conv2d(in_channel, in_channel, 1)
        self.head = n_head
        self.softmax = Softmax(2)
        self.linear1 = nn.Linear(in_channel, in_channel)
        self.linear2 = nn.Linear(in_channel, in_channel)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.rowpool = nn.AdaptiveAvgPool2d((None, 1))
        self.colpool = nn.AdaptiveAvgPool2d((1, None))

        # self.head = n_head
        self.detecor = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=9, stride=1, padding=4, groups=hidden_dim, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def pro(self, q, x):
        b, _, h, w = x.size()

        h_dis = q.view(b, 1, -1, h * w).repeat(1, b, 1, 1)
        w_dis = x.view(1, b, -1, h * w).repeat(b, 1, 1, 1)

        fea = torch.cat([h_dis, w_dis], dim=2).view(b * b, -1, h, w)
        flow = self.detecor(fea)
        return flow

    def flow_warp(self, input, flow):
        n, c, h, w = input.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(input).to(input.device)
        x = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        y = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)

        trans = torch.tensor([[0., 1.], [-1., 0.]]).to(input.device)
        reverse = torch.tensor([[1., 0.], [0., -1.]]).to(input.device)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) * 9 / norm

        grid = (grid.view(n * h * w, 2) @ reverse @ trans).view(n, h, w, 2)

        output = F.grid_sample(input, grid, align_corners=True, padding_mode="zeros")
        return output, grid

    def att(self, q, kv):
        b, c, h, w = q.size()

        q = q.view(b, 1, -1, self.head*h*w) # (b*head,1,c//head,h*w)
        k = self.key(kv).contiguous().view(b, b, -1, h * w).contiguous().view(b, b, -1, self.head, h * w)# (b*head,b,c//head,h*w)
        k = k.view(b,b,-1,h*w*self.head)# (b,b,c//head,h*w*head)

        v = self.value(kv).contiguous().view(b, b, -1, h * w).contiguous().view(b, b, -1, self.head, h * w)
        v = v.view(b,b,-1,h*w*self.head)# (b,b,c//head,h*w*head)

        long = q.size(2)
        energy = torch.einsum("kisn,kjsn->kijn", q, k) * (long ** -0.5)
        att_ = self.softmax(energy)  # (b*head,1,b,h*w)

        out = torch.einsum("kisn,ksjn->kijn", att_, v).contiguous().view(b, c, h, w)  # (b*head,1,c,h*w)
        return self.dropout1(out)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x, **kwargs):
        x_org = x
        x_ = self.F(x)
        aff_ = x
        batch_size, c, h, w = x.size()
        offset = self.pro(x_, x_)  # (b*b,head*2,h,w)
        aff = torch.eye(batch_size).unsqueeze(2).repeat(1, 1, 2 * h * w).view(batch_size * batch_size, 2, h, w).to(
            x.device)  # (b*b*head,2,h,w)
        offset = offset.view(batch_size * batch_size, 2, h, w)  # (b*b*head,2,h,w)
        offset = (1 - aff) * offset
        flow_fea = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).view(batch_size * batch_size, -1, h, w)  # (b*b,c,h,w)

        # if 'mask' in kwargs.keys():
        #     if kwargs['mask'] is not None:
        #         visual_offset(offset,**kwargs)

        kv_d, _ = self.flow_warp(flow_fea.detach(), offset)  # (b*b,c,h,w)
        kv, _ = self.flow_warp(flow_fea, offset)
        aff = self.att(x, kv)
        # x = x.permute(0,2,3,1).contiguous().view(batch_size*h*w,-1)
        x = self.norm1((aff+x).permute(0,2,3,1).contiguous().view(-1, c))
        # x = (aff).permute(0,2,3,1).contiguous().view(-1, c)
        x = self.norm2(self._ff_block(x) + x).contiguous().view(batch_size, h, w, -1).permute(0,3,1,2)
        return x, kv_d, aff_.detach()


class ACCAttention(nn.Module):

    def __init__(self, in_dim=512, mid_dim=512, out_dim=512, num_classes=20, size=[384 // 16, 384 // 16],
                 is_train=False, head=1, kernel_size=7):
        super(ACCAttention, self).__init__()
        h, w = size[0], size[1]
        self.crtt = CCAttention_vCR(in_dim, mid_dim, out_dim, token='all')
        # self.crtt = CrissCrossAttention(in_dim)

        self.mid = mid_dim
        self.num_classes = num_classes

        self.h_conv1 = nn.Sequential(
            nn.Conv1d(out_dim//4, num_classes, 3, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.w_conv1 = nn.Sequential(
            nn.Conv1d(out_dim//4, num_classes, 3, stride=1, bias=True),
            nn.Sigmoid(),
        )

        self.is_train = False
        if self.is_train:
            self.deformable_bf1 = BatchFormer_with_deformal(in_dim, 4, out_dim, 256)
            self.read_out = nn.Conv2d(in_dim,4,1)

        self.rowpool = nn.AdaptiveAvgPool2d((h, 1))
        self.colpool = nn.AdaptiveAvgPool2d((1, w))

        self.relu = nn.ReLU()
        self.ccatt = CCAttention_vCR(in_dim, mid_dim, out_dim, token='all')

    def hw_fea_get(self, fea_hw):
        fea_h, fea_w = fea_hw[0], fea_hw[1]
        fea_wp1 = self.colpool(fea_w).squeeze(2)  # bs,c,w/16
        fea_wp1 = self.w_conv1(fea_wp1)  # bs,num_class,w/16
        fea_hp1 = self.rowpool(fea_h).squeeze(3)  # bs,c,h/16
        fea_hp1 = self.h_conv1(fea_hp1)  # bs,num_class,h/16
        return fea_hp1, fea_wp1

    def forward(self, fea,**kwargs):
        hw_fea_list = []

        fea, fea_1 = self.crtt(fea, **kwargs)  # bs , c ,h/16 ,w/16
        if self.is_train:
            fea, aff, aff_ = self.deformable_bf1(fea, first=True, **kwargs)
        fea, fea_2 = self.ccatt(fea, **kwargs)

        hw_fea_list.append(fea_1)
        hw_fea_list.append(fea_2)

        hw_list = []
        for hw in hw_fea_list:
            p = self.hw_fea_get(hw)
            hw_list.append(p)

        if self.is_train:
            org_fp = self.read_out(aff_)
            warped_fp = self.read_out(aff)
            return fea, [org_fp, warped_fp]

        return fea, hw_list


class BatchFormer_with_deformal(nn.Module):
    def __init__(self, in_channel, n_head, forward_dim, hidden_dim):
        super(BatchFormer_with_deformal, self).__init__()
        self.def_bf = Deformable_Batchformer(in_channel, n_head, forward_dim, hidden_dim)

    def forward(self, x, first=False, **kwargs):
        if first:
            org = x
        else:
            dim = x.size(0) // 2
            org = x[:dim]
            x = x[dim:]
        x, aff, aff_ = self.def_bf(x,**kwargs)
        x = torch.cat([org, x], dim=0)
        return x, aff, aff_

class BatchFormer(nn.Module):
    def __init__(self, in_channel, n_head, forward_dim):
        super(BatchFormer, self).__init__()
        self.bf = torch.nn.TransformerEncoderLayer(in_channel, n_head, forward_dim, batch_first=True, dropout=0.5)

    def forward(self,x, first=False, **kwargs):
        if first:
            org = x
        else:
            dim = x.size(0) // 2
            org = x[:dim]
            x = x[dim:]
        b, c ,h , w = x.size()
        x = x.view(b,c,-1).permute(2,0,1)
        x = self.bf(x).permute(1,2,0)
        x = x.view(b, c ,h , w)
        x = torch.cat([org, x], dim=0)
        return x


class CCAttention_vCR(nn.Module):

    def __init__(self, in_dim=512, mid_dim=512, out_dim=512, head=4, token=None):
        '''
        Parameters
        ----------
        in_dim : int
            channels of input
        '''
        super(CCAttention_vCR, self).__init__()
        self.head = head
        self.qk_dim = out_dim
        self.hw_dim = in_dim // 2

        self.softmax = Softmax(dim=3)
        if token == "Hori" or 'all':
            self.query_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=self.qk_dim, kernel_size=1,groups=2)
            self.key_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=self.qk_dim, kernel_size=1,groups=2)
            self.bnhd = nn.BatchNorm2d(out_dim // 2)

        if token == "Verti" or 'all':
            self.query_conv_w = self.query_conv_h
            self.key_conv_w = self.key_conv_h
            self.bnwd = nn.BatchNorm2d(out_dim // 2)

        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim//2 , kernel_size=1,groups=2)
        self.gamma = nn.Parameter(torch.ones(1)/10)
        # self.beta = nn.Parameter(torch.ones(1))

        self.FFN = nn.Conv2d(out_dim, out_dim, 1)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        self.token = token

    def forward(self, x,**kwargs):
        # mask = None
        org_x = x
        m_batchsize, _, height, width = x.size()
        head = self.head
        x= torch.cat([x[:,:self.hw_dim,:,:].detach(),x[:,self.hw_dim:,:,:]],dim=1)
        adding = 0
        p = []
        if self.token == 'Hori' or 'all':
            x_h = x

            padding_h = torch.zeros((m_batchsize, _, adding, width)).to(x.device)
            x_h = torch.cat([x_h, padding_h], dim=2)
            proj_query_h = self.query_conv_h(x_h)

            proj_query_h = proj_query_h.contiguous().view(m_batchsize * head, -1, height + adding, width).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2).contiguous().view(
                m_batchsize * head * width, -1, height + adding).permute(0, 2,
                                                                         1)
            proj_key_h = self.key_conv_h(x_h)

            proj_key_h = proj_key_h.contiguous().view(m_batchsize * head, -1, height + adding, width).permute(0, 3, 1,
                                                                                                              2).contiguous().view(
                m_batchsize * head * width, -1, height + adding)
            proj_value = self.value_conv(x)

            proj_value_H = proj_value.contiguous().view(m_batchsize * head, -1, height, width).permute(0, 3, 1,
                                                                                                       2).contiguous().view(
                m_batchsize * head * width, -1, height)
            _, _, long = proj_query_h.shape
            energy_H = torch.bmm(proj_query_h, proj_key_h).contiguous().view(m_batchsize * head, width, height + adding,
                                                                             height + adding).permute(0, 2, 1, 3) * (
                               long ** -0.5)
            energy_H = self.softmax(energy_H)[:, :height, :, :height]
            att_H = energy_H.permute(0, 2, 1, 3).contiguous().view(m_batchsize * head * width, height, height)
            out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).contiguous().view(m_batchsize * head, width,
                                                                                      -1, height).permute(0, 2, 3,
                                                                                                          1).contiguous().view(
                m_batchsize, -1, height, width)
            out_H_ = out_H
            out_H = self.bnhd(out_H)
            p.append(out_H[:,:self.hw_dim//2,:,:])

        if self.token == 'Verti' or 'all':
            x_w = x

            proj_query_w = self.query_conv_w(x_w)

            proj_query_w = proj_query_w.contiguous().view(m_batchsize * head, -1, height, width + adding).permute(0, 2,
                                                                                                                  1,
                                                                                                                  3).contiguous().view(
                m_batchsize * head * height, -1, width + adding).permute(0, 2,
                                                                         1)
            proj_key_w = self.key_conv_w(x_w)

            proj_key_w = proj_key_w.contiguous().view(m_batchsize * head, -1, height, width + adding).permute(0, 2, 1,
                                                                                                              3).contiguous().view(
                m_batchsize * head * height, -1, width + adding)

            proj_value_W = proj_value.contiguous().view(m_batchsize * head, -1, height, width).permute(0, 2, 1,
                                                                                                       3).contiguous().view(
                m_batchsize * head * height, -1, width)

            _, _, long = proj_query_w.shape
            energy_W = torch.bmm(proj_query_w, proj_key_w).view(m_batchsize * head, height, width + adding,
                                                                width + adding) * (long ** -0.5)
            energy_W = self.softmax(energy_W)[:, :, :width, :width]
            att_W = energy_W.contiguous().view(m_batchsize * head * height, width, width)
            out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).contiguous().view(m_batchsize * head, height,
                                                                                      -1, width).permute(0, 2, 1,
                                                                                                         3).contiguous().view(
                m_batchsize, -1, height, width)
            out_W_ = out_W
            out_W = self.bnwd(out_W)
            p.append(out_W[:,:self.hw_dim//2,:,:])

        if self.token == 'all':
            out_temp = torch.cat([out_H_, out_W_], dim=1)
        ##可视化横向注意力
        # fea_visual(energy_W[:,:,:,:].permute(0,2,1,3).contiguous().view(-1,width,height).unsqueeze(0))

        ##可视化纵向注意力
        # fea_visual(energy_H[:, :, :, :].contiguous().view(-1, width, height).permute(0,2,1).unsqueeze(0))

        ##热力图显示
        # # # fea = energy_H[:, :, :, :].contiguous().view(-1, width, height).permute(0,2,1).contiguous().view(m_batchsize*head,height,height,width)
        # fea = energy_W[:,:,:,:].permute(0,2,1,3).contiguous().view(m_batchsize*head,width,height,width)
        # for y in range(30):
        #     visual_on_image(fea.detach(),y,0,head=3,index=4,**kwargs)

        # out_H = self.bn1(out_H.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)).contiguous().view(m_batchsize,width,-1,height).permute(0,2,3,1)

        # out_W = self.bn2(out_W.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)).contiguous().view(m_batchsize,height,-1,width).permute(0,2,1,3)
        out = out_temp
        out = self.gamma*self.relu(out)+ org_x

        return out, p

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*self.relu(out_H + out_W) + x,[out_H[:,:128],out_W[:,:128]]


class DHWttention(nn.Module):

    def __init__(self, in_dim=512, mid_dim=512, out_dim=512, num_classes=20, size=[384 // 16, 384 // 16],
                 is_train=False, head=1, kernel_size=7):
        super(DHWttention, self).__init__()
        h, w = size[0], size[1]
        self.crtt = DAttention_vCR(in_dim, mid_dim, out_dim, token='all',size=size)

        self.mid = mid_dim
        self.num_classes = num_classes

        self.h_conv1 = nn.Sequential(
            nn.Conv1d(mid_dim//2, num_classes, 3, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.w_conv1 = nn.Sequential(
            nn.Conv1d(mid_dim//2, num_classes, 3, stride=1, bias=True),
            nn.Sigmoid(),
        )

        self.is_train = True
        # self.deformable_bf1 = BatchFormer_with_deformal(in_dim, 4, out_dim, mid_dim)

        self.relu = nn.ReLU()
        self.ccatt = DAttention_vCR(in_dim, mid_dim, out_dim, token='all',size=size)

        # self.read_out = nn.Conv2d(mid_dim,4,1)

    def hw_fea_get(self, fea_hw):
        fea_h, fea_w = fea_hw[0], fea_hw[1]
        fea_wp1 = self.w_conv1(fea_h)  # bs,num_class,w/16
        fea_hp1 = self.h_conv1(fea_w)  # bs,num_class,h/16
        return fea_hp1, fea_wp1

    def forward(self, x,**kwargs):
        hw_fea_list = []
        fea, fea_1 = self.crtt(x, **kwargs)  # bs , c ,h/16 ,w/16

        if self.is_train:
            fea,aff,aff_ = self.deformable_bf1(fea,first=True,**kwargs)

        fea, fea_2 = self.ccatt(fea, **kwargs)
        hw_fea_list.append(fea_1)
        hw_fea_list.append(fea_2)

        hw_list = []
        for hw in hw_fea_list:
            p = self.hw_fea_get(hw)
            hw_list.append(p)

        if self.is_train:
            org_fp = self.read_out(aff_)
            warped_fp = self.read_out(aff)
            return fea, hw_list, [org_fp, warped_fp]

        return fea, hw_list


