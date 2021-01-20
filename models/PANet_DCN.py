# -*- coding: utf-8 -*-
# @Author  : Xiangyi Zhang
# @File    : FewShotSegResnet.py
# @Email   : zhangxy9@shanghaitech.edu.cn

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNetBackbone import resnet50


class OneNet(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()

        # Encoder
        self.encoder = resnet50(cfg=cfg)
        self.device = torch.device('cuda')


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(batch_size):
            ###### Extract bg,fg features ######

            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]  # Wa x Sh x B x H' x W'
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)] # Wa x Sh x B x H' x W'
            ###### Learn Prototype ######
            if self.training:
                out_support = [[self.segmentation(supp_fts[way, shot, [epi]],supp_fg_fts[way, shot],supp_bg_fts[way, shot]) for shot in range(n_shots) ] for way in range(n_ways)] # Wa x Sh x B x H' x W'
            pred = self.segmentation(qry_fts[:,epi],supp_fg_fts,supp_bg_fts)

            ###### Compute the distance ######
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        output_semantic = torch.zeros(1).to(torch.device('cuda'))
        return output, output_semantic, align_loss / batch_size


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = fts * mask[None,...]  # 1 x C x H' x W'
        return masked_fts
    def segmentation(self,fts,supp_fg_fts,supp_bg_fts):
        pass

    def getAlignLoss(self,out_support,):