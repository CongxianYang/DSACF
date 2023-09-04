# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from .attention import GlobalAttentionBlock, CBAM
from .RGBT_fusion import RGBT_fusion
class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        ##add attention
        self.template_attention_block = GlobalAttentionBlock()
        self.detection_attention_block = CBAM(512)
        ##Multi-modal feature fusion CA-MF
        ##zf shoudle CIF torch.cat    xf shoudle DFF
        self.feature_fusion=RGBT_fusion()
        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z_rgb,z_t):
        zf_r= self.backbone(z_rgb)
        zf_t=self.backbone(z_t)
        if cfg.ADJUST.ADJUST:
            zf_r= self.neck(zf_r)
            zf_t = self.neck(zf_t)
    # ger attention feature
        for i in range(len(zf_r)):
            zf_r[i], zf_t[i] = self.template_attention_block(zf_r[i], zf_t[i])
        # fusion features (rgb and rgbt)
        features_fusion_z = list(range(3))
        for i in range(len(zf_r)):
            features_fusion_z[i] = self.feature_fusion(zf_r[i], zf_t[i])

        self.zf = features_fusion_z

    def track(self, x_rgb,x_t):
        xf_rgb = self.backbone(x_rgb)
        xf_t = self.backbone(x_t)
        if cfg.ADJUST.ADJUST:
            xf_rgb = self.neck(xf_rgb)
            xf_t = self.neck(xf_t)
            # ger attention feature
        for i in range(len(xf_rgb)):
            union = torch.cat((xf_rgb[i], xf_t[i]), 1)
            xf_rgb[i], xf_t[i] = self.detection_attention_block(union)
        # fusion features (rgb and rgbt)
        xf = list(range(3))
        for i in range(len(xf_rgb)):
            xf[i] = self.feature_fusion(xf_rgb[i], xf_t[i])
        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)
        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template_rgb = data['template_rgb'].cuda()
        template_t = data['template_t'].cuda()
        search_rgb = data['search_rgb'].cuda()
        search_t = data['search_t'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature(rgb anf rgbt)
        zf_rgb = self.backbone(template_rgb)
        zf_t = self.backbone(template_t)

        xf_rgb = self.backbone(search_rgb)
        xf_t = self.backbone(search_t)

       ##adjust layer ##neck为AdjustAllLayer，通过 1×1将这三层调整为256通道。
        if cfg.ADJUST.ADJUST:
            zf_rgb = self.neck(zf_rgb) ##(bs,256,7,7)
            zf_t = self.neck(zf_t)
            xf_rgb = self.neck(xf_rgb)##(bs,256,31,31)
            xf_t = self.neck(xf_t)
       # ger attention feature
        for i in range(len(zf_rgb)):
            zf_rgb[i], zf_t[i] = self.template_attention_block(zf_rgb[i], zf_t[i])
            union = torch.cat((xf_rgb[i], xf_t[i]), 1)
            xf_rgb[i], xf_t[i] = self.detection_attention_block(union)
        #fusion features (rgb and rgbt)
        features_fusion_z=list(range(3))
        features_fusion_x=list(range(3))
        for i in range(len(zf_rgb)):
            features_fusion_z[i]=self.feature_fusion(zf_rgb[i],zf_t[i])
            features_fusion_x[i]=self.feature_fusion(xf_rgb[i],xf_t[i])

  ##deep corr_realation
        features = self.xcorr_depthwise(features_fusion_x[0],features_fusion_z[0])
        for i in range(len(features_fusion_z)-1):
            features_new = self.xcorr_depthwise(features_fusion_x[i],features_fusion_z[i])
            features = torch.cat([features,features_new],1)##[bs,256x3,25,25]
        features = self.down(features)#[bs,256,25,25]

        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
