# Parts of this code come from https://github.com/STVIR/pysot
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from snot.core.config import cfg
from snot.models.backbone import get_backbone
from snot.models.head import get_rpn_head, get_mask_head, get_refine_head
from snot.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS) #RPN++中间用到三层特征层，三层特征的特征数不一样，adjustlayer用来统一

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z) #SiamRPN++提取特征时用到了三个阶段的特征层，[1,44,15,15] [1,134,15,15] [1,448,15,15]，分别进行RPN中互卷积
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf) #三个特征都被调整为[1,256,7,7]，center_size参数根据论文设置是7，也就是取特征图的[1,256,4：11,4:11]
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x) #[1,3,256,256]->[1,44,31,31] [1,134,31,31] [1,448,31,31]，分别进行RPN中互卷积
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf) #这里搜索域的特征在neck里不像模板域会被抽取中心的7*7，在adjust模块里判断阈值为20，31>20,15<20, 所以模板域被抽取7*7，搜索域没有。全被调整为[1,256,31,31]
        cls, loc = self.rpn_head(self.zf, xf) # 输入：3*[1，256，31，31] & 3*[1,256,7,7]   输出：1* [1,2*anchornum,25,25] & [1,4*anchornum,25,25]
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)
