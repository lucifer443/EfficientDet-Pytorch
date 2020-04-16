import numpy as np
import torch.nn as nn
from mmcv.cnn import kaiming_init

from ..registry import HEADS
from ..utils import bias_init_with_prob, SeparableConv2d
from .anchor_head import AnchorHead


@HEADS.register_module
class RetinaSepConvHead(AnchorHead):
    """"RetinaHead with separate BN.

    In RetinaHead, conv/norm layers are shared across different FPN levels,
    while in RetinaSepBNHead, conv layers are shared across different FPN
    levels, but BN layers are separated.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaSepConvHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                SeparableConv2d(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    activation="Swish",
                    bias=True,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                SeparableConv2d(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    activation="Swish",
                    bias=True,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = SeparableConv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1,
            bias=True,
            norm_cfg=None)
        self.retina_reg = SeparableConv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1, bias=True, norm_cfg=None)

    def init_weights(self):
        for m in self.cls_convs:
            kaiming_init(m.depthwise, mode='fan_in')
            kaiming_init(m.pointwise.conv, mode='fan_in')
        for m in self.reg_convs:
            kaiming_init(m.depthwise, mode='fan_in')
            kaiming_init(m.pointwise.conv, mode='fan_in')
        bias_cls = bias_init_with_prob(0.01)
        kaiming_init(self.retina_cls.depthwise, mode='fan_in')
        kaiming_init(self.retina_cls.pointwise.conv, mode='fan_in', bias=bias_cls)
        kaiming_init(self.retina_reg.depthwise, mode='fan_in')
        kaiming_init(self.retina_reg.pointwise.conv, mode='fan_in')

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
