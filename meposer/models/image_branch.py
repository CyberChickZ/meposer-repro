import torch
from torch import nn
from torchvision.models import RegNet_Y_400MF_Weights, regnet_y_400mf

from ..smpl.constants import NUM_BODY_JOINTS
from .layers import conv_bn_act, mlp

IMAGE_MEAN, IMAGE_STD = 0.45, 0.225


def _conv_out(size, stride, times):
    for _ in range(times):
        size = (size + 2 - 3) // stride + 1
    return size


class StereoImageBranch(nn.Module):
    def __init__(self, cfg, heatmap_size, num_joints=NUM_BODY_JOINTS):
        super().__init__()
        assert cfg.backbone == "regnet_y_400mf", cfg.backbone
        net = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2 if cfg.pretrained else None)
        conv = net.stem[0]
        mono = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        mono.weight.data.copy_(conv.weight.data.sum(1, keepdim=True))
        net.stem[0] = mono
        self.stem = net.stem
        self.stages = nn.ModuleList([net.trunk_output.block1, net.trunk_output.block2, net.trunk_output.block3, net.trunk_output.block4])
        widths = [48, 104, 208, 440]
        d = cfg.feat_dim
        self.feature_fusion = cfg.get("feature_fusion", "fpn")
        assert self.feature_fusion in ("fpn", "single"), self.feature_fusion
        if self.feature_fusion == "fpn":
            self.lateral = nn.ModuleList([nn.Conv2d(w, 128, 1) for w in widths[1:]])
            self.fuse = conv_bn_act(128, d)
        else:
            self.fuse = conv_bn_act(widths[1], d, k=1)
        self.num_joints = num_joints
        self.heatmap_size = tuple(heatmap_size)
        self.heatmap_head = nn.Sequential(conv_bn_act(2 * d, 256), nn.Conv2d(256, 2 * num_joints, 1))
        w, h = self.heatmap_size
        self.heatmap_encoder_kind = cfg.get("heatmap_encoder", "conv")
        assert self.heatmap_encoder_kind in ("conv", "flatten"), self.heatmap_encoder_kind
        if self.heatmap_encoder_kind == "conv":
            enc_flat = 64 * _conv_out(h, 2, 2) * _conv_out(w, 2, 2)
            self.heatmap_encoder = nn.Sequential(
                nn.Conv2d(2 * num_joints, 64, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Flatten(), nn.Linear(enc_flat, cfg.heatmap_feat_dim), nn.ReLU(inplace=True),
            )
        else:
            self.heatmap_encoder = nn.Sequential(nn.Flatten(), mlp([2 * num_joints * h * w, cfg.heatmap_feat_dim, cfg.heatmap_feat_dim], final_act=True))
        self.local3d = mlp([cfg.heatmap_feat_dim, 256, num_joints * 3])
        self.out_dim = cfg.heatmap_feat_dim

    def features(self, x):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        c2, c3, c4 = outs[1:]
        if self.feature_fusion == "single":
            return self.fuse(c2)
        p = self.lateral[0](c2)
        p = p + nn.functional.interpolate(self.lateral[1](c3), size=p.shape[-2:], mode="bilinear", align_corners=False)
        p = p + nn.functional.interpolate(self.lateral[2](c4), size=p.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(p)

    def forward(self, images):
        b = images.shape[0]
        x = (images.float() / 255.0 - IMAGE_MEAN) / IMAGE_STD
        x = x.reshape(b * 2, 1, *x.shape[-2:])
        f = self.features(x)
        f = f.reshape(b, 2 * f.shape[1], *f.shape[-2:])
        logits = self.heatmap_head(f)
        if logits.shape[-2:] != self.heatmap_size[::-1]:
            logits = nn.functional.interpolate(logits, size=self.heatmap_size[::-1], mode="bilinear", align_corners=False)
        feat = self.heatmap_encoder(torch.sigmoid(logits))
        joints_local = self.local3d(feat).reshape(b, self.num_joints, 3)
        return {"heatmap_logits": logits, "image_feat": feat, "joints_local": joints_local}
