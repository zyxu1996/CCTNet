import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# from tools.heatmap_fun import draw_features

from models.resT import rest_tiny, rest_small, rest_base, rest_large
from models.swinT import swin_tiny, swin_small, swin_base, swin_large
from models.volo import volo_d1, volo_d2, volo_d3, volo_d4, volo_d5
from models.cswin import cswin_tiny, cswin_base, cswin_small, cswin_large
from models.beit import beit_base, beit_large

from models.resnet import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b

from models.head import *


up_kwargs = {'mode': 'bilinear', 'align_corners': False}


class _CAFM(nn.Module):
    """Coupled attention fusion module"""
    def __init__(self, channels):
        super(_CAFM, self).__init__()
        self.conv_value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv_query = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_key = nn.Conv2d(channels, channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        value = self.conv_value(y)
        value = value.view(value.size(0), value.size(1), -1)

        query = self.conv_query(x)
        key = self.conv_key(y)
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)

        key_mean = key.mean(2).unsqueeze(2)
        query_mean = query.mean(2).unsqueeze(2)
        key -= key_mean
        query -= query_mean

        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = self.softmax(sim_map)
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        out_sim = out_sim.transpose(1, 2)
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.gamma * out_sim

        return out_sim


class CAFM(nn.Module):
    """Coupled attention fusion module"""
    def __init__(self, channels_trans, channels_cnn, channels_fuse):
        super(CAFM, self).__init__()
        self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, kernel_size=1)
        self.conv_cnn = nn.Conv2d(channels_cnn, channels_fuse, kernel_size=1)

        self._CAFM1 = _CAFM(channels=channels_fuse)
        self._CAFM2 = _CAFM(channels=channels_fuse)

        self.fuse_conv = nn.Conv2d(2 * channels_fuse, channels_fuse, kernel_size=1)

    def forward(self, x, y):
        """x:transformer, y:cnn"""
        x = self.conv_trans(x)
        y = self.conv_cnn(y)
        local2global = x + self._CAFM1(x, y).contiguous()
        global2local = y + self._CAFM2(y, x).contiguous()
        fuse = self.fuse_conv(torch.cat((local2global, global2local), dim=1))

        return fuse


class LAFM(nn.Module):
    def __init__(self, channels_trans, channels_cnn, channels_fuse, residual=False):
        super(LAFM, self).__init__()
        self.channels_fuse = channels_fuse
        self.residual = residual
        self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, kernel_size=1)
        self.conv_cnn = nn.Conv2d(channels_cnn, channels_fuse, kernel_size=1)
        self.conv_fuse = nn.Conv2d(2 * channels_fuse, 2 * channels_fuse, kernel_size=1)
        self.conv1 = nn.Conv2d(channels_fuse, channels_fuse, kernel_size=1)
        self.conv2 = nn.Conv2d(channels_fuse, channels_fuse, kernel_size=1)
        self.softmax = nn.Softmax(dim=0)
        if residual:
            self.conv = nn.Conv2d(channels_trans + channels_cnn, channels_fuse, kernel_size=1)

    def forward(self, x, y):
        """x:transformer, y:cnn"""
        if self.residual:
            residual = self.conv(torch.cat((x, y), dim=1))
        x = self.conv_trans(x)
        y = self.conv_cnn(y)
        x_ori, y_ori = x, y
        xy_fuse = self.conv_fuse(torch.cat((x, y), 1))
        xy_split = torch.split(xy_fuse, self.channels_fuse, dim=1)
        x = torch.sigmoid(self.conv1(xy_split[0]))
        y = torch.sigmoid(self.conv2(xy_split[1]))
        weights = self.softmax(torch.stack((x, y), 0))
        out = weights[0] * x_ori + weights[1] * y_ori
        if self.residual:
            out = out + residual

        return out


class CNN(nn.Module):
    def __init__(self, cnn_name, dilated=False, pretrained_base=True, **kwargs):
        super(CNN, self).__init__()
        self.pretrained = eval(cnn_name)(pretrained=pretrained_base, dilated=dilated, **kwargs)

    def forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        p1 = self.pretrained.layer1(x)
        p2 = self.pretrained.layer2(p1)
        p3 = self.pretrained.layer3(p2)
        p4 = self.pretrained.layer4(p3)

        return p1, p2, p3, p4


class CCTNet(nn.Module):

    def __init__(self, transformer_name, cnn_name, nclass, img_size, aux=False, pretrained=False, head='seghead', edge_aux=False):
        super(CCTNet, self).__init__()
        cnn_dict = {
            'resnet18_v1b': 'resnet18_v1b', 'resnet18': 'resnet18_v1b',
            'resnet34_v1b': 'resnet34_v1b', 'resnet34': 'resnet34_v1b',
            'resnet50_v1b': 'resnet50_v1b', 'resnet50': 'resnet50_v1b',
            'resnet101_v1b': 'resnet101_v1b', 'resnet101': 'resnet101_v1b',
        }
        self.aux = aux
        self.edge_aux = edge_aux
        self.head_name = head

        self.model = eval(transformer_name)(nclass=nclass, img_size=img_size, aux=aux, pretrained=pretrained)
        self.transformer_backbone = self.model.backbone
        head_dim = self.model.head_dim

        self.cnn_backbone = CNN(cnn_name=cnn_dict[cnn_name], dilated=False, pretrained_base=pretrained)
        if 'resnet18' in cnn_name or 'resnet34' in cnn_name:
            self.cnn_head_dim = [64, 128, 256, 512]
        if 'resnet50' in cnn_name or 'resnet101' in cnn_name:
            self.cnn_head_dim = [256, 512, 1024, 2048]

        self.fuse_dim = head_dim

        if self.head_name == 'apchead':
            self.decode_head = APCHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'aspphead':
            self.decode_head = ASPPHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'asppplushead':
            self.decode_head = ASPPPlusHead(in_channels=head_dim[3], num_classes=nclass, in_index=[0, 3])

        if self.head_name == 'dahead':
            self.decode_head = DAHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'dnlhead':
            self.decode_head = DNLHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'fcfpnhead':
            self.decode_head = FCFPNHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.head_name == 'cefpnhead':
            self.decode_head = CEFPNHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.head_name == 'fcnhead':
            self.decode_head = FCNHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'gchead':
            self.decode_head = GCHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'psahead':
            self.decode_head = PSAHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'psphead':
            self.decode_head = PSPHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'seghead':
            self.decode_head = SegHead(in_channels=self.fuse_dim, num_classes=nclass, in_index=[0, 1, 2, 3])
            self.decode_head_cnn = SegHead(in_channels=self.cnn_head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])
            self.decode_head_trans = SegHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        if self.head_name == 'unethead':
            self.decode_head = UNetHead(in_channels=self.fuse_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        if self.head_name == 'uperhead':
            self.decode_head = UPerHead(in_channels=self.fuse_dim, num_classes=nclass)

        if self.head_name == 'annhead':
            self.decode_head = ANNHead(in_channels=head_dim[2:], num_classes=nclass, in_index=[2, 3], channels=512)

        if self.head_name == 'mlphead':
            self.decode_head = MLPHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.aux:
            self.auxiliary_head = FCNHead(num_convs=1, in_channels=self.fuse_dim[2], num_classes=nclass, in_index=2, channels=256)

        if self.edge_aux:
            self.edge_head = EdgeHead(in_channels=head_dim[0:2], in_index=[0, 1], channels=self.fuse_dim[0])

        self.fuse1 = LAFM(head_dim[0], self.cnn_head_dim[0], self.fuse_dim[0], residual=True)
        self.fuse2 = LAFM(head_dim[1], self.cnn_head_dim[1], self.fuse_dim[1], residual=True)
        # self.fuse3 = LAFM(head_dim[2], self.cnn_head_dim[2], self.fuse_dim[2], residual=True)
        # self.fuse4 = LAFM(head_dim[3], self.cnn_head_dim[3], self.fuse_dim[3], residual=True)

        # self.fuse1 = CAFM(channels_trans=head_dim[0], channels_cnn=self.cnn_head_dim[0], channels_fuse=self.fuse_dim[0])
        # self.fuse2 = CAFM(channels_trans=head_dim[1], channels_cnn=self.cnn_head_dim[1], channels_fuse=self.fuse_dim[1])
        self.fuse3 = CAFM(channels_trans=head_dim[2], channels_cnn=self.cnn_head_dim[2], channels_fuse=self.fuse_dim[2])
        self.fuse4 = CAFM(channels_trans=head_dim[3], channels_cnn=self.cnn_head_dim[3], channels_fuse=self.fuse_dim[3])

    def forward(self, x):
        size = x.size()[2:]
        outputs = []

        out_backbone = self.transformer_backbone(x)
        c1, c2, c3, c4 = out_backbone[:4]
        p1, p2, p3, p4 = self.cnn_backbone(x)

        x_cp1 = self.fuse1(c1, p1)
        x_cp2 = self.fuse2(c2, p2)
        x_cp3 = self.fuse3(c3, p3)
        x_cp4 = self.fuse4(c4, p4)

        out_backbone = [x_cp1, x_cp2, x_cp3, x_cp4]
        out_backbone_trans = [c1, c2, c3, c4]
        out_backbone_cnn = [p1, p2, p3, p4]

        # for i, out in enumerate(out_backbone):
        #     draw_features(out, f'C{i}')

        x0 = self.decode_head(out_backbone)
        x1 = self.decode_head_cnn(out_backbone_cnn)
        x2 = self.decode_head_trans(out_backbone_trans)

        x0 = [x0, x1, x2]
        if isinstance(x0, (list, tuple)):
            for out in x0:
                out = F.interpolate(out, size, **up_kwargs)
                outputs.append(out)
        else:
            x0 = F.interpolate(x0, size, **up_kwargs)
            outputs.append(x0)

        if self.aux:
            x1 = self.auxiliary_head(out_backbone)
            x1 = F.interpolate(x1, size, **up_kwargs)
            outputs.append(x1)

        if self.edge_aux:
            edge = self.edge_head(out_backbone)
            edge = F.interpolate(edge, size, **up_kwargs)
            outputs.append(edge)

        return outputs


if __name__ == '__main__':
    """Notice if torch1.6, try to replace a / b with torch.true_divide(a, b)"""
    from tools.flops_params_fps_count import flops_params_fps

    model = CCTNet(transformer_name='cswin_tiny', cnn_name='resnet50', nclass=6, img_size=512, aux=True,
                   edge_aux=False, head='seghead', pretrained=False)
    flops_params_fps(model)
