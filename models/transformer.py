import torch
import torch.nn as nn
import torch.nn.functional as F
from models.head import *
from models.resT import rest_tiny, rest_small, rest_base, rest_large
from models.swinT import swin_tiny, swin_small, swin_base, swin_large
from models.volo import volo_d1, volo_d2, volo_d3, volo_d4, volo_d5
from models.cswin import cswin_tiny, cswin_base, cswin_small, cswin_large
from models.beit import beit_base, beit_large
#from tools.heatmap_fun import draw_features

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


class Transformer(nn.Module):

    def __init__(self, transformer_name, nclass, img_size, aux=False, pretrained=False, head='seghead', edge_aux=False):
        super(Transformer, self).__init__()
        self.aux = aux
        self.edge_aux = edge_aux
        self.head_name = head

        self.model = eval(transformer_name)(nclass=nclass, img_size=img_size, aux=aux, pretrained=pretrained)
        self.backbone = self.model.backbone

        head_dim = self.model.head_dim
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
            self.decode_head = SegHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        if self.head_name == 'unethead':
            self.decode_head = UNetHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        if self.head_name == 'uperhead':
            self.decode_head = UPerHead(in_channels=head_dim, num_classes=nclass)

        if self.head_name == 'annhead':
            self.decode_head = ANNHead(in_channels=head_dim[2:], num_classes=nclass, in_index=[2, 3], channels=512)

        if self.head_name == 'mlphead':
            self.decode_head = MLPHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.aux:
            self.auxiliary_head = FCNHead(num_convs=1, in_channels=head_dim[2], num_classes=nclass, in_index=2, channels=256)

        if self.edge_aux:
            self.edge_head = EdgeHead(in_channels=head_dim[0:2], in_index=[0, 1], channels=head_dim[0])

    def forward(self, x):
        size = x.size()[2:]
        outputs = []

        out_backbone = self.backbone(x)

        # for i, out in enumerate(out_backbone):
        #     draw_features(out, f'C{i}')

        x0 = self.decode_head(out_backbone)
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

    model = Transformer(transformer_name='cswin_tiny', nclass=6, img_size=512, aux=True, edge_aux=False,
                        head='uperhead', pretrained=False)
    flops_params_fps(model)
