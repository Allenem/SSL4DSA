import torch
import torch.nn as nn
import numpy as np
import copy

import sys
sys.path.append('../../')

from networks.SwinUnet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.IMG_SIZE,
                                patch_size=config.PATCH_SIZE,
                                in_chans=config.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.EMBED_DIM,
                                depths=config.DEPTHS,
                                num_heads=config.NUM_HEADS,
                                window_size=config.WINDOW_SIZE,
                                mlp_ratio=config.MLP_RATIO,
                                qkv_bias=config.QKV_BIAS,
                                qk_scale=config.QK_SCALE,
                                drop_rate=config.DROP_RATE,
                                drop_path_rate=config.DROP_PATH_RATE,
                                ape=config.APE,
                                patch_norm=config.PATCH_NORM,
                                use_checkpoint=config.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.PRETRAIN_PATH
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


import ml_collections


img_size = 512

swin_config = ml_collections.ConfigDict({
    'IMG_SIZE': img_size,
    'PATCH_SIZE': 4,
    'IN_CHANS': 3,
    'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 6, 2],
    'NUM_HEADS': [3, 6, 12, 24],
    'WINDOW_SIZE': 8,
    'MLP_RATIO': 4.,
    'QKV_BIAS': True,
    'QK_SCALE': None,
    'DROP_RATE': 0.0,
    'DROP_PATH_RATE': 0.1,
    'APE': False,
    'PATCH_NORM': True,
    'USE_CHECKPOINT': False,
    'PRETRAIN_PATH': 'C:/Users/siat/Downloads/swin_tiny_patch4_window7_224.pth'
})


def get_swin_unet(img_size, out_ch):
    
    swin_config.IMG_SIZE = img_size

    model = SwinUnet(config=swin_config, img_size=img_size, num_classes=out_ch)

    return model


if __name__ == '__main__':

    model = SwinUnet(config=swin_config, img_size=img_size, num_classes=1)

    total_params = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total_params/1e6))
    
    res = model(torch.randn(8, 1, img_size, img_size))
    
    print(res.shape)