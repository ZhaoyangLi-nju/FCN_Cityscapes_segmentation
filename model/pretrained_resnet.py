import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):

    def __init__(self, resnet=None, cfg=None):
        super(ResNet, self).__init__()

        if resnet == 'resnet18':

            if cfg.CONTENT_PRETRAINED == 'place':
                resnet_model = models.__dict__['resnet18'](num_classes=365)
                load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
                checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                resnet_model.load_state_dict(state_dict)
                print('content model pretrained using place')
            else:
                resnet_model = models.resnet18(True)
                print('content model pretrained using imagenet')

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def forward(self, x, out_keys, in_channel=3):

        out = {}
        out['0'] = self.relu(self.bn1(self.conv1(x)))
        out['1'] = self.layer1(self.maxpool(out['0']))
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])
        return [out[key] for key in out_keys]

class VGG_bn(nn.Module):
    def __init__(self, arch):
        super(VGG_bn, self).__init__()

        if 'vgg11' in arch:
            features = models.vgg11_bn(pretrained=True).features
            self.layer0 = nn.Sequential(*list(features)[:3])
            self.layer1 = nn.Sequential(*list(features)[3:7])
            self.layer2 = nn.Sequential(*list(features)[7:14])
            self.layer3 = nn.Sequential(*list(features)[14:21])
            self.layer4 = nn.Sequential(*list(features)[21:28])

        elif 'vgg19' in arch:
            features = models.vgg19_bn(pretrained=True).features
            self.layer0 = nn.Sequential(*list(features)[:6])
            self.layer1 = nn.Sequential(*list(features)[6:13])
            self.layer2 = nn.Sequential(*list(features)[13:26])
            self.layer3 = nn.Sequential(*list(features)[26:39])
            self.layer4 = nn.Sequential(*list(features)[39:52])

    def forward(self, x, out_keys, in_channel=3):
        out = {}
        out['0'] = self.layer0(x)
        out['1'] = self.layer1(out['0'])
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])
        return [out[key] for key in out_keys]
class VGG(nn.Module):
    def __init__(self, arch):
        super(VGG, self).__init__()

        if 'vgg16' in arch:
            features = models.vgg16(pretrained=True).features
            self.layer0 = nn.Sequential(*list(features)[:5])
            self.layer1 = nn.Sequential(*list(features)[5:10])
            self.layer2 = nn.Sequential(*list(features)[10:17])
            self.layer3 = nn.Sequential(*list(features)[17:24])
            self.layer4 = nn.Sequential(*list(features)[24:])


    def forward(self, x, out_keys, in_channel=3):
        out = {}
        out['0'] = self.layer0(x)
        out['1'] = self.layer1(out['0'])
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])
        return [out[key] for key in out_keys]