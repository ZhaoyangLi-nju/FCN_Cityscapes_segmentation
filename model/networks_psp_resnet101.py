import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torchvision.models.resnet import *


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def fix_grad(net):
    print(net.__class__.__name__)

    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)


def define_TrecgNet(cfg, using_semantic_branch=None, device=None):
    if using_semantic_branch is None:
        using_semantic_branch = cfg.USING_SEMANTIC_BRANCH

    if 'resnet' in cfg.ARCH:
        if cfg.MULTI_SCALE:
            # model = TRecgNet_Upsample_Resiual_Multiscale_Maxpool(cfg, encoder=cfg.ARCH,
            #                                              using_semantic_branch=using_semantic_branch, device=device)
            model = TRecgNet_Upsample_Resiual_Multiscale(cfg, encoder=cfg.ARCH,
                                                         using_semantic_branch=using_semantic_branch, device=device)
        elif cfg.MULTI_MODAL:
            model = TRecgNet_Upsample_Resiual_MultiModalTarget(cfg, encoder=cfg.ARCH,
                                                         using_semantic_branch=using_semantic_branch, device=device)
        else:
            model = TRecgNet_Upsample_Resiual(cfg, encoder=cfg.ARCH, using_semantic_branch=using_semantic_branch, device=device)

    return model


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]


def expand_Conv(module, in_channels):
    def expand_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.in_channels = in_channels
            m.out_channels = m.out_channels
            mean_weight = torch.mean(m.weight, dim=1, keepdim=True)
            m.weight.data = mean_weight.repeat(1, in_channels, 1, 1).data

    module.apply(expand_func)


##############################################################################
# Moduels
##############################################################################
class Upsample_Interpolate(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear',
                 reduce_dim=False):
        super(Upsample_Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        if reduce_dim:
            dim_out = int(dim_out / 2)
            self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=1, stride=1, padding=0, norm=norm)
            # self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_in, kernel_size=3, stride=1, padding=1, norm=norm)
        else:
            self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding,
                                                  norm=norm)
            # self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_out, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x, activate=True):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        x = self.conv_norm_relu1(x)
        # x = self.conv_norm_relu2(x)
        return x


class Upconv_ConvTransposed(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, output_padding=1):
        super(Upconv_ConvTransposed, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class UpBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm, using_semantic_branch=None):
        super(UpBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        self.using_semantic_branch = using_semantic_branch

    def forward(self, x):
        residual = x
        if self.using_semantic_branch is not None:
            x, conv_out = self.using_semantic_branch(x, activate=False)
            residual = conv_out

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class UpsampleBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear', upsample=True):
        super(UpsampleBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        if inplanes != planes:
            kernel_size, padding = 1, 0
        else:
            kernel_size, padding = 3, 1

        if upsample:

            self.upsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                norm(planes))
        else:
            self.upsample = None

        self.scale = scale
        self.mode = mode

    def forward(self, x):

        if self.upsample is not None:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
            residual = self.upsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return out

#########################################

##############################################################################
# Translate to recognize
##############################################################################
class Content_Model(nn.Module):

    def __init__(self, cfg, criterion=None):
        super(Content_Model, self).__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.net = cfg.WHICH_CONTENT_NET

        if 'resnet' in self.net:
            from .pretrained_resnet import ResNet
            self.model = ResNet(self.net, cfg)

        fix_grad(self.model)
        # print_network(self)

    def forward(self, x, target, in_channel=3, layers=None):

        # important when set content_model as the attr of trecg_net
        self.model.eval()

        layers = layers
        if layers is None or not layers:
            layers = self.cfg.CONTENT_LAYERS.split(',')

        input_features = self.model((x + 1) / 2, layers)
        target_targets = self.model((target + 1) / 2, layers)
        len_layers = len(layers)
        loss_fns = [self.criterion] * len_layers
        alpha = [1] * len_layers

        content_losses = [alpha[i] * loss_fns[i](gen_content, target_targets[i])
                          for i, gen_content in enumerate(input_features)]
        loss = sum(content_losses)
        return loss
class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class TRecgNet_Upsample_Resiual(nn.Module):

    def __init__(self, cfg, num_classes=19, encoder='resnet101', using_semantic_branch=True, device=None):
        super(TRecgNet_Upsample_Resiual, self).__init__()

        self.encoder = encoder
        self.cfg = cfg
        self.using_semantic_branch = using_semantic_branch
        # self.dim_noise = 128
        # self.use_noise = use_noise
        self.device = device
        # self.avg_pool_size = 7

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        fc_input_nc = dims[4] if encoder == 'resnet18' else dims[6]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet101'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet101 loaded....')
        else:
            resnet = resnet101(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
        m_out_sz = resnet.fc.in_features
        norm_layer = nn.BatchNorm2d
        

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )
        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz//2, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        if self.using_semantic_branch:
            self.build_upsample_content_layers(dims, num_classes)

        # self.score_main_256 = nn.Conv2d(1024, num_classes, 1)
        # self.score_main_128 = nn.Conv2d(512, num_classes, 1)
        # self.score_main_64 = nn.Conv2d(256, num_classes, 1)

        # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)

        if pretrained:
            # init_weights(self.head, 'normal')

            if using_semantic_branch:
                init_weights(self.lat1, 'normal')
                init_weights(self.lat2, 'normal')
                init_weights(self.lat3, 'normal')
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

                # init_weights(self.score_up_64, 'normal')
                # init_weights(self.score_up_128, 'normal')
                # init_weights(self.score_up_256, 'normal')

            init_weights(self.master_branch, 'normal')
            init_weights(self.auxiliary_branch, 'normal')
            # init_weights(self.head, 'normal')
            # init_weights(self.score_main_64, 'normal')
            # init_weights(self.score_main_128, 'normal')
            # init_weights(self.score_main_256, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims, num_classes):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = UpsampleBasicBlock(dims[4], dims[3], norm=norm)
        self.up2 = UpsampleBasicBlock(dims[3], dims[2], norm=norm)
        self.up3 = UpsampleBasicBlock(dims[2], dims[1], norm=norm)
        self.up4 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)

        self.lat1 = nn.Conv2d(dims[3], dims[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.lat2 = nn.Conv2d(dims[2], dims[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.lat3 = nn.Conv2d(dims[1], dims[1], kernel_size=1, stride=1, padding=0, bias=False)

        self.up_image_content = nn.Sequential(
            conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        # self.score_up_256 = nn.Sequential(
        #     nn.Conv2d(256, num_classes, 1)
        # )

        # self.score_up_128 = nn.Sequential(
        #     nn.Conv2d(128, num_classes, 1)
        # )
        # self.score_up_64 = nn.Sequential(
        #     nn.Conv2d(64, num_classes, 1)
        # )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_0 = self.relu(self.bn1(self.conv1(source)))
        if not self.using_semantic_branch:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        output=self.master_branch(layer_4)
        result['cls'] = F.interpolate(output, source.size()[2:], mode='bilinear')

        if self.using_semantic_branch and phase == 'train':
            # content model branch
            skip_1 = self.lat1(layer_3)
            skip_2 = self.lat2(layer_2)
            skip_3 = self.lat3(layer_1)

            up1 = self.up1(layer_4)
            up2 = self.up2(up1 + skip_1)
            up3 = self.up3(up2 + skip_2)
            up4 = self.up4(up3 + skip_3)

            result['gen_img'] = self.up_image_content(up4)

        # segmentation branch

        # if self.cfg.WHICH_SCORE == 'main':
        #     score_main_256 = self.score_main_256(layer_3)
        #     score_main_128 = self.score_main_128(layer_2)
        #     score_main_64 = self.score_main_64(layer_1)

        #     score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
        #     score = score + score_main_256
        #     score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
        #     score = score + score_main_128
        #     score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
        #     score = score + score_main_64
        # elif self.cfg.WHICH_SCORE == 'up':
        #     score_up_256 = self.score_up_256(up1)
        #     score_up_128 = self.score_up_128(up2)
        #     score_up_64 = self.score_up_64(up3)
        #     score = F.interpolate(score_512, layer_3.size()[2:], mode='bilinear', align_corners=True)
        #     score = score + score_up_256
        #     score = F.interpolate(score, layer_2.size()[2:], mode='bilinear', align_corners=True)
        #     score = score + score_up_128
        #     score = F.interpolate(score, layer_1.size()[2:], mode='bilinear', align_corners=True)
        #     score = score + score_up_64
        if phase=='train':
            aux = self.auxiliary_branch(layer_3)
            aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear')
            loss_classification = self.cls_criterion(result['cls'], label)
            loss_aux = self.cls_criterion(aux, label)
            result['loss_cls'] = loss_classification+0.4*loss_aux


        if self.using_semantic_branch and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        # if 'CLS' in self.cfg.LOSS_TYPES:
        if phase=='test':
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result