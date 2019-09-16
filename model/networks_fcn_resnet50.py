import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torchvision.models.resnet import resnet18


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


class TRecgNet_Upsample_Resiual(nn.Module):

    def __init__(self, cfg, num_classes=19, encoder='resnet50', using_semantic_branch=True, device=None):
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
            resnet = models.__dict__['resnet50'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet50_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet50 loaded....')
        else:
            resnet = resnet50(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(2048, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.using_semantic_branch:
            self.build_upsample_content_layers(dims, num_classes)

        self.score_main_256 = nn.Conv2d(1024, num_classes, 1)
        self.score_main_128 = nn.Conv2d(512, num_classes, 1)
        self.score_main_64 = nn.Conv2d(256, num_classes, 1)

        # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)

        if pretrained:
            init_weights(self.head, 'normal')

            if using_semantic_branch:
                init_weights(self.lat1, 'normal')
                init_weights(self.lat2, 'normal')
                init_weights(self.lat3, 'normal')
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

                init_weights(self.score_up_64, 'normal')
                init_weights(self.score_up_128, 'normal')
                init_weights(self.score_up_256, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_main_64, 'normal')
            init_weights(self.score_main_128, 'normal')
            init_weights(self.score_main_256, 'normal')

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

        self.score_up_256 = nn.Sequential(
            nn.Conv2d(256, num_classes, 1)
        )

        self.score_up_128 = nn.Sequential(
            nn.Conv2d(128, num_classes, 1)
        )
        self.score_up_64 = nn.Sequential(
            nn.Conv2d(64, num_classes, 1)
        )

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

        if self.using_semantic_branch:
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
        score_512 = self.head(layer_4)

        if self.cfg.WHICH_SCORE == 'main':
            score_main_256 = self.score_main_256(layer_3)
            score_main_128 = self.score_main_128(layer_2)
            score_main_64 = self.score_main_64(layer_1)

            score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_256
            score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_128
            score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_64
        elif self.cfg.WHICH_SCORE == 'up':
            score_up_256 = self.score_up_256(up1)
            score_up_128 = self.score_up_128(up2)
            score_up_64 = self.score_up_64(up3)
            score = F.interpolate(score_512, layer_3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_256
            score = F.interpolate(score, layer_2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_128
            score = F.interpolate(score, layer_1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_64
        elif self.cfg.WHICH_SCORE == 'both':
            score_main_256 = self.score_main_256(layer_3)
            score_main_128 = self.score_main_128(layer_2)
            score_main_64 = self.score_main_64(layer_1)
            score_up_256 = self.score_up_256(up1)
            score_up_128 = self.score_up_128(up2)
            score_up_64 = self.score_up_64(up3)
            
            score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_256 + score_main_256
            score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_128 + score_main_128
            score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_64 + score_main_64

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if self.using_semantic_branch and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result

class TRecgNet_Upsample_Resiual_MultiModalTarget(nn.Module):

    def __init__(self, cfg, num_classes=37, encoder='resnet18', using_semantic_branch=True, device=None):
        super(TRecgNet_Upsample_Resiual_MultiModalTarget, self).__init__()

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
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.using_semantic_branch:
            self.build_upsample_content_layers(dims, num_classes)

        self.score_main_256 = nn.Conv2d(256, num_classes, 1)
        self.score_main_128 = nn.Conv2d(128, num_classes, 1)
        self.score_main_64 = nn.Conv2d(64, num_classes, 1)

        # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)

        if pretrained:
            init_weights(self.head, 'normal')

            if using_semantic_branch:
                init_weights(self.lat1, 'normal')
                init_weights(self.lat2, 'normal')
                init_weights(self.lat3, 'normal')
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

                init_weights(self.score_up_64, 'normal')
                init_weights(self.score_up_128, 'normal')
                init_weights(self.score_up_256, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_main_64, 'normal')
            init_weights(self.score_main_128, 'normal')
            init_weights(self.score_main_256, 'normal')

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

        self.up_depth = nn.Sequential(
            conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_seg = nn.Sequential(
            conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.score_up_256 = nn.Sequential(
            nn.Conv2d(dims[3], num_classes, 1)
        )

        self.score_up_128 = nn.Sequential(
            nn.Conv2d(dims[2], num_classes, 1)
        )
        self.score_up_64 = nn.Sequential(
            nn.Conv2d(dims[1], num_classes, 1)
        )

    def forward(self, source=None, target_1=None, target_2=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.using_semantic_branch:
            # content model branch
            skip_1 = self.lat1(layer_3)
            skip_2 = self.lat2(layer_2)
            skip_3 = self.lat3(layer_1)

            up1 = self.up1(layer_4)
            up2 = self.up2(up1 + skip_1)
            up3 = self.up3(up2 + skip_2)
            up4 = self.up4(up3 + skip_3)

            result['gen_depth'] = self.up_depth(up4)
            result['gen_seg'] = self.up_seg(up4)

        # segmentation branch
        score_512 = self.head(layer_4)

        if self.cfg.WHICH_SCORE == 'main':
            score_main_256 = self.score_main_256(layer_3)
            score_main_128 = self.score_main_128(layer_2)
            score_main_64 = self.score_main_64(layer_1)

            score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_256
            score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_128
            score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_64
        elif self.cfg.WHICH_SCORE == 'up':
            score_up_256 = self.score_up_256(up1)
            score_up_128 = self.score_up_128(up2)
            score_up_64 = self.score_up_64(up3)
            score = F.interpolate(score_512, layer_3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_256
            score = F.interpolate(score, layer_2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_128
            score = F.interpolate(score, layer_1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_64
        elif self.cfg.WHICH_SCORE == 'both':
            score_main_256 = self.score_main_256(layer_3)
            score_main_128 = self.score_main_128(layer_2)
            score_main_64 = self.score_main_64(layer_1)
            score_up_256 = self.score_up_256(up1)
            score_up_128 = self.score_up_128(up2)
            score_up_64 = self.score_up_64(up3)
            score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_256 + score_main_256
            score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_128 + score_main_128
            score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_64 + score_main_64

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if self.using_semantic_branch and phase == 'train':
            result['loss_content_depth'] = self.content_model(result['gen_depth'], target_1, layers=content_layers)
            result['loss_content_seg'] = self.content_model(result['gen_seg'], target_2, layers=content_layers)
            # result['loss_content'] = loss_content_1 + loss_content_2

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class TRecgNet_Upsample_Resiual_Multiscale(nn.Module):

    def __init__(self, cfg, num_classes=37, encoder='resnet18', using_semantic_branch=True, device=None):
        super(TRecgNet_Upsample_Resiual_Multiscale, self).__init__()

        self.encoder = encoder
        self.cfg = cfg
        self.using_semantic_branch = using_semantic_branch
        self.device = device

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)

        self.score_main_256 = nn.Conv2d(256, num_classes, 1)
        self.score_main_128 = nn.Conv2d(128, num_classes, 1)
        self.score_main_64 = nn.Conv2d(64, num_classes, 1)

        if self.using_semantic_branch:
            self.build_upsample_content_layers(dims, num_classes)

        if pretrained:
            if using_semantic_branch:
                # init_weights(self.up_image_14, 'normal')
                init_weights(self.up_image_28, 'normal')
                init_weights(self.up_image_56, 'normal')
                init_weights(self.up_image_112, 'normal')
                init_weights(self.up_image_224, 'normal')
                init_weights(self.lat1, 'normal')
                init_weights(self.lat2, 'normal')
                init_weights(self.lat3, 'normal')
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')
                init_weights(self.score_up_64, 'normal')
                init_weights(self.score_up_128, 'normal')
                init_weights(self.score_up_256, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_main_64, 'normal')
            init_weights(self.score_main_128, 'normal')
            init_weights(self.score_main_256, 'normal')

        elif not pretrained:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims, num_classes):

        # norm = nn.InstanceNorm2d
        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = UpsampleBasicBlock(dims[4], dims[1], norm=norm)
        self.up2 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)
        self.up3 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)
        self.up4 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)

        self.lat1 = nn.Conv2d(dims[3], dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.lat2 = nn.Conv2d(dims[2], dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.lat3 = nn.Conv2d(dims[1], dims[1], kernel_size=1, stride=1, padding=0, bias=False)

        # self.up_image_14 = nn.Sequential(
        #     conv_norm_relu(dims[1], dims[1], norm=norm),
        #     nn.Conv2d(dims[1], 3, 3, 1, 1, bias=False),
        #     nn.Tanh()
        # )

        self.up_image_28 = nn.Sequential(
            conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_56 = nn.Sequential(
            conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_112 = nn.Sequential(
            conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_224 = nn.Sequential(
            conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        # segmentation
        self.score_up_256 = nn.Sequential(
            nn.Conv2d(64, num_classes, 1)
        )
        self.score_up_128 = nn.Sequential(
            nn.Conv2d(64, num_classes, 1)
        )
        self.score_up_64 = nn.Sequential(
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        # content model branch
        if self.using_semantic_branch:

            scale_times = self.cfg.MULTI_SCALE_NUM
            ms_compare = []

            skip_1 = self.lat1(layer_3)
            skip_2 = self.lat2(layer_2)
            skip_3 = self.lat3(layer_1)

            up1 = self.up1(layer_4)
            up2 = self.up2(up1 + skip_1)
            up3 = self.up3(up2 + skip_2)
            up4 = self.up4(up3 + skip_3)

            compare_28 = self.up_image_28(up1)
            compare_56 = self.up_image_56(up2)
            compare_112 = self.up_image_112(up3)
            compare_224 = self.up_image_224(up4)

            ms_compare.append(compare_224)
            ms_compare.append(compare_112)
            ms_compare.append(compare_56)
            ms_compare.append(compare_28)
            # ms_compare.append(compare_14)
            # ms_compare.append(compare_7)

            result['gen_img'] = ms_compare[:scale_times]

        # segmentation branch
        score_512 = self.head(layer_4)

        if self.cfg.WHICH_SCORE == 'main':
            score_main_256 = self.score_main_256(layer_3)
            score_main_128 = self.score_main_128(layer_2)
            score_main_64 = self.score_main_64(layer_1)

            score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_256
            score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_128
            score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_main_64
        elif self.cfg.WHICH_SCORE == 'up':
            score_up_256 = self.score_up_256(up1)
            score_up_128 = self.score_up_128(up2)
            score_up_64 = self.score_up_64(up3)
            score = F.interpolate(score_512, layer_3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_256
            score = F.interpolate(score, layer_2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_128
            score = F.interpolate(score, layer_1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_64
        elif self.cfg.WHICH_SCORE == 'both':
            score_main_256 = self.score_main_256(layer_3)
            score_main_128 = self.score_main_128(layer_2)
            score_main_64 = self.score_main_64(layer_1)
            score_up_256 = self.score_up_256(up1)
            score_up_128 = self.score_up_128(up2)
            score_up_64 = self.score_up_64(up3)
            score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_256 + score_main_256
            score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_128 + score_main_128
            score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_up_64 + score_main_64

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if self.using_semantic_branch and phase == 'train':
            scale_times = self.cfg.MULTI_SCALE_NUM
            content_loss_list = []
            for i, (gen, _target) in enumerate(zip(result['gen_img'], target)):
                assert (gen.size()[-1] == _target.size()[-1])
                # content_layers = [str(layer) for layer in range(5 - i)]
                content_loss_list.append(self.content_model(gen, _target, layers=content_layers))

            loss_coef = [1] * scale_times
            ms_losses = [loss_coef[i] * loss for i, loss in enumerate(content_loss_list)]
            result['loss_content'] = sum(ms_losses)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class TRecgNet_Upsample_Resiual_Multiscale_Maxpool(nn.Module):

    def __init__(self, cfg, num_classes=37, encoder='resnet18', using_semantic_branch=True, device=None):
        super(TRecgNet_Upsample_Resiual_Multiscale_Maxpool, self).__init__()

        self.encoder = encoder
        self.cfg = cfg
        self.using_semantic_branch = using_semantic_branch
        # self.dim_noise = 128
        # self.use_noise = use_noise
        self.device = device
        # self.avg_pool_size = 7

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        # self.head = nn.Conv2d(512, num_classes, 1)
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)

        if self.using_semantic_branch:
            self.build_upsample_content_layers(dims, num_classes)

        self.score_main_256 = nn.Conv2d(256, num_classes, 1)
        self.score_main_128 = nn.Conv2d(128, num_classes, 1)
        self.score_main_64 = nn.Conv2d(64, num_classes, 1)

        if pretrained:
            if using_semantic_branch:
                init_weights(self.up_image_14, 'normal')
                init_weights(self.up_image_28, 'normal')
                init_weights(self.up_image_56, 'normal')
                init_weights(self.up_image_112, 'normal')
                init_weights(self.up_image_224, 'normal')
                init_weights(self.lat1, 'normal')
                init_weights(self.lat2, 'normal')
                init_weights(self.lat3, 'normal')
                init_weights(self.lat4, 'normal')
                init_weights(self.top_layer, 'normal')
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')
                init_weights(self.up5, 'normal')

                init_weights(self.score_up_64, 'normal')
                init_weights(self.score_up_128, 'normal')
                init_weights(self.score_up_256, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_main_64, 'normal')
            init_weights(self.score_main_128, 'normal')
            init_weights(self.score_main_256, 'normal')

        elif not pretrained:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims, num_classes):

        # norm = nn.InstanceNorm2d
        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.top_layer = conv_norm_relu(dims[4], dims[1], kernel_size=1, padding=0, norm=norm)
        self.up1 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)
        self.up2 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)
        self.up3 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)
        self.up4 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)
        self.up5 = UpsampleBasicBlock(dims[1], dims[1], norm=norm)

        self.lat1 = nn.Conv2d(dims[3], dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.lat2 = nn.Conv2d(dims[2], dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.lat3 = nn.Conv2d(dims[1], dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.lat4 = nn.Conv2d(dims[1], dims[1], kernel_size=1, stride=1, padding=0, bias=False)

        self.up_image_14 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.up_image_28 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_image_56 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_image_112 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_224 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        # segmentation
        self.score_up_256 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(64, num_classes, 1)
        )
        self.score_up_128 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(64, num_classes, 1)
        )
        # self.score_up_64 = nn.Sequential(
        #     # conv_norm_relu(dims[1], dims[1], norm=norm),
        #     nn.Conv2d(64, num_classes, 1)
        # )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        out = {}
        # out['0'] = self.relu(self.bn1(self.conv1(source)))
        # if not self.using_semantic_branch:
        out['0'] = self.relu(self.bn1(self.conv1(source)))
        maxpool = self.maxpool(out['0'])
        out['1'] = self.layer1(maxpool)
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])

        # content model branch
        if self.using_semantic_branch:

            scale_times = self.cfg.MULTI_SCALE_NUM
            ms_compare = []

            skip_1 = self.lat1(out['3'])
            skip_2 = self.lat2(out['2'])
            skip_3 = self.lat3(out['1'])
            skip_4 = self.lat4(out['0'])

            top = self.top_layer(out['4'])
            up1 = self.up1(top)
            up2 = self.up2(up1 + skip_1)
            up3 = self.up3(up2 + skip_2)
            up4 = self.up4(up3 + skip_3)
            up5 = self.up5(up4 + skip_4)

            compare_14 = self.up_image_14(up1)
            compare_28 = self.up_image_28(up2)
            compare_56 = self.up_image_56(up3)
            compare_112 = self.up_image_112(up4)
            compare_224 = self.up_image_224(up5)

            ms_compare.append(compare_224)
            ms_compare.append(compare_112)
            ms_compare.append(compare_56)
            ms_compare.append(compare_28)
            ms_compare.append(compare_14)
            # ms_compare.append(compare_7)

            out['gen_img'] = ms_compare[:scale_times]

            # segmentation branch

            score_up_256 = self.score_up_256(up1)
            score_up_128 = self.score_up_128(up2)
            # score_up_64 = self.score_up_64(up3)

        score_main_256 = self.score_main_256(out['3'])
        score_main_128 = self.score_main_128(out['2'])
        score_main_64 = self.score_main_64(out['1'])

        score_512 = self.head(out['4'])

        score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_main_256
        score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_main_128
        score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_main_64


        # score = F.interpolate(score_512, out['3'].size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_up_256
        # score = F.interpolate(score, out['2'].size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_up_128
        # score = F.interpolate(score, out['1'].size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_up_64

        # score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_up_256 + score_main_256
        # score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_up_128 + score_main_128
        # score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_up_64 + score_main_64

        # score = F.interpolate(score_512, score_main_256.size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_main_256
        # score = F.interpolate(score, score_main_128.size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_main_128
        # score = F.interpolate(score, score_main_64.size()[2:], mode='bilinear', align_corners=True)
        # score = score + score_main_64

        out['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        loss_content = None
        loss_cls = None
        loss_pix2pix = None

        if self.using_semantic_branch and phase == 'train':
            scale_times = self.cfg.MULTI_SCALE_NUM
            content_loss_list = []
            for i, (gen, _target) in enumerate(zip(out['gen_img'], target)):
                assert (gen.size()[-1] == _target.size()[-1])
                # content_layers = [str(layer) for layer in range(scale_times - i)]
                content_loss_list.append(self.content_model(gen, _target, layers=content_layers))

            loss_coef = [1] * scale_times
            ms_losses = [loss_coef[i] * loss for i, loss in enumerate(content_loss_list)]
            loss_content = sum(ms_losses)

        if 'CLS' in self.cfg.LOSS_TYPES:
            loss_cls = self.cls_criterion(out['cls'], label)

        result = []
        for key in out_keys:

            if isinstance(key, list):
                item = [out[subkey] for subkey in key]
            else:
                item = out[key]
            result.append(item)

        return result, {'cls_loss': loss_cls, 'content_loss': loss_content, 'pix2pix_loss': loss_pix2pix}

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)