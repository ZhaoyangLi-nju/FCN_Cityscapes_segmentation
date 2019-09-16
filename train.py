import os
import random
from functools import reduce

import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader  # new add

import util.utils as util
from config.default_config import DefaultConfig
from config.resnet18_sunrgbd_config import RESNET18_SUNRGBD_CONFIG
from data import segmentation_dataset
from model.trecg_model import TRecgNet
from model.trecg_model_multimodal import TRecgNet_MULTIMODAL

cfg = DefaultConfig()
args = {
    'resnet18': RESNET18_SUNRGBD_CONFIG().args(),
}

# Setting random seed
if cfg.MANUAL_SEED is None:
    cfg.MANUAL_SEED = random.randint(1, 10000)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

# args for different backbones
cfg.parse(args['resnet18'])             

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS
device_ids = torch.cuda.device_count()
print('device_ids:', device_ids)
project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
util.mkdir('logs')

# resize_size = 256
# crop_size = 224
# image_h,image_w=416,544

train_transforms = list()
train_transforms.append(segmentation_dataset.Resize((384, 768)))
train_transforms.append(segmentation_dataset.ColorJitter(brightness = 0.5,contrast = 0.5,saturation = 0.5))
train_transforms.append(segmentation_dataset.RandomScale((1.0, 1.25, 1.5, 1.75, 2.0)))
train_transforms.append(segmentation_dataset.RandomCrop((384,768)))
train_transforms.append(segmentation_dataset.RandomHorizontalFlip())
# if cfg.MULTI_SCALE:
#     train_transforms.append(segmentation_dataset.MultiScale((cfg.FINE_SIZE, cfg.FINE_SIZE), scale_times=cfg.MULTI_SCALE_NUM)),
train_transforms.append(segmentation_dataset.ToTensor())
train_transforms.append(segmentation_dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

val_transforms = list()
# val_transforms.append(segmentation_dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
val_transforms.append(segmentation_dataset.Resize((384,768)))
# val_transforms.append(segmentation_dataset.CenterCrop((240,480)))
val_transforms.append(segmentation_dataset.ToTensor())
val_transforms.append(segmentation_dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

# train_data = segmentation_dataset.SUNRGBD(transform=transforms.Compose(train_transforms), phase_train=True,
#                                           data_dir='/data0/lzy/SUNRGBD')
# val_data = segmentation_dataset.SUNRGBD(transform=transforms.Compose(val_transforms), phase_train=False,
#                                         data_dir='data0/lzy/SUNRGBD')
train_data = segmentation_dataset.CityScapes(transform=transforms.Compose(train_transforms), phase_train=True,
                                          data_dir='/home/lzy/cityscapes')
val_data = segmentation_dataset.CityScapes(transform=transforms.Compose(val_transforms), phase_train=False,
                                        data_dir='/home/lzy/cityscapes')
train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.WORKERS, pin_memory=False)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False,
                              num_workers=cfg.WORKERS, pin_memory=False)
unlabeled_loader=None
# class weights
# num_classes_train = list(Counter([i[1] for i in train_loader.dataset.imgs]).values())
num_train = len(train_data)
num_val = len(val_data)

cfg.CLASS_WEIGHTS_TRAIN = torch.FloatTensor(num_train)

writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard

if cfg.MULTI_MODAL:
    model = TRecgNet_MULTIMODAL(cfg, writer=writer)
else:
    model = TRecgNet(cfg, writer=writer)
model.set_data_loader(train_loader, val_loader, unlabeled_loader,num_train,num_val)

def train():

    if cfg.RESUME:
        checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
        checkpoint = torch.load(checkpoint_path)
        load_epoch = checkpoint['epoch']
        model.load_checkpoint(model.net, checkpoint_path, checkpoint, data_para=True)
        cfg.START_EPOCH = load_epoch

        if cfg.INIT_EPOCH:
            # just load pretrained parameters
            print('load checkpoint from another source')
            cfg.START_EPOCH = 1

    print('>>> task path is {0}'.format(project_name))

    # train
    model.train_parameters(cfg)

    print('save model ...')
    model_filename = '{0}_{1}_{2}.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION, cfg.NITER_TOTAL)
    model.save_checkpoint(cfg.NITER_TOTAL, model_filename)

    if writer is not None:
        writer.close()

if __name__ == '__main__':
    train()
