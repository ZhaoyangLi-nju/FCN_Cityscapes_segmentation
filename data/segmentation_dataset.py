import os
import random

import h5py
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numbers
from util.utils import color_label_np
import PIL.ImageEnhance as ImageEnhance
# image_h, image_w=416,544

"""
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),

"""
img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'
label_dir_train_file = './data/label_train.txt'
img_dir_test_file = './data/img_dir_test.txt'
depth_dir_test_file = './data/depth_dir_test.txt'
label_dir_test_file = './data/label_test.txt'

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
def convert(label):
    # size = label.size
    label_copy = label.copy()
    # print(map.size)
    ids = {-1:19, 0: 19, 1:19, 2:19 ,3:19, 4:19, 5:19, 6:19,7:0, 8:1, 9:19, 10: 19, 11:2, 12: 3, 13:4,
    14:19, 15:19, 16:19, 17: 5,18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28:15, 29:19, 30:19, 31: 16, 32:17, 33: 18}
    # ids = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    for K, V in ids.items():
        label_copy[label == K] = V
    return label_copy
class CityScapes(torch.utils.data.Dataset):
    def __init__(self,transform=None,phase_train=True,data_dir=None):
        self.transform=transform
        if phase_train:
            trainFlag="train"
            file_dir=train_dirs
        else:
            trainFlag="val"
            file_dir=val_dirs

        self.img_dir = data_dir + "/leftImg8bit/"+trainFlag+"/"
        self.label_dir = data_dir + "/gtFine/"+trainFlag+"/"



        self.examples = []
        for train_dir in file_dir:
            train_img_dir_path = self.img_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = self.label_dir +train_dir+'/'+img_id + "_gtFine_labelIds.png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __len__(self):
        return self.num_examples
    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        # img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
        image = Image.open(img_path).convert('RGB')
        # image = F.resize(image, (320, 640))

        # resize img without interpolation (want the image to still match
        label_img_path = example["label_img_path"]
        label_img = Image.open(label_img_path).convert('L') # (shape: (1024, 2048))
        # label_img = F.resize(label_img, (320, 640), interpolation=Image.NEAREST)

        label_img = convert(np.array(label_img))

        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        seg = Image.fromarray((color_label_np(label_img).astype(np.uint8)), mode='RGB')

        label = Image.fromarray(label_img.astype(np.uint8))
        # depth = Image.open(depth_dir[idx]).convert('RGB')
        depth=None

        sample = {'image': image, 'depth': depth, 'label': label, 'seg': seg}

        # sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class SUNRGBD(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):

        self.phase_train = phase_train
        self.transform = transform
        self.id_to_trainid = {-1: 255, 0: 255, 1: 0, 2: 1,
                              3: 2, 4: 3, 5: 4, 6: 5,
                              7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12,
                              14: 13, 15: 14, 16: 15, 17: 16,
                              18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26,
                              28: 27, 29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36}

        try:
            with open(img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            with open(img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            with open(depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            with open(label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
        except:
            if data_dir is None:
                data_dir = '/path/to/SUNRGB-D'
            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
                depth_bfx_path = os.path.join(real_dir, 'hha/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path)
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path)
                    self.label_dir_train = np.append(self.label_dir_train, label_path)
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path)
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path)
                    self.label_dir_test = np.append(self.label_dir_test, label_path)

            local_file_dir = '/'.join(img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test

        # label = np.load(label_dir[idx])
        image = Image.open(img_dir[idx]).convert('RGB')
        # image_np = np.asarray(image)

        # depth = Image.fromarray(np.uint8(color_label_np(np.load(label_dir[idx]))), mode='RGB')
        # depth = Image.fromarray(color_label_np(np.uint8(np.load(label_dir[idx]))), mode='RGB')
        # depth = Image.open(depth_dir[idx]).convert('RGB')

        # label = np.uint8(np.maximum(_label, 0))
        # label = Image.fromarray(label)
        # label = Image.fromarray(np.uint8(np.load(label_dir[idx])))
        # label_np = np.asarray(label)

        _label = np.load(label_dir[idx])
        _label_copy = _label.copy()
        for k, v in self.id_to_trainid.items():
            _label_copy[_label == k] = v

        label = Image.fromarray(_label_copy.astype(np.uint8))
        depth = Image.open(depth_dir[idx]).convert('RGB')
        seg = Image.fromarray((color_label_np(_label_copy).astype(np.uint8)), mode='RGB')

        # # if len(label)==530:
        #     f = open('/home/lzy/ResNet_Backbone_segmentation/result','w')
        #     for i in range(len(label)):
        #         for j in range(len(label[0])):
        #             f.write(str(int(label[i][j]))+' ')
        #         f.write('\r')
        #     f.close()
        #     for i in range(1000000):
        #         print(i)
        # cv2.imshow('label',label)


        sample = {'image': image, 'depth': depth, 'label': label, 'seg': seg}
        # sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# class RandomHSV(object):
#     """
#         Args:
#             h_range (float tuple): random ratio of the hue channel,
#                 new_h range from h_range[0]*old_h to h_range[1]*old_h.
#             s_range (float tuple): random ratio of the saturation channel,
#                 new_s range from s_range[0]*old_s to s_range[1]*old_s.
#             v_range (int tuple): random bias of the value channel,
#                 new_v range from old_v-v_range to old_v+v_range.
#         Notice:
#             h range: 0-1
#             s range: 0-1
#             v range: 0-255
#         """
#
#     def __init__(self, h_range, s_range, v_range):
#         assert isinstance(h_range, (list, tuple)) and \
#                isinstance(s_range, (list, tuple)) and \
#                isinstance(v_range, (list, tuple))
#         self.h_range = h_range
#         self.s_range = s_range
#         self.v_range = v_range
#
#     def __call__(self, sample):
#         img = sample['image']
#         img_hsv = matplotlib.colors.rgb_to_hsv(img)
#         img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
#         h_random = np.random.uniform(min(self.h_range), max(self.h_range))
#         s_random = np.random.uniform(min(self.s_range), max(self.s_range))
#         v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
#         img_h = np.clip(img_h * h_random, 0, 1)
#         img_s = np.clip(img_s * s_random, 0, 1)
#         img_v = np.clip(img_v + v_random, 0, 255)
#         img_hsv = np.stack([img_h, img_s, img_v], axis=2)
#         img_new = matplotlib.colors.hsv_to_rgb(img_hsv)
#
#         # img = sample['depth']
#         # img_hsv = matplotlib.colors.rgb_to_hsv(img)
#         # img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
#         # h_random = np.random.uniform(min(self.h_range), max(self.h_range))
#         # s_random = np.random.uniform(min(self.s_range), max(self.s_range))
#         # v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
#         # img_h = np.clip(img_h * h_random, 0, 1)
#         # img_s = np.clip(img_s * s_random, 0, 1)
#         # img_v = np.clip(img_v + v_random, 0, 255)
#         # img_hsv = np.stack([img_h, img_s, img_v], axis=2)
#         # depth_new = matplotlib.colors.hsv_to_rgb(img_hsv)
#         return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}

class Resize(transforms.Resize):

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = self.size[0]
        w = self.size[1]
        # h = 1024
        # w = 2048
        sample['image'] = F.resize(image, (h, w))
        if sample['depth'] != None:
            sample['depth'] = F.resize(depth, (h, w))
        sample['label'] = F.resize(label, (h, w), interpolation=Image.NEAREST)

        if 'seg' in sample.keys():
            sample['seg'] = F.resize(sample['seg'], (h, w))

        return sample

# class scaleNorm(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, sample):
#         image_h = self.size[0]
#         image_w = self.size[1]
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#
#         # Bi-linear
#         # image = skimage.transform.resize(image, (image_h, image_w), order=1,
#         #                                  mode='reflect', preserve_range=True)
#         image=cv2.resize(image, (image_w, image_h),interpolation=cv2.INTER_LINEAR)
#         # Nearest-neighbor
#         depth=cv2.resize(depth, (image_w, image_h),interpolation=cv2.INTER_LINEAR)
#         # depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
#         #                                  mode='reflect', preserve_range=True)
#         # label = skimage.transform.resize(label, (image_h, image_w), order=0,
#         #                                  mode='reflect', preserve_range=True)
#         label=cv2.resize(label, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
#
#         return {'image': image, 'depth': depth, 'label': label}


# class RandomScale(object):
#     def __init__(self, scale):
#         self.scale_low = min(scale)
#         self.scale_high = max(scale)

#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']

#         target_scale = random.uniform(self.scale_low, self.scale_high)
#         # (H, W, C)
#         target_height = int(round(target_scale * image.size[1]))
#         target_width = int(round(target_scale * image.size[0]))
#         # print(target_width)
#         # Bi-linear
#         sample['image']=F.resize(image, (target_height, target_width))
#         if depth != None:
#             sample['depth']=F.resize(depth, (target_height, target_width))
#         sample['label']=F.resize(label, (target_height, target_width), interpolation=Image.NEAREST)
#         if 'seg' in sample.keys():
#             sample['seg'] = F.resize(sample['seg'], (target_height, target_width))


#         return sample

class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        # W, H = im.size
        target_scale = random.choice(self.scales)
        target_height = int(round(target_scale * image.size[1]))
        target_width = int(round(target_scale * image.size[0]))
        # print(target_width)
        # Bi-linear
        sample['image']=F.resize(image, (target_height, target_width))
        if depth != None:
            sample['depth']=F.resize(depth, (target_height, target_width))
        sample['label']=F.resize(label, (target_height, target_width), interpolation=Image.NEAREST)
        if 'seg' in sample.keys():
            sample['seg'] = F.resize(sample['seg'], (target_height, target_width))
        return sample


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, sample):
        image = sample['image']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        image = ImageEnhance.Brightness(image).enhance(r_brightness)
        image = ImageEnhance.Contrast(image).enhance(r_contrast)
        image = ImageEnhance.Color(image).enhance(r_saturation)
        sample['image']=image
        return sample
# class RandomCrop(object):
#     def __init__(self, th, tw):
#         self.th = th
#         self.tw = tw
#
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         h = image.shape[0]
#         w = image.shape[1]
#
#         i = random.randint(0, h-self.th)
#         j = random.randint(0, w-self.tw)
#
#
#         return {'image': image[i:i + self.th, j:j + self.tw,:],
#                 'depth': depth[i:i + self.th, j:j + self.tw,:],
#                 'label': label[i:i + self.th, j:j + self.tw]}

class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        i, j, h, w = self.get_params(image, self.size)

        sample['image'] = F.crop(image, i, j, h, w)
        if sample['depth'] != None:
            sample['depth'] = F.crop(depth, i, j, h, w)
        sample['label'] = F.crop(label, i, j, h, w)

        if 'seg' in sample.keys():
            sample['seg'] = F.crop(sample['seg'], i, j, h, w)

        return sample

class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        sample['image'] = F.center_crop(image, self.size)
        if sample['depth'] != None:
            sample['depth'] = F.center_crop(depth, self.size)
        sample['label'] = F.center_crop(label, self.size)

        if 'seg' in sample.keys():
            sample['seg'] = F.center_crop(sample['seg'], self.size)


        return sample

# class RandomRotation(object):
#
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         angle = random.randint(-13, 13)
#         image = Image.fromarray(np.uint8(image)).rotate(angle, Image.BILINEAR)
#         depth = Image.fromarray(np.uint8(depth)).rotate(angle, Image.BILINEAR)
#         label = Image.fromarray(np.uint8(label)).rotate(angle, Image.NEAREST)
#
#         return {'image': image, 'depth': depth, 'label': label}
# class randomColor(object):
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         random_factor = np.random.randint(6,14) / 10.  # 随机因子
#         brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)
#         random_factor = np.random.randint(5,15) / 10.  # 随机因1子
#         contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        # return {'image': contrast_image, 'depth': depth, 'label': label}

# class CenterCrop(object):
#     def __init__(self, th, tw):
#         self.th = th
#         self.tw = tw
#
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         h = image.shape[0]
#         w = image.shape[1]
#         i = int(h/2)-int(self.th/2)
#         j = int(w/2)-int(self.tw/2)
#
#         return {'image': image[i:i + image_h, j:j + image_w,:],
#                 'depth': depth[i:i + image_h, j:j + image_w,:],
#                 'label': label[i:i + image_h, j:j + image_w]}

# class FourCrop(object):
#     def __init__(self, th, tw):
#         self.th = th
#         self.tw = tw

#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         raw_h = image.shape[1]
#         raw_w = image.shape[2]
#         image_h=self.th
#         image_w=self.tw
#         dist_h=raw_h-image_h
#         dist_w=raw_w-image_w
#         crop1= image[:,:image_h,:image_w]#upleft
#         crop2= image[:,:image_h,dist_w:raw_w]#upright
#         crop3= image[:,dist_h:raw_h,:image_w]#downleft
#         crop4= image[:,dist_h:raw_h,dist_w:raw_w]#downright

#         sample['image']=np.concatenate((crop1,crop2,crop3,crop4), axis=0)
#         return sample

# class RandomFlip(object):
#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         if random.random() > 0.5:
#             image = np.fliplr(image).copy()
#             depth = np.fliplr(depth).copy()
#             label = np.fliplr(label).copy()
#
#         return {'image': image, 'depth': depth, 'label': label}

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            sample['image'] = F.hflip(image)
            if sample['depth'] != None:
                sample['depth'] = F.hflip(depth)
            sample['label'] = F.hflip(label)

            if 'seg' in sample.keys():
                sample['seg'] = F.hflip(sample['seg'])

        return sample

# class RandomScale(object):

#     def __init__(self, scale):
#         self.scale_low = min(scale)
#         self.scale_high = max(scale)

#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         target_scale = random.uniform(self.scale_low, self.scale_high)
#         target_h = int(round(target_scale * image.size[0]))
#         target_w = int(round(target_scale * image.size[1]))

#         sample['scale_image'] = F.resize(image, (target_h, target_w))
#         if sample['depth'] != None:
#             sample['scale_depth'] = F.resize(depth, (target_h, target_w))
#         sample['scale_label'] = F.resize(label, (target_h, target_w), interpolation=Image.NEAREST)

class MultiScale(object):

    def __init__(self, size, scale_times=4):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_times = scale_times

    def __call__(self, sample):
        h = self.size[0]
        w = self.size[1]
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # sample['A'] = [F.resize(A, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]
        if sample['depth'] != None:
            sample['depth'] = [F.resize(depth, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]

        return sample


class ToTensor(object):
    def __call__(self, sample):

        image, depth, label = sample['image'], sample['depth'], sample['label']

        sample['image'] = F.to_tensor(image)
        _label = np.maximum(np.array(label, dtype=np.int32), 0)
        sample['label'] = torch.from_numpy(_label).long()
        # sample['label'] = torch.squeeze(torch.from_numpy(np.expand_dims(np.asarray(label), 0)), 0)
        if sample['depth'] != None:
            if isinstance(depth, list):
                sample['depth'] = [F.to_tensor(item) for item in depth]
            else:
                sample['depth'] = F.to_tensor(depth)

        if 'seg' in sample.keys():
            sample['seg'] = F.to_tensor(sample['seg'])

        return sample


class Normalize(transforms.Normalize):

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        sample['image'] = F.normalize(image, self.mean, self.std)
        if sample['depth'] != None:
            if isinstance(depth, list):
                sample['depth'] = [F.normalize(item, self.mean, self.std) for item in depth]
            else:
                sample['depth'] = F.normalize(depth, self.mean, self.std)

        if 'seg' in sample.keys():
            sample['seg'] = F.normalize(sample['seg'], self.mean, self.std)
        if sample['depth'] == None:
            sample = {'image': image, 'label': label, 'seg': sample['seg'] }
        return sample