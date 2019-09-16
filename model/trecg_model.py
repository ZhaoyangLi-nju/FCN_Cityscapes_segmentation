import math
import os
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision

import util.utils as util
from util.average_meter import AverageMeter
from . import networks_fcn_resnet18 as networks
from .base_model import BaseModel

class TRecgNet(BaseModel):

    def __init__(self, cfg, vis=None, writer=None):
        super(TRecgNet, self).__init__(cfg)

        util.mkdir(self.save_dir)
        assert (self.cfg.WHICH_DIRECTION is not None)
        self.AtoB = self.cfg.WHICH_DIRECTION == 'AtoB'
        self.modality = 'rgb' if self.AtoB else 'depth'
        self.sample_model = None
        self.phase = cfg.PHASE
        self.using_semantic_branch= cfg.USING_SEMANTIC_BRANCH
        self.content_model = None
        self.content_layers = []
        self.batch_size=cfg.BATCH_SIZE

        self.writer = writer
        self.vis = vis

        # networks
        self.use_noise = cfg.WHICH_DIRECTION == 'BtoA'
        self.net = networks.define_TrecgNet(cfg, using_semantic_branch=cfg.USING_SEMANTIC_BRANCH,
                                            device=self.device)
        networks.print_network(self.net)

    def build_output_keys(self, upsample=True, cls=True):

        out_keys = []

        if upsample:
            out_keys.append('gen_img')

        if cls:
            out_keys.append('cls')

        return out_keys

    def _optimize(self, iters):

        self._forward(iters)
        self.optimizer_ED.zero_grad()
        total_loss = self._construct_TRAIN_G_LOSS(iters)
        total_loss.backward()
        self.optimizer_ED.step()

    def train_parameters(self, cfg):

        assert (self.cfg.LOSS_TYPES)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.EVALUATE:
            # criterion_segmentation = util.CrossEntropyLoss2d_new(ignore_index=255)
            criterion_segmentation = util.CrossEntropyLoss2d_semantic_segmentation(ignore_index=19)
            # criterion_segmentation = util.CrossEntropyLoss2d()
            self.net.set_cls_criterion(criterion_segmentation)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            criterion_content = torch.nn.L1Loss()
            content_model = networks.Content_Model(cfg, criterion_content).to(self.device)
            self.net.set_content_model(content_model)

        self.set_optimizer(cfg)
        self.set_log_data(cfg)
        self.set_schedulers(cfg)
        self.net = nn.DataParallel(self.net).to(self.device)

        train_total_steps = 0
        train_total_iter = 0
        best_prec = 0

        total_epoch = int(cfg.NITER_TOTAL / math.ceil((self.train_image_num / cfg.BATCH_SIZE)))
        print('total epoch:{0}, total iters:{1}'.format(total_epoch, cfg.NITER_TOTAL))

        for epoch in range(cfg.START_EPOCH, total_epoch + 1):

            if train_total_iter > cfg.NITER_TOTAL:
                break

            self.print_lr()

            self.imgs_all = []
            self.pred_index_all = []
            self.target_index_all = []
            self.fake_image_num = 0

            start_time = time.time()

            data_loader = self.get_dataloader(cfg, epoch)

            self.phase = 'train'
            self.net.train()

            for key in self.loss_meters:
                self.loss_meters[key].reset()

            iters = 0
            # self.evaluate(cfg=self.cfg, epoch=epoch)
            for i, data in enumerate(data_loader):

                self.update_learning_rate(epoch=train_total_iter)

                # self.set_input(data, self.cfg.DATA_TYPE)
                self.source_modal = data['image'].to(self.device)
                self.label = data['label'].to(self.device)
                target_modal = data['seg']
                # target_modal = data['depth']

                # if isinstance(target_modal, list):
                #     self.target_modal = list()
                #     for i, item in enumerate(target_modal):
                #         self.target_modal.append(item.to(self.device))
                # else:
                    # self.target_modal = util.color_label(self.label)
                self.target_modal = target_modal.to(self.device)

                train_total_steps += self.batch_size
                train_total_iter += 1
                iters += 1

                self._optimize(train_total_iter)
                # self._write_loss(phase=self.phase, global_step=train_total_iter)

            # self.train_iou = self.loss_meters['TRAIN_I'].sum / (self.loss_meters['TRAIN_U'].sum + 1e-10)
            print('iters in one epoch:', iters)
            print('gpu_ids:', cfg.GPU_IDS)
            self._write_loss(phase=self.phase, global_step=train_total_iter)
            # train_errors = self.get_current_errors(current=False)
            print('Epoch: {epoch}/{total}'.format(epoch=epoch, total=total_epoch))
            train_errors = self.get_current_errors(current=False)
            print('#' * 10)
            self.print_current_errors(train_errors, epoch)
            print('#' * 10)
            print('Training Time: {0} sec'.format(time.time() - start_time))

            # Validate cls
            if (epoch % 10 == 0 and epoch>50 )or epoch == total_epoch:
                if cfg.EVALUATE:
                    self.val_iou = self.validate(train_total_iter)
                    # self.val_iou = self.loss_meters['VAL_I'].sum / (self.loss_meters['VAL_U'].sum + 1e-10)
                    # print('MIOU:', self.val_iou.mean() * 100)
                    print('MIOU:', self.val_iou * 100)
                    self._write_loss(phase=self.phase, global_step=train_total_iter)

            print('End of iter {0} / {1} \t '
                  'Time Taken: {2} sec'.format(train_total_iter, cfg.NITER_TOTAL, time.time() - start_time))
            print('-' * 80)

    # encoder-decoder branch
    def _forward(self, iters):

        self.gen = None
        self.cls_loss = None

        if self.phase == 'train':

            if 'CLS' not in self.cfg.LOSS_TYPES:
                if_upsample = True
                if_cls = False

            elif 'SEMANTIC' in self.cfg.LOSS_TYPES and 'CLS' in self.cfg.LOSS_TYPES:
                if_upsample = True
                if_cls = True
            else:
                if_upsample = False
                if_cls = True
        else:
            if_cls = True
            if_upsample = False
            # for time saving
            if iters > self.cfg.NITER_TOTAL - 500 and 'SEMANTIC' in self.cfg.LOSS_TYPES:
                if_upsample = True

        self.source_modal_show = self.source_modal  # rgb
        out_keys = self.build_output_keys(upsample=if_upsample, cls=if_cls)
        self.result = self.net(source=self.source_modal, target=self.target_modal, label=self.label, out_keys=out_keys,
                               phase=self.phase)

        if if_cls:
            self.cls = self.result['cls']

        if if_upsample:
            if self.cfg.MULTI_MODAL:
                self.gen = [self.result['gen_img_1'], self.result['gen_img_2']]
            else:
                self.gen = self.result['gen_img']


    def _construct_TRAIN_G_LOSS(self, iters=None):

        loss_total = torch.zeros(1)
        if self.use_gpu:
            loss_total = loss_total.to(self.device)

        # if self.gen is not None:
        #     assert (self.gen.size(-1) == self.cfg.FINE_SIZE)

        if 'CLS' in self.cfg.LOSS_TYPES:
            cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
            loss_total = loss_total + cls_loss

            cls_loss = round(cls_loss.item(), 4)
            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss)
            # self.Train_predicted_label = self.cls.data
            self.Train_predicted_label = self.cls.data.max(1)[1].cpu().numpy()


        # ) content supervised

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and self.using_semantic_branch:

            # content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * (iters / self.cfg.NITER_TOTAL)
            # content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * max(0, (self.cfg.NITER_TOTAL - iters) / self.cfg.NITER_TOTAL)
            content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT
            loss_total = loss_total + content_loss

            content_loss = round(content_loss.item(), 4)
            self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss)

        return loss_total


    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_G_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'VAL_CLS_LOSS',
            'TRAIN_CLS_MEAN_IOU',
            'VAL_CLS_MEAN_IOU',
            'TRAIN_I',
            'TRAIN_U',
            'VAL_I',
            'VAL_U',
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def save_checkpoint(self, iter, filename=None):

        if filename is None:
            filename = 'TRecg2Net_{0}_{1}.pth'.format(self.cfg.WHICH_DIRECTION, iter)

        net_state_dict = self.net.state_dict()
        save_state_dict = {}
        for k, v in net_state_dict.items():
            if 'content_model' in k:
                continue
            save_state_dict[k] = v

        state = {
            'iter': iter,
            'state_dict': save_state_dict,
            'optimizer_ED': self.optimizer_ED.state_dict(),
        }
        if 'GAN' in self.cfg.LOSS_TYPES:
            state['state_dict_D'] = self.net_D.state_dict()
            state['optimizer_D'] = self.optimizer_D.state_dict()

        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, net, checkpoint_path, checkpoint, optimizer=None, data_para=True):

        keep_fc = not self.cfg.NO_FC

        if os.path.isfile(checkpoint_path):

            # load from pix2pix net_G, no cls weights, selected update
            state_dict = net.state_dict()
            state_checkpoint = checkpoint['state_dict']
            if data_para:
                new_state_dict = OrderedDict()
                for k, v in state_checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_checkpoint = new_state_dict

            if keep_fc:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict}
            else:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict and 'fc' not in k}

            state_dict.update(pretrained_G)
            net.load_state_dict(state_dict)

            if self.phase == 'train' and not self.cfg.INIT_EPOCH:
                optimizer.load_state_dict(checkpoint['optimizer_ED'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> !!! No checkpoint found at '{}'".format(self.cfg.RESUME))
            return

    def set_optimizer(self, cfg):

        self.optimizers = []
        # self.optimizer_ED = torch.optim.Adam([{'params': self.net.fc.parameters(), 'lr': cfg.LR}], lr=cfg.LR / 10, betas=(0.5, 0.999))

        self.optimizer_ED = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        print('optimizer G: ', self.optimizer_ED)
        self.optimizers.append(self.optimizer_ED)

        # if 'GAN' in self.cfg.LOSS_TYPES:
        #     self.optimizer_D = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net_D.parameters()), cfg.LR,
        #                                        momentum=cfg.MOMENTUM,
        #                                        weight_decay=cfg.WEIGHT_DECAY)
        #     print('optimizer D: ', self.optimizer_D)
        #     self.optimizers.append(self.optimizer_D)

    def validate(self, iters):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()

        self.imgs_all = []
        self.pred_index_all = []
        self.target_index_all = []

        inputs_all, gts_all, predictions_all = [], [], []
        with torch.no_grad():

            print('# Cls val images num = {0}'.format(self.val_image_num))
            # batch_index = int(self.val_image_num / cfg.BATCH_SIZE)
            # random_id = random.randint(0, batch_index)
            # confusion_matrix = np.zeros((37,37))
            for i, data in enumerate(self.val_loader):
                # self.set_input(data, self.cfg.DATA_TYPE)
                self.source_modal=data['image'].to(self.device)
                # self.target_modal=data['depth'].cuda()
                self.label=data['label'].to(self.device)

                self._forward(iters)
                self.val_predicted_label = self.cls.data.max(1)[1].cpu().numpy()

                gts_all.append(self.label.data.cpu().numpy())
                predictions_all.append(self.val_predicted_label)

                # self.val_predicted_label = self.cls.data
                # _, pred = torch.max(self.cls.data, dim=1)
                # acc, pix = util.accuracy(pred + 1, self.label.long())
                # intersection, union = util.intersectionAndUnion(pred, self.label, 37)
                # self.loss_meters['VAL_CLS_ACC'].update(acc, pix, acc_Flag=True)
                # self.loss_meters['VAL_I'].update(intersection)
                # self.loss_meters['VAL_U'].update(union)

        gts_all = np.concatenate(gts_all)
        predictions_all = np.concatenate(predictions_all)

        acc, acc_cls, mean_iu, fwavacc = util.evaluate(predictions_all, gts_all, self.cfg.NUM_CLASSES)

        return mean_iu
        # self.loss_meters['VAL_CLS_ACC'].update(acc, pix, acc_Flag=True)
        # self.loss_meters['VAL_I'].update(intersection)
        # self.loss_meters['VAL_U'].update(union)

                # ########################
                # seg_pred = np.asarray(np.argmax(self.cls.data, axis=1),dtype=np.uint8)+1#0-36->1-37
                # seg_gt = np.asarray(self.label.cpu().numpy(), dtype=np.int)
                # ignore_index = ((seg_gt != 255) & (seg_gt>0))
                # seg_gt = seg_gt[ignore_index]
                # seg_pred = seg_pred[ignore_index]
                # confusion_matrix += get_confusion_matrix(seg_gt, seg_pred)

                # ########################
                # if i % 50 == 0:
                #     print('step:',i,'Val_loss:',round(self.loss_meters['VAL_CLS_LOSS'].average(),4),'Val_acc:',round(float(self.loss_meters['VAL_CLS_ACC'].average())*100.0,4))
            # pos = confusion_matrix.sum(1)
            # res = confusion_matrix.sum(0)
            # tp = np.diag(confusion_matrix)
            # IU_array = (tp / np.maximum(1.0, pos + res - tp))
            # mean_IU = IU_array.mean()

            # self.writer.add_scalar('VAL_MEAN_IOU_confusion_matrix', float(mean_IU)*100.0,
            #                        global_step=epoch)



    def _write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES

        self.label_show = self.label.data.cpu().numpy()
        self.source_modal_show = self.source_modal
        self.target_modal_show = self.target_modal

        if phase == 'train':

            self.writer.add_scalar('LR', self.optimizer_ED.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('Seg/TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_ACC', self.loss_meters['TRAIN_CLS_ACC'].avg*100.0,
                #                        global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_MEAN_IOU', float(self.train_iou.mean())*100.0,
                #                        global_step=global_step)

          
            if 'SEMANTIC' in loss_types and self.using_semantic_branch:
                self.writer.add_scalar('Seg/TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                                       global_step=global_step)

                if isinstance(self.target_modal, list):
                    for i, (gen, target) in enumerate(zip(self.gen, self.target_modal)):

                        self.writer.add_image('Seg/2_Train_Gen_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                              torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                                                                          normalize=True),
                                               global_step=global_step)
                        self.writer.add_image('Seg/3_Train_Target_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                              torchvision.utils.make_grid(target[:6].clone().cpu().data, 3,
                                                                          normalize=True),
                                              global_step=global_step)
                else:
                    self.writer.add_image('Seg/Train_groundtruth_depth',
                                      torchvision.utils.make_grid(self.target_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                    self.writer.add_image('Seg/Train_predicted_depth  ',
                                      torchvision.utils.make_grid(self.gen.data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)


            self.writer.add_image('Seg/Train_image',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if 'CLS' in loss_types:
                self.writer.add_image('Seg/Train_predicted_label',
                                      torchvision.utils.make_grid(torch.from_numpy(util.color_label(self.Train_predicted_label[:6])), 3, normalize=True, range=(0, 255)), global_step=global_step)
                                      # torchvision.utils.make_grid(util.color_label(torch.max(self.Train_predicted_label[:6], 1)[1]+1), 3, normalize=False,range=(0, 255)), global_step=global_step)
                self.writer.add_image('Seg/Train_ground_label',
                                      torchvision.utils.make_grid(torch.from_numpy(util.color_label(self.label_show[:6])), 3, normalize=True, range=(0, 255)), global_step=global_step)
                                      # torchvision.utils.make_grid(util.color_label(self.label_show[:6]), 3, normalize=False,range=(0, 255)), global_step=global_step)
            # if self.upsample:
            #     self.writer.add_image('Gen_depth', torchvision.utils.make_grid(self.gen[:3].clone().cpu().data, 3,
            #                                                                      normalize=True),
            #                           global_step=global_step)
            #     self.writer.add_image('Train_3_Target',
            #                           torchvision.utils.make_grid(self.target_modal_show[:3].clone().cpu().data, 3,
                                                                  # normalize=True), global_step=global_step)

        if phase == 'test':

            self.writer.add_image('Seg/Val_image',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)

            self.writer.add_image('Seg/Val_predicted_label',
                                  torchvision.utils.make_grid(torch.from_numpy(util.color_label(self.val_predicted_label[:6])), 3, normalize=True,range=(0, 255)), global_step=global_step)
                                  # torchvision.utils.make_grid(util.color_label(torch.max(self.val_predicted_label[:3], 1)[1]+1), 3, normalize=False,range=(0, 255)), global_step=global_step)
            self.writer.add_image('Seg/Val_ground_label',
                                  torchvision.utils.make_grid(torch.from_numpy(util.color_label(self.label_show[:6])), 3, normalize=True,range=(0, 255)), global_step=global_step)

            # self.writer.add_scalar('Seg/VAL_CLS_LOSS', self.loss_meters['VAL_CLS_LOSS'].avg,
            #                        global_step=global_step)
            # self.writer.add_scalar('Seg/VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].avg*100.0,
            #                        global_step=global_step)
            self.writer.add_scalar('Seg/VAL_CLS_MEAN_IOU', float(self.val_iou.mean())*100.0,
                                   global_step=global_step)
            # self.writer.add_scalar('Seg/VAL_CLS_MEAN_IOU', float(self.val_iou.mean())*100.0,
            #                        global_step=global_step)
def get_confusion_matrix(gt_label, pred_label, class_num=37):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))
    

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix
