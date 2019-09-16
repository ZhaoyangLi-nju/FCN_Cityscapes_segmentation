import os
import socket
from datetime import datetime

class RESNET18_SUNRGBD_CONFIG:

    def args(self):
        args = {'ROOT_DIR': '/home/lzy/summary/'}
        # args = {'ROOT_DIR': '/home/lzy/summary'}
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')

        ########### Quick Setup ############
        model = 'FCN'

        task_name = 'add_semantic_resnet18_2e-4_384*768_'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'place'
        content_pretrained = 'place'
        gpus = '1,2,3,4,5'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size = 30
        direction = 'AtoB'  # AtoB: RGB->Depth
        # direction = 'BtoA'
        # loss = ['CLS']  # remove 'CLS' if trained with unlabeled data
        loss = ['CLS', 'SEMANTIC']  # remove 'CLS' if trained with unlabeled data

        using_semantic_branch = True  # True for removing Decoder network
        unlabeld = False     # True for training with unlabeled data
        evaluate = True      # report mean acc after each epoch
        content_layers = '0,1,2,3,4' # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.2
        multi_scale = False
        multi_modal = False
        which_score = 'up'
        norm = 'in'

        len_gpu = str(len(gpus.split(',')))

        use_fake = False
        fake_rate = 0.3
        sample_path = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                                 'PSG_BtoA.pth')
        resume = False
        resume_path = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                     '10k_place_AtoB.pth')
        resume_path_AtoB = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                     'PS_AtoB.pth')
        resume_path_BtoA = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                     'PS_BtoA.pth')


        log_path = os.path.join(args['ROOT_DIR'], model, content_pretrained,
                                ''.join([task_name, '_', lr_schedule, '_', 'gpus-', len_gpu
                                ]), current_time)

        return {

            'MODEL': model,
            'GPU_IDS': gpus,
            'WHICH_DIRECTION': direction,
            'BATCH_SIZE': batch_size,
            'LOSS_TYPES': loss,
            'PRETRAINED': pretrained,

            'LOG_PATH': log_path,
            'data_dir': '/data0/lzy/SUNRGBD',

            # MODEL
            'ARCH': 'resnet18',
            'SAVE_BEST': True,
            'USING_SEMANTIC_BRANCH': using_semantic_branch,

            #### DATA
            'NUM_CLASSES': 19,
            'UNLABELED': unlabeld,
            'USE_FAKE_DATA': use_fake,
            'SAMPLE_MODEL_PATH': sample_path,
            'FAKE_DATA_RATE': fake_rate,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'RESUME_PATH_AtoB': resume_path_AtoB,
            'RESUME_PATH_BtoA': resume_path_BtoA,
            'LR_POLICY': lr_schedule,

            'NITER': 2000,
            'NITER_DECAY': 8000,
            'NITER_TOTAL': 10000,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,

            # translation task
            'WHICH_CONTENT_NET': 'resnet18',
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'MULTI_SCALE': multi_scale,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm
        }
