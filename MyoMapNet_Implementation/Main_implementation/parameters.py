'''
Created on May 17, 2018
@author: helrewaidy
'''
# models

import argparse

from pycparser.c_ast import Switch
import torch
import numpy as np
import os

########################## Initializations ########################################
model_names = 'recoNet_Model1'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--cpu', '-c', action='store_true',
                    help='Do not use the cuda version of the net',
                    default=False)
parser.add_argument('--viz', '-v', action='store_true',
                    help='Visualize the images as they are processed',
                    default=False)
parser.add_argument('--no-save', '-n', action='store_false',
                    help='Do not save the output masks',
                    default=False)
parser.add_argument('--model', '-m', default='MODEL_EPOCH417.pth',
                    metavar='FILE',
                    help='Specify the file in which is stored the model'
                         " (default : 'MODEL.pth')")
###################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'


class Parameters():
    def __init__(self):
        super(Parameters, self).__init__()

        ## Hardware/GPU parameters =================================================
        self.Op_Node = 'spider'  # 'alpha_V12' # 'myPC', 'O2', 'spider'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tbVisualize = False
        self.tbVisualize_kernels = False
        self.tbVisualize_featuremaps = False
        self.multi_GPU = True

        if self.Op_Node in ['myPC', 'alpha_V12']:
            self.device_ids = [0]
        elif self.Op_Node in ['spider', 'O2']:
            self.device_ids = range(0, torch.cuda.device_count())

        if self.Op_Node in ['spider', 'O2', 'alpha_V12']:
            self.data_loders_num_workers = 40
        else:
            self.data_loders_num_workers = 4

        ## Network/Model parameters =================================================
        self.network_type = '2D'
        self.num_slices_3D = 7
        if self.Op_Node in ['myPC', 'alpha_V12']:
            self.batch_size = 2
        elif self.Op_Node in ['spider', 'O2']:
            self.batch_size = 1 # * len(self.device_ids) // 8 // (self.num_slices_3D if self.network_type == '3D' else 1)

        print('-- # GPUs: ', len(self.device_ids))
        print('-- batch_size: ', self.batch_size)
        self.args = parser.parse_args()

        self.activation_func = 'CReLU'  # 'CReLU' 'CLeakyeak' # 'modReLU' 'KAF2D' 'ZReLU'

        self.args.lr = 1e-3
        self.dropout_ratio = 0.0
        self.epochs = 1000
        self.training_percent = 0.8
        self.nIterations = 1
        self.magnitude_only = False


        self.training_with_Sim = False
        self.training_with_Invivo = True


        self.NetName = 'MyoMapNet_5PreGd'  #trained for native T1
        self.NetName = 'MyoMapNet_4PreGd'
        self.NetName = 'MyoMapNet_4PreandPostGd'
        self.NetName = 'MyoMapNet_4PostGd'

        self.inputLen = 4

        self.trialNum = '_T1Cal_MAE_alldata' #best results '_T1_5016' and 5031, and _T1_5041_MOLLI
        self.arch_name = 'Model_Net_' + str(self.MODEL) + '_Trial_NetName_'+ self.NetName +'_' + self.trialNum

        self.model_save_dir = '/Models/' + self.arch_name + '/'
        self.net_save_dir = '/NetSaveDir/' + self.arch_name + '/'
        self.tensorboard_dir = '/Models/' + self.arch_name + '_tensorboard/'
        self.tensorboard_dir = '/Models/' + self.arch_name + '_tensorboardLoss/'

        self.validation_dir = '/data2/rguo/Projects/MyoMapNet1.5/Models/' + self.arch_name + '_validation/PreContrast/'
        self.validation_dir = '/data2/rguo/Projects/MyoMapNet1.5/Models/' + self.arch_name + '_validation/Loss/'
        self.TestingResults_dir = '/data2/rguo/Projects/MyoMapNet1.5/Models/' + self.arch_name + '_testing/Prospective0616/'
        # self.TestingResults_dir = '/data2/rguo/Projects/MyoMapNet1.5/Models/' + self.arch_name + '_testing/Retrospective_allPreContrast'+ '_'+self.arch_name+'//'

        self.preMyoSimFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Preconstrast_numerical_simulation/precontrastMyo_s1500000.mat'
        self.preBloodSimFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Preconstrast_numerical_simulation/precontrastBlood_s1500000.mat'

        self.postMyoSimFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Postconstrast_numerical_simulation/postcontrastMyo_s1500000.mat'
        self.postBloodSimFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Postconstrast_numerical_simulation/postcontrastBlood_s1500000.mat'

        self.preInvivoFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Precontrast_MOLLI53_20190101_20201231/allInvivoT1map_160_160.mat'
        self.preInvivoFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Precontrast_MOLLI53_20190101_20201231/allInvivoT1map_160_160_Circlemask.mat'
        self.preInvivoFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Precontrast_MOLLI53_20190101_20201231/allInvivoT1map_160_160_rmNoiseMask.mat'
        self.postInvivoFile = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Training/Postcontrast_MOLLI53_20190101_20201231/allInvivoT1map_160_160_mask.mat'

        self.ValidationPreInvivo_5HBs = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Validation/Precontrast_MOLLI53_20200619_20200818/allInvivoT1map_160_160_5T1w.mat'
        self.ValidationPostInvivo_4HBs = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Validation/Postcontrast_MOLLI432_20200619_20200818/allInvivoT1map_160_160.mat'
        self.ValidationPrePostInvivo_5HBs = '/mnt/alp/Users/RuiGuo/MyoMapNet1.5/Validation/allInvivoT1map_160_160_mask.mat'




        self.args.model = self.model_save_dir + 'MODEL_EPOCH.pth'
