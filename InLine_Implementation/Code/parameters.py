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
"""
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
"""
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'


class Parameters():
    def __init__(self):
        super(Parameters, self).__init__()

        ## Hardware/GPU parameters =================================================
        self.Op_Node = 'spider'  # 'alpha_V12' # 'myPC', 'O2', 'spider'
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.tbVisualize = False
        self.tbVisualize_kernels = False
        self.tbVisualize_featuremaps = False
        self.multi_GPU = False

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
        # self.args = parser.parse_args()

        self.activation_func = 'CReLU'  # 'CReLU' 'CLeakyeak' # 'modReLU' 'KAF2D' 'ZReLU'
        self.lr = 1e-8 #0.0001
        self.dropout_ratio = 0.0
        self.epochs = 4996
        self.training_percent = 0.8
        self.nIterations = 1
        self.magnitude_only = False
        self.Validation_Only = True     #option for training or validation
        self.Evaluation = False

        #########
        # self.MODEL = 0 # Original U-net implementation
        # self.MODEL = 1 # Shallow U-net implementation with combination of magnitude and phase
        #         self.MODEL = 2 # #The OLD working Real and Imaginary network (Residual Network with one global connection)
        #         self.MODEL = 3 # Complex Shallow U-net
        #         self.MODEL = 3.1 # Complex stacked convolution layers
        #         self.MODEL = 3.2 # Complex Shallow U-net with different kernel configuration
        #         self.MODEL = 3.3 # Complex 3D U-net with multi GPU implemntation to fit whole 3D volume
        #         self.MODEL = 3.4 # Complex 3D U-net with multi GPU implemntation to fit whole 3D volume
        #        self.MODEL = 4 # Complex Shallow U-Net with residual connection
        #         self.MODEL = 5 # Complex Shallow U-Net with residual connection with 32 multi coil output
        #         self.MODEL = 6 # Complex fully connected layer
        #         self.MODEL = 7 # Real shallow U-net layer [double size]
        #         self.MODEL = 8 # complex conv network that maps k-space to image domain
        # self.MODEL = 9  # Complex Network takes neighborhood matrix input and image domain output
        self.MODEL = 10
        #########
        if self.MODEL in [2, 3, 3.1, 3.2, 3.3, 3.4, 4, 5, 6, 8, 9]:
            self.complex_net = True
        else:
            self.complex_net = False

        ## Dataset and paths =================================================
        self.g_methods = ['grid_kernels', 'neighbours_matrix', 'pyNUFFT', 'BART', 'python_interp']
        self.gridding_method = 'MOLLI_MIRT_NUFFT'

        self.gd_methods = ['RING', 'AC-ADDAPTIVE', 'NONE']
        self.gradient_delays_method = ''  # self.gd_methods[0]
        self.rot_angle = True

        self.k_neighbors = 20

        self.ds_total_num_slices = 0
        self.patients = []
        self.num_phases = 5
        self.radial_cine = True
        self.n_spokes = 40  # 16 #20 #33
        self.Rate = np.round(198 / self.n_spokes) if self.radial_cine else 3
        self.input_slices = list()
        self.num_slices_per_patient = list()
        self.groundTruth_slices = list()
        self.training_patients_index = list()
        self.us_rates = list()
        self.saveVolumeData = False
        self.multiCoilInput = True
        self.coilCombinedInputTV = True
        self.moving_window_size = 5

        if self.network_type == '2D':
            self.img_size = [416, 416]
        else:
            self.img_size = [416, 416, self.moving_window_size]  # [50, 50, 20]  # 64, 256, 320

        if self.multiCoilInput:
            self.n_channels = 1
        else:
            self.n_channels = 1

        self.cropped_dataset64 = False
        if self.cropped_dataset64:
            crop_txt = '_cropped64'
        else:
            crop_txt = ''
        self.trialNum = '_T1Fitting_5071_MOLLI5_MAE_alldata' #best results '_T1_5016' and 5031, and _T1_5041_MOLLI
        self.arch_name = 'Model_0' + str(self.MODEL) + '_R' + str(self.Rate) + 'Trial' + self.trialNum

        if self.Op_Node == 'alpha_V12':
            if self.coilCombinedInputTV:
                self.dir = {'/mnt/D/Image Reconstruction-Hossam/NoDataset/',
                            '/mnt/C/Hossam/ReconData_coilCombTVDL/Rate_' + str(self.Rate) + '/'
                            }
            else:
                self.dir = {'/mnt/D/Image Reconstruction-Hossam/Dataset/ReconData_cmplxDL/',
                            '/mnt/C/Hossam/ReconData_cmplxDL/'
                            }
            self.model_save_dir = '/mnt/C/Hossam/RecoNet-Model/' + self.arch_name + '/'
            self.net_save_dir = '/mnt/D/Image Reconstruction-Hossam/MatData/'
            self.tensorboard_dir = '/mnt/C/Hossam/RecoNet-Model/' + self.arch_name + '_tensorboard/'


        elif self.Op_Node == 'myPC':
            if self.coilCombinedInputTV:
                self.dir = {'/media/helrewaidy/F/Image Reconstruction/ReconData_coilCombTVDL/Rate_' + str(
                    self.Rate) + crop_txt + '/',
                            '/mnt/D/BIDMC Workspace/Image Reconstruction/ReconData_coilCombTVDL/Rate_' + str(
                                self.Rate) + '/'
                            }
            else:
                self.dir = {'/mnt/C/Image Reconstruction/ReconData_cmplxDL/',
                            # '/media/helrewaidy/Windows7_OS/LGE Dataset/',
                            '/mnt/D/BIDMC Workspace/Image Reconstruction/ReconData_cmplxDL/'
                            }
            self.model_save_dir = '/mnt/D/BIDMC Workspace/Image Reconstruction/RecoNet-Model/' + self.arch_name + '/'
            self.net_save_dir = '/mnt/D/BIDMC Workspace/Image Reconstruction/MatData/'
            self.tensorboard_dir = '/mnt/D/BIDMC Workspace/Image Reconstruction/RecoNet-Model/' + self.arch_name + '_tensorboard/'


        elif self.Op_Node == 'O2':
            if self.coilCombinedInputTV:
                self.dir = {'/n/scratch2/hae1/ReconData/ReconData_coilCombTVDL/Rate_' + str(self.Rate) + '/'
                            # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                            }
            else:
                self.dir = {'/n/scratch2/hae1/ReconData/ReconData_cmplxDL/'
                            # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                            }
            self.model_save_dir = '/n/data2/bidmc/medicine/nezafat/hossam/DeepLearning/RecoNet_Model/' + self.arch_name + '/'
            self.net_save_dir = '/n/data2/bidmc/medicine/nezafat/hossam/DeepLearning/MatData/'
            self.tensorboard_dir = '/n/data2/bidmc/medicine/nezafat/hossam/DeepLearning/RecoNet_Model/' + self.arch_name + '_tensorboard/'



        elif self.Op_Node == 'spider':

            if self.radial_cine:
                self.dir = ['/data1/helrewaidy/cine_recon/ICE_recon_dat_files/ice_dat_files/'
                            ]
                # ['/data2/helrewaidy/cine_recon/ICE_recon_dat_files/ice_dat_files/'
                #  ]
                self.model_save_dir = '/data2/helrewaidy/cine_recon/models/' + self.arch_name + '/'
                self.net_save_dir = '/data2/helrewaidy/t1mapping_recon/matlab_workspace/' + self.arch_name + '/'
                self.tensorboard_dir = '/data2/helrewaidy/cine_recon/models/' + self.arch_name + '_tensorboard/'

            else:
                if self.coilCombinedInputTV:
                    self.dir = ['/data2/helrewaidy/ReconData_coilCombTVDL/Rate_' + str(self.Rate) + '/'
                                # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                                ]
                else:
                    self.dir = {'/n/scratch2/hae1/ReconData/ReconData_cmplxDL/'
                                # '/media/helrewaidy/Windows7_OS/LGE Dataset/'
                                }
                self.model_save_dir = '/data2/helrewaidy/Models/ReconNet_Model/' + self.arch_name + '/'
                self.net_save_dir = '/data2/helrewaidy/Models/MatData/'
                self.tensorboard_dir = '/data2/helrewaidy/Models/ReconNet_Model/' + self.arch_name + '_tensorboard/'

        self.model = self.model_save_dir + 'MODEL_EPOCH.pth'














