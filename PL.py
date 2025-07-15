import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS,Decoder_modual,IDH_network
from models.unet import UNet3D
import torch.distributed as dist
from models import criterions
from contextlib import nullcontext
from data.BraTS_IDH import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from utils.pcgrad import PCGrad
import nibabel as nib
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from models.criterions import MultiTaskLossWrapper
import models.link as link
import utils.train_util as pseudo
from sklearn.metrics import roc_auc_score,accuracy_score
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()
# Basic Information
parser.add_argument('--user', default='yihao', type=str)

parser.add_argument('--experiment', default='TransBraTS_IDH', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBraTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='/public/home/hpc226511030/Archive/BraTS2021_TrainingData/', type=str)

parser.add_argument('--train_dir', default='MICCAI_BraTS2021_TrainingData', type=str)

parser.add_argument('--valid_dir', default='MICCAI_BraTS2021_TrainingData', type=str)

parser.add_argument('--test_dir', default='MICCAI_BraTS2021_TrainingData', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train_list_1.txt', type=str) 

parser.add_argument('--valid_file', default='test_list_1.txt', type=str) 

parser.add_argument('--test_file', default='test_list_1.txt', type=str)

parser.add_argument('--dataset', default='brats_IDH', type=str)

parser.add_argument('--model_name', default='TransBraTS', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=240, type=int)

parser.add_argument('--input_W', default=240, type=int)

parser.add_argument('--input_D', default=155, type=int)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1', type=str)

parser.add_argument('--num_workers', default=0, type=int)  #ORIGINAL 8

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=200, type=int)

parser.add_argument('--save_freq', default=50, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

parser.add_argument('--tao_u', default=0.05, type=float)    

parser.add_argument('--tao_c', default=0.9, type=float)   

args = parser.parse_args()

model_PL = {}

model_PL['en'] = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
model_PL['seg'] = seg_model = Decoder_modual()
model_PL['idh'] = IDH_model = IDH_network()

model_PL['en'] = nn.DataParallel(model_PL['en'])
model_PL['seg'] = nn.DataParallel(model_PL['seg'])
model_PL['idh'] = nn.DataParallel(model_PL['idh'])

resume = "/public/home/hpc226511030/GMMAS/checkpoint/TransBraTS_IDH2023-12-06/model_epoch_best.pth"
checkpoint = torch.load(resume, map_location='cpu')

model_PL['en'].load_state_dict(checkpoint['en_state_dict'])
model_PL['seg'].load_state_dict(checkpoint['seg_state_dict'])
model_PL['idh'].load_state_dict(checkpoint['idh_state_dict'])

print('loaded checkpoint {}'.format(resume))

ub_root = '/public/home/hpc226511030/Archive/BraTS2021_TrainingData/MICCAI_BraTS2021_TrainingData'
ub_list = os.path.join(ub_root, 'pq_ub_1.txt')
unlabel_set = BraTS(ub_list, ub_root, 'valid')
names = unlabel_set.names
pseudo_loader = DataLoader(unlabel_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
pseudo.pq_pseudo_labeling(pseudo_loader,model_PL,names,args)

logging.info('----------------------------------The pseudo labeling process finished!-----------------------------------')