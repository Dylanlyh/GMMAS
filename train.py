# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py
import shutil
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
from models.TransBraTS.head import Unet_head
import torch.distributed as dist
from models import criterions
from contextlib import nullcontext
from data.BraTS_IDH import BraTS                      # TODO 此处修改dataset
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from utils.pcgrad import PCGrad
import nibabel as nib
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from models.criterions import MultiTaskLossWrapper
import utils.train_util as pseudo
from sklearn.metrics import roc_auc_score,accuracy_score
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# TODO: all parameters to be determined before training

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

parser.add_argument('--train_file', default='train_list.txt', type=str)   #TODO cutmix : data_cutmix.txt

parser.add_argument('--valid_file', default='test_list.txt', type=str)    

parser.add_argument('--test_file', default='test_list.txt', type=str)

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
parser.add_argument('--lr', default=0.00001, type=float)

parser.add_argument('--weight_decay', default=1e-3, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1,2,3,4,5', type=str)

parser.add_argument('--num_workers', default=8, type=int)  #ORIGINAL 8

parser.add_argument('--batch_size', default=6, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=200, type=int)

parser.add_argument('--save_freq', default=20, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
# args.local_rank = int(os.environ['local_rank'])

parser.add_argument('--tao_u', default=0.1, type=float)    

parser.add_argument('--tao_c', default=0.9, type=float)   

args = parser.parse_args()


# TODO: core code for training and validation
def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    # TODO: code for distributed training
    # torch.distributed supports three built-in backends, each with different capabilities
    # 'nccl' for GPU https://pytorch.org/docs/stable/distributed.html
    torch.distributed.init_process_group('nccl')
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.barrier()
    
    torch.manual_seed(args.seed+rank)  #CPU随机种子
    torch.cuda.manual_seed(args.seed+rank)  #CUDA（GPU）随机种子
    random.seed(args.seed+rank)   #Python内置的random库的随机种子
    np.random.seed(args.seed+rank)   #numpy的随机种子

    # TODO: load model from here (Create instance)
    # here three modules are loaded, all of which plays different roles
    # find the model, this should be a callback
    head_model = Unet_head()
    head_model = head_model.cuda(args.local_rank)
    model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    seg_model = Decoder_modual()
    IDH_model = IDH_network()
    # Loss function selection
    criterion = getattr(criterions, args.criterion)    # args.criterion = softmax_dice
    idh_criterion = getattr(criterions,'idh_cross_entropy')  # idh_focal_loss, idh_cross_entropy
    grade_criterion = getattr(criterions,'grade_cross_entropy')
    # criterion = FocalLoss_seg()
    MTL = MultiTaskLossWrapper(3, loss_fn=[criterion, idh_criterion, grade_criterion])  # 3 is the number of tasks
    # TODO: This dict is created for BatchNorm Synchronization (As different processes runs different batches, so a synchronization is necessary)
    nets = {
        'head': torch.nn.SyncBatchNorm.convert_sync_batchnorm(head_model).cuda(args.local_rank),
        'en': torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.local_rank),
        'seg': torch.nn.SyncBatchNorm.convert_sync_batchnorm(seg_model).cuda(args.local_rank),
        'idh': torch.nn.SyncBatchNorm.convert_sync_batchnorm(IDH_model).cuda(args.local_rank),
        # MTL module dose NOT contain batchnorm layer, no convert is applied to it
        'mtl': MTL.cuda(args.local_rank)
    }


    param = [p for v in nets.values() for p in list(v.parameters())]
   
    DDP_model = {
        'head': nn.parallel.DistributedDataParallel(nets['head'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True),
        'en': nn.parallel.DistributedDataParallel(nets['en'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True),
        'seg': nn.parallel.DistributedDataParallel(nets['seg'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True),
        'idh': nn.parallel.DistributedDataParallel(nets['idh'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True),
        'mtl': nn.parallel.DistributedDataParallel(nets['mtl'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    }

    # TODO: Optimizer selection
    optimizer = torch.optim.AdamW(param, lr=args.lr, weight_decay=args.weight_decay)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.exists("/public/home/hpc226511030/GMMAS_transfer_2/logs"):
            shutil.rmtree("/public/home/hpc226511030/GMMAS_transfer_2/logs")
        # todo SummaryWriter会自动创建一个文件夹
        writer = SummaryWriter("/public/home/hpc226511030/GMMAS_transfer_2/logs")

    resume_head = '/public/home/hpc226511030/GMMAS_transfer/checkpoint/GMMAS2024-09-08/model_epoch_bestSeg_10.pth'
    resume = '/public/home/hpc226511030/GMMAS_after_revise/checkpoint-simsiam_finetune/TransBraTS_IDH2024-08-08/model_epoch_mAX_ACC_2.pth'

    # TODO: Load saved parameters
    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        checkpoint_head = torch.load(resume_head, map_location=lambda storage, loc: storage)
        
        unet_state_dict = {k.replace('module.',''):v for k,v in checkpoint_head['en_state_dict'].items()}
        middle_state_dict = {k.replace('module.',''):v for k,v in checkpoint_head['middle_state_dict'].items()}
        decoder_state_dict = {k.replace('module.',''):v for k,v in checkpoint_head['decoder_state_dict'].items()}
        
        DDP_model['head'].module.UNET.load_state_dict(unet_state_dict)
        DDP_model['head'].module.MIDDLE_ONE.load_state_dict(middle_state_dict)
        DDP_model['head'].module.DECODER.load_state_dict(decoder_state_dict)
        # DDP_model['head'].load_state_dict(checkpoint['head_state_dict'])
        
        DDP_model['en'].load_state_dict(checkpoint['en_state_dict'], strict=False)
        DDP_model['seg'].load_state_dict(checkpoint['seg_state_dict'], strict=False)
        DDP_model['idh'].load_state_dict(checkpoint['idh_state_dict'], strict=False)
        # todo load optimizer 注意换了优化器种类后的加载优化器会导致不匹配
        # optimizer.load_state_dict(checkpoint['optim_dict'])
        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, args.mode, idh_file='', grade_file='', pq_file='')     #TODO 在这里添加半监督file
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=args.local_rank, shuffle=True)
    logging.info('Samples for train = {}'.format(len(train_set)))

    num_gpu = (len(args.gpu)+1) // 2
    train_loader = DataLoader(dataset= train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                    drop_last=True, num_workers=args.num_workers, pin_memory=True)
    
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, 'valid')
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info('Samples for valid = {}'.format(len(valid_set)))
    
    start_time = time.time()
    torch.set_grad_enabled(True)

    best_seg_epoch = 0
    best_IDH_epoch = 0
    best_grade_epoch = 0

    min_loss = 100.0
    min_seg_loss = 100
    min_IDH_loss = 100
    min_grade_loss = 100
    min_acc = 0.0
    min_pq_acc = 0.0
    min_3_acc = 0.0

    def Dice(output,target, eps=1e-5):
        target = target.float()
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
        return 1.0 - num/den

    def softmax_dice(output, target): 
        '''
        The dice loss for using softmax activation function
        :param output: (b, num_class, d, h, w)
        :param target: (b, d, h, w)
        :return: softmax dice loss
        '''
        Dice_background = Dice(output[:, 0, ...], (target == 0).float())
        # TODO 计算各区域的Dice
        Dice_ncr = Dice(output[:, 1, ...], (target == 1).float())
        Dice_ed = Dice(output[:, 2, ...], (target == 2).float())
        Dice_et = Dice(output[:, 3, ...], (target == 4).float())
        
        return Dice_ncr + Dice_ed + Dice_et, 1 - Dice_background.data, 1 - Dice_ncr.data, 1 - Dice_ed.data, 1 - Dice_et.data

    for epoch in range(args.start_epoch, args.end_epoch):
        DDP_model['head'].train()    # 不涉及梯度开关，开启dropout和BN
        DDP_model['en'].train()
        DDP_model['seg'].train()
        DDP_model['idh'].train()
        DDP_model['mtl'].train()
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()

        epoch_train_loss = 0.0
        epoch_train_seg_loss = 0.0
        epoch_train_idh_loss = 0.0
        epoch_train_grade_loss = 0.0
        epoch_train_mgmt_loss = 0.0
        epoch_train_pq_loss = 0.0

        epoch_uncertainty_seg = 0.0
        epoch_uncertainty_idh = 0.0
        epoch_uncertainty_grade = 0.0
        epoch_uncertainty_mgmt = 0.0
        epoch_uncertainty_pq = 0.0


        for i, data in enumerate(train_loader):
            # break
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            optimizer.zero_grad()
            
            x_head, target_head, x, target, weight_seg, grade,idh, mgmt, _1p19q = data
            # print('1',target.shape)
            # print('1',weight_seg.shape)
            x_head = x_head.cuda(args.local_rank, non_blocking=True)
            target_head = target_head.cuda(args.local_rank, non_blocking=True)
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            weight_seg = weight_seg.cuda(args.local_rank, non_blocking=True)

            grade = grade.cuda(args.local_rank, non_blocking=True)
            idh = idh.cuda(args.local_rank, non_blocking=True)
            mgmt = mgmt.cuda(args.local_rank, non_blocking=True)
            _1p19q = _1p19q.cuda(args.local_rank, non_blocking=True)
            #MU:57-label_1, WT:91-label_0  #HGG:294-label_1,LGG:77-label_0
            weight_idh = torch.tensor([112, 245]).float().cuda(args.local_rank, non_blocking=True)  
            weight_grade = torch.tensor([390, 292]).float().cuda(args.local_rank, non_blocking=True) 
            # TODO : COUNT
            weight_mgmt = torch.tensor([337,320]).float().cuda(args.local_rank, non_blocking=True)  
            weight_pq = torch.tensor([128,166]).float().cuda(args.local_rank, non_blocking=True) 

            y_ROI, x_middle, y_output, y4_1, y3_1, y2_1 = DDP_model['head'](x_head)  
            x1_1, x2_1, x3_1,x4_1, encoder_output, weights_x, decoder_input = DDP_model['en'](x, y_ROI, x_middle)
            y = DDP_model['seg'](x1_1, x2_1, x3_1,decoder_input, y4_1, y3_1, y2_1)  #y: (1, 4, 128, 128, 128)
            idh_out, grade_out,mgmt_out,pq_out = DDP_model['idh'](x4_1, decoder_input, idh, grade, mgmt, _1p19q) 

            loss, seg_loss, idh_loss, grade_loss, mgmt_loss, pq_loss, Dice_0, Dice_NCR, Dice_ED, Dice_ET, uncertainty_seg, uncertainty_idh, \
            uncertainty_grade, uncertainty_mgmt, uncertainty_pq = DDP_model['mtl']([y,idh_out,grade_out,mgmt_out,pq_out], \
            [target,idh,grade,mgmt,_1p19q],[weight_seg,weight_idh,weight_grade,weight_mgmt,weight_pq], [ 1, 1, 1, 1, 1])
            
            if torch.all((weight_seg > 0.99) & (weight_seg < 1.01)):    
                seg_loss_head, Dice_0_head, Dice_NCR_head, Dice_ED_head, Dice_ET_head = tuple(x * 0 for x in softmax_dice(y_output, target_head))
            else:
                seg_loss_head, Dice_0_head, Dice_NCR_head, Dice_ED_head, Dice_ET_head = softmax_dice(y_output, target_head)
            loss += seg_loss_head
            
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            # reduce_idh_loss = all_reduce_tensor(idh_loss, world_size=num_gpu).data.cpu().numpy()
            reduce_seg_loss = seg_loss.data.cpu().numpy()
            reduce_idh_loss = idh_loss.data.cpu().numpy()
            reduce_grade_loss = grade_loss.data.cpu().numpy()  
            reduce_mgmt_loss = mgmt_loss.data.cpu().numpy()
            reduce_pq_loss = pq_loss.data.cpu().numpy()

            reduce_uncertainty_seg = all_reduce_tensor(uncertainty_seg, world_size=num_gpu).data.cpu().numpy()
            reduce_uncertainty_idh = all_reduce_tensor(uncertainty_idh, world_size=num_gpu).data.cpu().numpy()
            reduce_uncertainty_grade = all_reduce_tensor(uncertainty_grade, world_size=num_gpu).data.cpu().numpy()
            reduce_uncertainty_mgmt = all_reduce_tensor(uncertainty_mgmt, world_size=num_gpu).data.cpu().numpy()
            reduce_uncertainty_pq = all_reduce_tensor(uncertainty_pq, world_size=num_gpu).data.cpu().numpy()

            epoch_train_loss += reduce_loss/len(train_loader)
            epoch_train_seg_loss += reduce_seg_loss/len(train_loader)
            epoch_train_idh_loss += reduce_idh_loss/len(train_loader)
            epoch_train_grade_loss += reduce_grade_loss/len(train_loader)
            epoch_train_mgmt_loss += reduce_mgmt_loss/len(train_loader)
            epoch_train_pq_loss += reduce_pq_loss/len(train_loader)

            epoch_uncertainty_seg += reduce_uncertainty_seg/len(train_loader)
            epoch_uncertainty_idh += reduce_uncertainty_idh/len(train_loader)
            epoch_uncertainty_grade += reduce_uncertainty_grade/len(train_loader)
            epoch_uncertainty_mgmt += reduce_uncertainty_mgmt/len(train_loader)
            epoch_uncertainty_pq += reduce_uncertainty_pq/len(train_loader)

            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter:{} loss: {:.5f} '.format(epoch, i, reduce_loss))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

        if args.local_rank == 0:
            logging.info('Epoch: {} epoch_loss: {:.5f} '.format(epoch, epoch_train_loss))

        idh_probs = []
        idh_class = []
        idh_target = []

        grade_probs = []
        grade_class = []
        grade_target = []

        mgmt_probs = []
        mgmt_class = []
        mgmt_target = []

        pq_probs = []
        pq_class = []
        pq_target = []

        with torch.no_grad():
            DDP_model['head'].eval()
            DDP_model['en'].eval() 
            DDP_model['seg'].eval()
            DDP_model['idh'].eval()
            DDP_model['mtl'].eval()
            
            epoch_valid_loss = 0.0
            epoch_seg_loss = 0.0
            epoch_idh_loss = 0.0
            epoch_grade_loss = 0.0
            epoch_mgmt_loss = 0.0
            epoch_pq_loss = 0.0

            epoch_flair_weight = 0.0
            epoch_ce_weight = 0.0
            epoch_t1_weight = 0.0
            epoch_t2_weight = 0.0
            
            epoch_dice_0 = 0.0
            epoch_dice_NCR = 0.0
            epoch_dice_ED = 0.0
            epoch_dice_ET = 0.0

            epoch_dice_0_head = 0.0
            epoch_dice_NCR_head = 0.0
            epoch_dice_ED_head = 0.0
            epoch_dice_ET_head = 0.0
            for i, data in enumerate(valid_loader):

                x_head, target_head, x, target,weight_seg, grade,idh, mgmt, _1p19q = data
                x_head = x_head.cuda(args.local_rank, non_blocking=True)
                target_head = target_head.cuda(args.local_rank, non_blocking=True)
                x = x.cuda(args.local_rank, non_blocking=True)
                target = target.cuda(args.local_rank, non_blocking=True)
                weight_seg = weight_seg.cuda(args.local_rank, non_blocking=True)

                grade = grade.cuda(args.local_rank, non_blocking=True)
                idh = idh.cuda(args.local_rank, non_blocking=True)
                mgmt = mgmt.cuda(args.local_rank, non_blocking=True)
                _1p19q = _1p19q.cuda(args.local_rank, non_blocking=True)

                weight_idh = torch.tensor([112, 245]).float().cuda(args.local_rank, non_blocking=True)  
                weight_grade = torch.tensor([390, 292]).float().cuda(args.local_rank, non_blocking=True) 
                # TODO : COUNT
                weight_mgmt = torch.tensor([337,320]).float().cuda(args.local_rank, non_blocking=True)  
                weight_pq = torch.tensor([128,166]).float().cuda(args.local_rank, non_blocking=True) 

                y_ROI, x_middle, y_output, y4_1, y3_1, y2_1 = DDP_model['head'](x_head)  
                x1_1, x2_1, x3_1,x4_1, encoder_output, weights_x, decoder_input = DDP_model['en'](x, y_ROI, x_middle)
                y = DDP_model['seg'](x1_1, x2_1, x3_1,decoder_input, y4_1, y3_1, y2_1)  #y: (1, 4, 128, 128, 128)
                idh_out, grade_out,mgmt_out,pq_out = DDP_model['idh'](x4_1, decoder_input, idh, grade, mgmt, _1p19q)  

                valid_loss, seg_loss, idh_loss, grade_loss, mgmt_loss, pq_loss, Dice_0, Dice_NCR, Dice_ED, Dice_ET, uncertainty_seg, uncertainty_idh,  \
                uncertainty_grade, uncertainty_mgmt, uncertainty_pq = DDP_model['mtl']([y,idh_out,grade_out,mgmt_out,pq_out],  \
                [target,idh,grade,mgmt,_1p19q],[weight_seg,weight_idh,weight_grade,weight_mgmt,weight_pq], [ 1, 1, 1, 1, 1])
                
                if torch.all((weight_seg > 0.99) & (weight_seg < 1.01)):    
                    seg_loss_head, Dice_0_head, Dice_NCR_head, Dice_ED_head, Dice_ET_head = tuple(x * 0 for x in softmax_dice(y_output, target_head))
                else:
                    seg_loss_head, Dice_0_head, Dice_NCR_head, Dice_ED_head, Dice_ET_head = softmax_dice(y_output, target_head)
                
                valid_loss += seg_loss_head

                flair_weight = torch.exp(weights_x[0]) ** 0.5
                ce_weight = torch.exp(weights_x[1]) ** 0.5
                t1_weight = torch.exp(weights_x[2]) ** 0.5
                t2_weight = torch.exp(weights_x[3]) ** 0.5

                epoch_flair_weight += flair_weight / len(valid_loader)
                epoch_ce_weight += ce_weight / len(valid_loader)
                epoch_t1_weight += t1_weight / len(valid_loader)
                epoch_t2_weight += t2_weight / len(valid_loader)

                epoch_valid_loss += valid_loss / len(valid_loader)
                epoch_seg_loss += seg_loss
                epoch_idh_loss += idh_loss
                epoch_grade_loss += grade_loss
                epoch_mgmt_loss += mgmt_loss
                epoch_pq_loss += pq_loss

                epoch_uncertainty_seg += uncertainty_seg / len(valid_loader)
                epoch_uncertainty_idh += uncertainty_idh / len(valid_loader)
                epoch_uncertainty_grade += uncertainty_grade / len(valid_loader)
                epoch_uncertainty_mgmt += uncertainty_mgmt / len(valid_loader)
                epoch_uncertainty_pq += uncertainty_pq / len(valid_loader)

                epoch_dice_0 += Dice_0 / 48  #验证集中有分割target的数量
                epoch_dice_NCR += Dice_NCR / 48
                epoch_dice_ED += Dice_ED / 48
                epoch_dice_ET += Dice_ET / 48

                epoch_dice_0_head += Dice_0_head / 48
                epoch_dice_NCR_head += Dice_NCR_head / 48
                epoch_dice_ED_head += Dice_ED_head / 48
                epoch_dice_ET_head += Dice_ET_head / 48

                if idh.item() != -1:
                    idh_pred = F.softmax(idh_out, 1)              #todo 这里可以直接加temprrature系数 此处为概率
                    idh_pred_class = torch.argmax(idh_pred, dim=1)
                    idh_probs.append(idh_pred[0][1].cpu())
                    idh_class.append(idh_pred_class.item())
                    idh_target.append(idh.item())

                if grade.item() != -1:
                    grade_pred = F.softmax(grade_out, 1)              
                    grade_pred_class = torch.argmax(grade_pred, dim=1) 
                    grade_probs.append(grade_pred[0][1].cpu())
                    grade_class.append(grade_pred_class.item())
                    grade_target.append(grade.item())

                if mgmt.item() != -1:
                    mgmt_pred = F.softmax(mgmt_out, 1)              
                    mgmt_pred_class = torch.argmax(mgmt_pred, dim=1) 
                    mgmt_probs.append(mgmt_pred[0][1].cpu())
                    mgmt_class.append(mgmt_pred_class.item())
                    mgmt_target.append(mgmt.item())
                
                if _1p19q.item() != -1:
                    pq_pred = F.softmax(pq_out, 1)              
                    pq_pred_class = torch.argmax(pq_pred, dim=1) 
                    pq_probs.append(pq_pred[0][1].cpu())
                    pq_class.append(pq_pred_class.item())
                    pq_target.append(_1p19q.item())
                if args.local_rank == 0:
                    logging.info('valid_Epoch:{}_Iter:{} valid_loss: {:.5f} '.format(epoch, i, valid_loss))

            logging.info('valid_Epoch:{} epoch_valid_loss: {:.5f} '.format(epoch, epoch_valid_loss))

            accuracy_idh = accuracy_score(idh_target,idh_class)
            auc_idh = roc_auc_score(idh_target,idh_probs)

            accuracy_grade = accuracy_score(grade_target,grade_class)
            auc_grade = roc_auc_score(grade_target,grade_probs)

            accuracy_mgmt = accuracy_score(mgmt_target,mgmt_class)
            auc_mgmt = roc_auc_score(mgmt_target,mgmt_probs)

            accuracy_pq = accuracy_score(pq_target,pq_class)
            auc_pq = roc_auc_score(pq_target,pq_probs)

            if args.local_rank == 0:
                if min_acc < accuracy_idh + accuracy_grade:
                    min_acc = accuracy_idh + accuracy_grade
                    best_acc_epoch = epoch
                    logging.info('there is an improvement on TMF ACC.')

                    file_name = os.path.join(checkpoint_dir, 'model_epoch_mAX_ACC.pth')
                    torch.save({
                        'epoch': epoch,
                        'head_state_dict': DDP_model['head'].state_dict(),
                        'en_state_dict': DDP_model['en'].state_dict(),
                        'seg_state_dict': DDP_model['seg'].state_dict(),
                        'idh_state_dict': DDP_model['idh'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)
                
                if min_pq_acc < accuracy_pq:
                    min_pq_acc = accuracy_pq
                    best_pq_epoch = epoch
                    logging.info('there is an improvement on PQ ACC.')

                    file_name = os.path.join(checkpoint_dir, 'model_epoch_pq_ACC.pth')
                    torch.save({
                        'epoch': epoch,
                        'head_state_dict': DDP_model['head'].state_dict(),
                        'en_state_dict': DDP_model['en'].state_dict(),
                        'seg_state_dict': DDP_model['seg'].state_dict(),
                        'idh_state_dict': DDP_model['idh'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)

                if min_3_acc < accuracy_pq + accuracy_idh + accuracy_grade:
                    min_3_acc = accuracy_pq + accuracy_idh + accuracy_grade
                    best_3_epoch = epoch
                    logging.info('there is an improvement on 3 ACC.')

                    file_name = os.path.join(checkpoint_dir, 'model_3max_ACC.pth')
                    torch.save({
                        'epoch': epoch,
                        'head_state_dict': DDP_model['head'].state_dict(),
                        'en_state_dict': DDP_model['en'].state_dict(),
                        'seg_state_dict': DDP_model['seg'].state_dict(),
                        'idh_state_dict': DDP_model['idh'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)

                logging.info('Epoch:{} best_epoch:{} epoch_ACC:{:.5f} '.format(epoch, best_acc_epoch, accuracy_idh + accuracy_grade))

                if min_loss >= epoch_valid_loss:
                    min_loss = epoch_valid_loss
                    best_epoch = epoch
                    logging.info('there is an improvement that update the metrics and save the best model.')

                    file_name = os.path.join(checkpoint_dir, 'model_epoch_minloss.pth')
                    torch.save({
                        'epoch': epoch,
                        'head_state_dict': DDP_model['head'].state_dict(),
                        'en_state_dict': DDP_model['en'].state_dict(),
                        'seg_state_dict': DDP_model['seg'].state_dict(),
                        'idh_state_dict': DDP_model['idh'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)

                logging.info('Epoch:{} best_epoch:{} epoch_valid_loss:{:.5f} '.format(epoch, best_epoch, epoch_valid_loss))

        end_epoch = time.time()
        if args.local_rank == 0:
            if epoch % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 :
                    # or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    # or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'head_state_dict': DDP_model['head'].state_dict(),
                    'en_state_dict': DDP_model['en'].state_dict(),
                    'seg_state_dict': DDP_model['seg'].state_dict(),
                    'idh_state_dict': DDP_model['idh'].state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            # writer.add_scalars('lr', {"lr": optimizer.param_groups[0]['lr']}, epoch)
            writer.add_scalars('loss',{"epoch_train_loss": epoch_train_loss},epoch)
            writer.add_scalars('loss', {"epoch_valid_loss": epoch_valid_loss}, epoch)

            writer.add_scalars('seg_loss:', {"seg_train_loss": epoch_train_seg_loss}, epoch)
            writer.add_scalars('idh_loss:',  {"idh_train_loss": epoch_train_idh_loss}, epoch)
            writer.add_scalars('grade_loss:',  {"grade_train_loss": epoch_train_grade_loss}, epoch)
            writer.add_scalars('mgmt_loss:',  {"mgmt_train_loss": epoch_train_mgmt_loss}, epoch)
            writer.add_scalars('pq_loss:',  {"pq_train_loss": epoch_train_pq_loss}, epoch)

            writer.add_scalars('Uncertainty:',  {"uncertainty_seg": epoch_uncertainty_seg}, epoch)
            writer.add_scalars('Uncertainty:',  {"uncertainty_idh": epoch_uncertainty_idh}, epoch)
            writer.add_scalars('Uncertainty:',  {"uncertainty_grade": epoch_uncertainty_grade}, epoch)
            writer.add_scalars('Uncertainty:',  {"uncertainty_mgmt": epoch_uncertainty_mgmt}, epoch)
            writer.add_scalars('Uncertainty:',  {"uncertainty_pq": epoch_uncertainty_pq}, epoch)

            writer.add_scalars('weight-x:',  {"epoch_t2_weight": epoch_t2_weight}, epoch)
            writer.add_scalars('weight-x:',  {"epoch_t1_weight": epoch_t1_weight}, epoch)
            writer.add_scalars('weight-x:',  {"epoch_ce_weight": epoch_ce_weight}, epoch)
            writer.add_scalars('weight-x:',  {"epoch_flair_weight": epoch_flair_weight}, epoch)

            writer.add_scalars('seg_loss:', {"seg_valid_loss": epoch_seg_loss},epoch)
            writer.add_scalars('idh_loss:', {"idh_valid_loss": epoch_idh_loss}, epoch)
            writer.add_scalars('grade_loss:', {"grade_valid_loss": epoch_grade_loss}, epoch)
            writer.add_scalars('mgmt_loss:', {"mgmt_valid_loss": epoch_mgmt_loss}, epoch)
            writer.add_scalars('pq_loss:', {"pq_valid_loss": epoch_pq_loss}, epoch)

            writer.add_scalars('auc_IDH:', {"auc": auc_idh}, epoch)
            writer.add_scalars('accuracy_IDH:', {"accuracy": accuracy_idh}, epoch)
            writer.add_scalars('auc_grade:', {"auc": auc_grade}, epoch)
            writer.add_scalars('accuracy_grade:', {"accuracy": accuracy_grade}, epoch)
            writer.add_scalars('auc_mgmt:', {"auc": auc_mgmt}, epoch)
            writer.add_scalars('accuracy_mgmt:', {"accuracy": accuracy_mgmt}, epoch)
            writer.add_scalars('auc_pq:', {"auc": auc_pq}, epoch)
            writer.add_scalars('accuracy_pq:', {"accuracy": accuracy_pq}, epoch)

            writer.add_scalars('dice:', {"dice_0": epoch_dice_0}, epoch)
            writer.add_scalars('dice:', {"dice_NCR": epoch_dice_NCR}, epoch)
            writer.add_scalars('dice:', {"dice_ED": epoch_dice_ED}, epoch)
            writer.add_scalars('dice:', {"dice_ET": epoch_dice_ET}, epoch)

            writer.add_scalars('dice_head:', {"dice_0": epoch_dice_0_head}, epoch)
            writer.add_scalars('dice_head:', {"dice_NCR": epoch_dice_NCR_head}, epoch)
            writer.add_scalars('dice_head:', {"dice_ED": epoch_dice_ED_head}, epoch)
            writer.add_scalars('dice_head:', {"dice_ET": epoch_dice_ET_head}, epoch)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        writer.close()
        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'head_state_dict': DDP_model['head'].state_dict(),
            'en_state_dict': DDP_model['en'].state_dict(),
            'seg_state_dict': DDP_model['seg'].state_dict(),
            'idh_state_dict': DDP_model['idh'].state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
