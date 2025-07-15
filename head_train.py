# Function: head_train.py
import shutil
import argparse
import os
import random
import logging
# 设置基本的配置
logging.basicConfig(level=logging.INFO, format=' %(message)s')

# 可以加上这行确保能看到INFO级别的输出
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import setproctitle
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS,Decoder_modual,IDH_network
from models.TransBraTS.head import Unet, Decoder_modual, middle_one
import torch.distributed as dist
from models import criterions
from contextlib import nullcontext
from data.BraTS_IDH import BraTS                      # TODO 此处修改dataset
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from utils.pcgrad import PCGrad
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from models.criterions import MultiTaskLossWrapper
import utils.train_util as pseudo
from sklearn.metrics import roc_auc_score,accuracy_score
import time

parser = argparse.ArgumentParser()
# DataSet Information
parser.add_argument('--experiment', default='GMMAS', type=str)

local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)


parser.add_argument('--root', default='/public/home/hpc226511030/Archive/BraTS2021_TrainingData/', type=str)

parser.add_argument('--train_dir', default='MICCAI_BraTS2021_TrainingData', type=str)

parser.add_argument('--valid_dir', default='MICCAI_BraTS2021_TrainingData', type=str)

parser.add_argument('--test_dir', default='MICCAI_BraTS2021_TrainingData', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train_head.txt', type=str)   #TODO cutmix : data_cutmix.txt

parser.add_argument('--valid_file', default='test_head.txt', type=str)    

parser.add_argument('--test_file', default='test_head.txt', type=str)

parser.add_argument('--dataset', default='brats_IDH', type=str)

parser.add_argument('--model_name', default='TransBraTS', type=str)

# Training Information
parser.add_argument('--lr', default=0.0001, type=float)

parser.add_argument('--weight_decay', default=1e-2, type=float)

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--gpu', default='0,1,2,3', type=str)

parser.add_argument('--num_workers', default=8, type=int)  #ORIGINAL 8

parser.add_argument('--batch_size', default=16, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=200, type=int)

parser.add_argument('--save_freq', default=20, type=int)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()

def main_worker():
    def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epoch, power), 8)

    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
        log_file = log_dir + '.txt'
        logging.info('----------------------------------------This is a halving line----------------------------------')

    # TODO: code for distributed training
    torch.distributed.init_process_group('nccl')
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    # torch.distributed.barrier()
    
    torch.manual_seed(args.seed+rank)  #CPU随机种子
    torch.cuda.manual_seed(args.seed+rank)  #CUDA（GPU）随机种子
    random.seed(args.seed+rank)   #Python内置的random库的随机种子
    np.random.seed(args.seed+rank)   #numpy的随机种子

    # TODO: load model from here (Create instance)
    head_model = Unet(in_channels=4, base_channels=16, num_classes=4)
    middle_model = middle_one()
    decoder_model = Decoder_modual()
    head_model = head_model.cuda(args.local_rank)
    middle_model = middle_model.cuda(args.local_rank)
    decoder_model = decoder_model.cuda(args.local_rank)
    
    head_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(head_model).cuda(args.local_rank)
    middle_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(middle_model).cuda(args.local_rank)
    decoder_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder_model).cuda(args.local_rank)
    nets = {'encoder':head_net, 'middle':middle_net, 'decoder':decoder_net}
    DDP_model = {'encoder':torch.nn.parallel.DistributedDataParallel(head_net, device_ids=[args.local_rank], output_device=args.local_rank),\
                'middle':torch.nn.parallel.DistributedDataParallel(middle_net, device_ids=[args.local_rank], output_device=args.local_rank),\
                'decoder':torch.nn.parallel.DistributedDataParallel(decoder_net, device_ids=[args.local_rank], output_device=args.local_rank)}
    

    param = [p for v in nets.values() for p in list(v.parameters())]
    optimizer = torch.optim.AdamW(param, lr=args.lr, weight_decay=args.weight_decay)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint_160', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.exists("/public/home/hpc226511030/GMMAS_transfer/logs_160"):
            shutil.rmtree("/public/home/hpc226511030/GMMAS_transfer/logs_160")
        # todo SummaryWriter会自动创建一个文件夹
        writer = SummaryWriter("/public/home/hpc226511030/GMMAS_transfer/logs_160")

    # todo load checkpoint
    resume = '/public/home/hpc226511030/GMMAS_transfer/checkpoint/GMMAS2024-09-07/THIRD/model_epoch_bestSeg_3.pth'
    logging.info('loading checkpoint {}'.format(resume))
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    # unet_state_dict = {key.replace('module.Unet', 'module'): value for key, value in checkpoint['en_state_dict'].items() if key.startswith('module.Unet')}
    # DDP_model['encoder'].load_state_dict(unet_state_dict)
    DDP_model['encoder'].load_state_dict(checkpoint['en_state_dict'])
    DDP_model['middle'].load_state_dict(checkpoint['middle_state_dict'])
    DDP_model['decoder'].load_state_dict(checkpoint['decoder_state_dict'])
    
    # todo load optimizer 注意换了优化器种类后的加载优化器会导致不匹配
    # optimizer.load_state_dict(checkpoint['optim_dict'])
    logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                    .format(resume, args.start_epoch))

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, args.mode, idh_file='', grade_file='', pq_file='')     #TODO 在这里添加半监督file
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=args.local_rank, shuffle=True)

    num_gpu = (len(args.gpu)+1) // 2
    train_loader = DataLoader(dataset= train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                    drop_last=True, num_workers=args.num_workers, pin_memory=True)
    
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, 'valid')
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size // num_gpu, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # 验证集为单卡进行，所以没有sampler，没用归约
    torch.set_grad_enabled(True)

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
        
        return Dice_ncr + Dice_ed + Dice_et, 1 - Dice_ncr.data, 1 - Dice_ed.data, 1 - Dice_et.data, 1 - Dice_background.data

    def focal_loss(outputs, targets, alpha=0.25, gamma=2):
        '''
        The focal loss for using softmax activation function
        :param outputs: (b, num_class, d, h, w) after softmax
        :param targets: (b, d, h, w) with class indices (0 to num_class-1)
        :return: softmax focal loss
        '''
        # Assume outputs are probabilities (after softmax)
        # Convert targets to one-hot format
        targets[targets == 4] = 3
        batch_size, num_classes, depth, height, width = outputs.size()
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        # Calculate log probabilities
        log_probs = torch.log(outputs + 1e-6)  # Adding a small number to avoid log(0)
        
        # Compute the loss
        focal_term = (1 - outputs)**gamma
        # print('1',targets_one_hot.shape)
        # print('2',focal_term.shape)
        # print('3',log_probs.shape)
        loss = -alpha * targets_one_hot * focal_term * log_probs
        
        # Reduce loss: mean over all the elements
        return loss.mean()

    min_loss = 1000
    max_dice = 0.0
    for epoch in range(200):
        train_loss = 0.0
        DDP_model['encoder'].train()
        DDP_model['middle'].train()
        DDP_model['decoder'].train()
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}/{}'.format(epoch+1, args.end_epoch))

        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            optimizer.zero_grad()
            x_head, target_head, x, target,weight_seg, grade,idh, mgmt, _1p19q = data
            x_head = x_head.cuda(args.local_rank, non_blocking=True)
            target_head = target_head.cuda(args.local_rank, non_blocking=True)

            x1_1,x2_1,x3_1,output,InitConv_vars = DDP_model['encoder'](x_head)  # 通过模型前向传播得到输出
            output = DDP_model['middle'](output)
            y = DDP_model['decoder'](x1_1, x2_1, x3_1,output)

            loss, Dice_ncr_train, Dice_ed_train, Dice_et_train, Dice_background_train = softmax_dice(y, target_head)  # 计算损失函数
            # ce_loss = focal_loss(y, target_head)
            # loss = loss + ce_loss
            loss.backward()  # 损失的反向传播，计算参数更新值
            optimizer.step()  # 将参数更新值施加到model的parameters上
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()

            train_loss += reduce_loss / len(train_loader)
            Dice_ncr_train += Dice_ncr_train / len(train_loader)
            Dice_ed_train += Dice_ed_train / len(train_loader)
            Dice_et_train += Dice_et_train / len(train_loader)
            Dice_background_train += Dice_background_train / len(train_loader)

            if args.local_rank == 0:
                logging.info(f'Epoch {epoch + 1}_iter {i} , iter_loss: {reduce_loss:.4f}')
            # break
        if args.local_rank == 0:
            logging.info(f'Epoch {epoch + 1}, train_loss: {train_loss:.4f}, \
            Dice_ncr: {Dice_ncr_train:.4f}, Dice_ed: {Dice_ed_train:.4f}, Dice_et: {Dice_et_train:.4f}, Dice_background: {Dice_background_train:.4f}')

        valid_loss = 0.0
        with torch.no_grad():
            DDP_model['encoder'].eval()
            DDP_model['middle'].eval()
            DDP_model['decoder'].eval()
            for i, data in enumerate(valid_loader):
                x_head, target_head, x, target,weight_seg, grade,idh, mgmt, _1p19q = data
                x_head = x_head.cuda(args.local_rank, non_blocking=True)
                target_head = target_head.cuda(args.local_rank, non_blocking=True)
                x1_1,x2_1,x3_1,output,InitConv_vars = DDP_model['encoder'](x_head)  # 通过模型前向传播得到输出
                output = DDP_model['middle'](output)
                y = DDP_model['decoder'](x1_1, x2_1, x3_1,output)
                loss, Dice_ncr, Dice_ed, Dice_et, Dice_background = softmax_dice(y, target_head)  # 计算损失函数

                valid_loss += loss / len(valid_loader)
                Dice_ncr += Dice_ncr / len(valid_loader)
                Dice_ed += Dice_ed / len(valid_loader)
                Dice_et += Dice_et / len(valid_loader)
                Dice_background += Dice_background / len(valid_loader)
                if args.local_rank == 0:
                    logging.info(f'valid Epoch {epoch + 1}_iter {i} , iter_loss: {loss:.4f}')
                # break
        if args.local_rank == 0:
            logging.info(f'Epoch {epoch + 1}, Valid_Loss: {valid_loss:.4f}, \
            Dice_ncr: {Dice_ncr:.4f}, Dice_ed: {Dice_ed:.4f}, Dice_et: {Dice_et:.4f}, Dice_background: {Dice_background:.4f}')

        if args.local_rank == 0:
            writer.add_scalars('loss',{"epoch_train_loss": train_loss},epoch)
            writer.add_scalars('loss', {"epoch_valid_loss": valid_loss}, epoch)
            writer.add_scalars('dice_train', {"epoch_dice_ncr": Dice_ncr_train}, epoch)
            writer.add_scalars('dice_train', {"epoch_dice_ed": Dice_ed_train}, epoch)
            writer.add_scalars('dice_train', {"epoch_dice_et": Dice_et_train}, epoch)
            writer.add_scalars('dice_train', {"epoch_dice_background": Dice_background_train}, epoch)
            
            writer.add_scalars('dice', {"epoch_dice_ncr": Dice_ncr}, epoch)
            writer.add_scalars('dice', {"epoch_dice_ed": Dice_ed}, epoch)
            writer.add_scalars('dice', {"epoch_dice_et": Dice_et}, epoch)
            writer.add_scalars('dice', {"epoch_dice_background": Dice_background}, epoch)
        if args.local_rank == 0:
            if valid_loss < min_loss:
                min_loss = valid_loss
                logging.info('there is an improvement on validation loss.')
                file_name = os.path.join(checkpoint_dir, 'model_epoch_minLoss.pth')
                logging.info(f'Saving model to {file_name}')
                torch.save({
                        'epoch': epoch,
                        'en_state_dict': DDP_model['encoder'].state_dict(),
                        'middle_state_dict': DDP_model['middle'].state_dict(),
                        'decoder_state_dict': DDP_model['decoder'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)

            epoch_dice_sum = Dice_ncr + Dice_ed + Dice_et
            if epoch_dice_sum >= max_dice:
                max_dice = epoch_dice_sum
                best_seg_epoch = epoch
                logging.info('there is an improvement on segmentation.')
                file_name = os.path.join(checkpoint_dir, 'model_epoch_bestSeg.pth')
                logging.info(f'Saving model to {file_name}')
                torch.save({
                        'epoch': epoch,
                        'en_state_dict': DDP_model['encoder'].state_dict(),
                        'middle_state_dict': DDP_model['middle'].state_dict(),
                        'decoder_state_dict': DDP_model['decoder'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)

            if epoch % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 :
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                logging.info(f'Saving model to {file_name}')
                torch.save({
                        'epoch': epoch,
                        'en_state_dict': DDP_model['encoder'].state_dict(),
                        'middle_state_dict': DDP_model['middle'].state_dict(),
                        'decoder_state_dict': DDP_model['decoder'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)
            
    logging.info('Training completed')
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()