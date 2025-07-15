import argparse
import os
import time
import random
import numpy as np
import setproctitle
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader
import pandas as pd

from data.BraTS_IDH import BraTS
from predict import validate_softmax
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS, Decoder_modual, IDH_network
from models.TransBraTS.head import Unet_head


parser = argparse.ArgumentParser()

parser.add_argument('--user', default='name of user', type=str)

# parser.add_argument('--root', default='F:/DCA_test/Xiangya_lyh/preprocessed/', type=str)

# parser.add_argument('--valid_dir', default='F:/DCA_test/Xiangya_lyh/preprocessed/', type=str)

parser.add_argument('--valid_dir', default='C:/Users/86137/Desktop/GMMAS_last/pkl', type=str)

parser.add_argument('--valid_file', default='C:/Users/86137/Desktop/GMMAS_last/test_list.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='TransBraTS_IDH', type=str) #TransBraTS

parser.add_argument('--test_date', default='2023-11-11', type=str)

parser.add_argument('--test_file', default='model_epoch_mAX_ACC.pth', type=str) #model_epoch_last

# 自定义参数类型，将字符串转换为布尔值
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--use_TTA', default=False, type=str2bool)

parser.add_argument('--post_process', default=True, type=str2bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='TransBraTS', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=0, type=int)

# 添加控制测试样本数量的参数
parser.add_argument('--num_samples', default=0, type=int, help='测试样本数量，默认为0表示使用所有样本')

# 添加分类任务选择参数
parser.add_argument('--task', default='brats_test', type=str, choices=['all', 'IDH_status', 'Grade', 'MGMT', '1p19q','brats_test'], 
                    help='选择测试的分类任务，可选项: all, IDH_status, Grade, MGMT, 1p19q')

# 添加标签文件路径参数
parser.add_argument('--label_file', default='C:/Users/86137/Desktop/GMMAS_last/data_label_lyh.csv', type=str,
                    help='标签文件路径')

# 添加Grad-CAM相关参数
parser.add_argument('--generate_gradcam', default=False, type=str2bool,
                    help='是否生成Grad-CAM可视化，默认为False')
parser.add_argument('--gradcam_layer', default=None, type=str,
                    help='指定用于生成Grad-CAM的层，默认为None（使用最后一层）')

args = parser.parse_args()

# 打印参数，特别是use_TTA的值和类型
print(f"解析后的参数 use_TTA = {args.use_TTA} (类型: {type(args.use_TTA)})")

def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 初始化主干模型
    model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    seg_model = Decoder_modual()
    IDH_model = IDH_network()
    
    # 初始化head模块
    head_model = Unet_head()
    
    # 并行处理
    model = torch.nn.DataParallel(model).cuda()
    seg_model = torch.nn.DataParallel(seg_model).cuda()
    IDH_model = torch.nn.DataParallel(IDH_model).cuda()
    head_model = torch.nn.DataParallel(head_model).cuda()

    # 将所有模型组件添加到字典中
    dict_model = {
        'en': model, 
        'seg': seg_model,
        'idh': IDH_model,
        'head': head_model
    }
    
    load_file = Path('C:/Users/86137/Desktop/GMMAS_last/model_epoch_mAX_ACC.pth')
    
    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        
        # 1. 从预训练模型中加载backbone模型参数
        dict_model['en'].load_state_dict(checkpoint['en_state_dict'])
        dict_model['seg'].load_state_dict(checkpoint['seg_state_dict'])
        dict_model['idh'].load_state_dict(checkpoint['idh_state_dict'])
        
        # 2. 从预训练模型中加载head模块参数
        if 'head_state_dict' in checkpoint:
            # 直接从checkpoint加载head模型
            dict_model['head'].load_state_dict(checkpoint['head_state_dict'])
            print('成功从预训练模型加载head模块参数')
        else:
            print('预训练模型中未找到head模块参数，请检查模型文件')
            
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(load_file))
    else:
        print('There is no resume file to load!')

    # 读取标签文件并根据选择的任务筛选样本
    if args.task == 'brats_test':
        valid_list = 'C:/Users/86137/Desktop/GMMAS_last/brats_test_list.txt'
    elif args.task != 'all' and os.path.exists(args.label_file):
        label_data = pd.read_csv(args.label_file)
        # 根据任务筛选有标签的样本
        filtered_data = label_data[~label_data[args.task].isna()]
        # 获取样本ID列表
        sample_ids = filtered_data['ID'].tolist()
        
        # 创建临时文件保存筛选后的样本列表
        temp_list_file = 'C:/Users/86137/Desktop/GMMAS_last/temp_sample_list.txt'
        with open(temp_list_file, 'w') as f:
            for sample_id in sample_ids:
                f.write(f"{sample_id}\n")
        valid_list = temp_list_file
        print(f"已根据任务 {args.task} 筛选出 {len(sample_ids)} 个样本")
    else:
        valid_list = args.valid_file

    # 使用绝对路径
    valid_root = args.valid_dir
    print(f"测试列表文件: {valid_list}")
    valid_set = BraTS(valid_list, valid_root, mode='test')
   
    print('原始样本数量 = {}'.format(len(valid_set)))
    
    # 如果指定了样本数量限制，则只使用指定数量的样本
    if args.num_samples > 0 and args.num_samples < len(valid_set):
        # 创建一个子集
        indices = list(range(args.num_samples))
        valid_set_limited = torch.utils.data.Subset(valid_set, indices)
        # 保留原始数据集的names属性
        valid_set_limited.names = [valid_set.names[i] for i in indices]
        valid_set = valid_set_limited
        print(f'限制测试样本数量为 {args.num_samples}')
    
    print('实际用于测试的样本数量 = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("已创建数据加载器，准备开始测试")

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment+args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment+args.test_date)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    savepath = False
    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=dict_model,
                         load_file=load_file,
                         multimodel=False,
                         savepath=savepath,
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=False,
                         task=args.task,
                         generate_gradcam=args.generate_gradcam,
                         gradcam_layer=args.gradcam_layer
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


