import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
import pandas as pd
import matplotlib.pyplot as plt
# import scikit-learn as sklearn
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import nibabel as nib
import SimpleITK as sitk
import copy
from numpy import float64
from scipy.ndimage import label
import os.path as osp
import cv2

# 添加路径到sys.path以便导入本地模块
import sys
sys.path.append('pytorch-grad-cam-master')

# 导入pytorch-grad-cam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def caijian(a,b,c):
    (zstart, ystart, xstart), (zstop, ystop, xstop) = c.min(axis=-1), c.max(axis=-1) + 1
    roi_image = a[zstart:zstop, ystart:ystop, xstart :xstop ]
    roi_mask = b[zstart:zstop, ystart:ystop, xstart :xstop ]
    roi_image[roi_mask == 0] = 0
    # plt.imshow(roi_image[:,:,25],'gray')
    # plt.show()
    return roi_image

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  ##获取原图size
    originSpacing = itkimage.GetSpacing()  ##获取原图spacing
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int)   ##spacing格式转换

    resampler.SetReferenceImage(itkimage)   ##指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  ##得到重新采样后的图像

    return itkimgResampled

def largest_block_to_100(image):
    mask  = np.zeros_like(image)
    # 标记连通区域
    labeled_image, num_features = label(image)
    
    # 计算每个连通区域的体积
    volumes = np.array([np.sum(labeled_image == i) for i in range(1, num_features + 1)])
    
    # 找到体积最大的连通区域的标签
    largest_block_label = np.argmax(volumes) + 1
    
    # 将体积最大的连通区域中的像素值设置为100
    mask[labeled_image == largest_block_label] = 100
    
    return mask

def plain_inference(x, x_head, model, idh, grade, pq):
    with torch.no_grad():
        y_ROI, x_middle, y_output, y4_1, y3_1, y2_1 = model['head'](x_head)
        # 按照train.py中的顺序传递参数
        x1_1, x2_1, x3_1, x4_1, encoder_output, weights_x, decoder_input = model['en'](x, y_ROI, x_middle)
        print(f"Encoder输出形状: x1_1={x1_1.shape}, encoder_output={encoder_output.shape}, decoder_input={decoder_input.shape}")

        # 按照train.py中的顺序传递参数，使用decoder_input而不是encoder_output
        seg_output = model['seg'](x1_1, x2_1, x3_1, decoder_input, y4_1, y3_1, y2_1)
        print(f"Seg输出形状: {seg_output.shape}")
    
        # 分类任务，使用decoder_input而不是encoder_output
        idh_out, grade_out, _, pq_out = model['idh'](x4_1, decoder_input, idh, grade, None, pq)


        return seg_output, idh_out, grade_out, pq_out


def tailor_and_concat(x, x_head, model, idh, grade, pq):
    """
    将输入图像分为8个重叠区域，分别通过模型处理，然后拼接结果
    
    Args:
        x: 输入图像 [B, C, D, H, W]
        x_head: 用于head模块的输入 [B, C, D, H, W]
        model: 包含编码器、解码器、分类器和head模块的模型字典
        idh, grade, pq: 用于分类任务的标签
        
    Returns:
        分割结果和四个分类任务的预测
    """
    # 打印输入的形状和通道数
    print(f"tailor_and_concat输入形状: {x.shape}, 通道数: {x.shape[1]}")
    
    # 存储每个区域的分割和分类结果
    temp = []
    idh_temp = []
    grade_temp = []
    pq_temp = []
    
    # 首先使用head模块处理完整的x_head，只计算一次
    try:
        with torch.no_grad():
            # 与train.py保持一致的命名
            y_ROI, x_middle, y_output, y4_1, y3_1, y2_1 = model['head'](x_head)
            print(f"Head模块输出形状: y_ROI={y_ROI.shape}, x_middle={x_middle.shape}")
    except Exception as e:
        print(f"Head模块处理时出错: {e}")
        raise
    
    # 将图像分为8个重叠区域
    try:
        regions = [
            x[..., :128, :128, :128],
            x[..., :128, 112:240, :128],
            x[..., 112:240, :128, :128],
            x[..., 112:240, 112:240, :128],
            x[..., :128, :128, 27:155],
            x[..., :128, 112:240, 27:155],
            x[..., 112:240, :128, 27:155],
            x[..., 112:240, 112:240, 27:155]
        ]
        
        for i, region in enumerate(regions):
            print(f"Region {i} shape: {region.shape}")
            # 使用模型处理当前区域
            with torch.no_grad():
                try:
                    # 按照train.py中的顺序传递参数
                    x1_1, x2_1, x3_1, x4_1, encoder_output, weights_x, decoder_input = model['en'](region, y_ROI, x_middle)
                    print(f"Encoder输出形状: x1_1={x1_1.shape}, encoder_output={encoder_output.shape}, decoder_input={decoder_input.shape}")

                    # 按照train.py中的顺序传递参数，使用decoder_input而不是encoder_output
                    seg_output = model['seg'](x1_1, x2_1, x3_1, decoder_input, y4_1, y3_1, y2_1)
                    print(f"Seg输出形状: {seg_output.shape}")
                
                    # 分类任务，使用decoder_input而不是encoder_output
                    idh_out, grade_out, _, pq_out = model['idh'](x4_1, decoder_input, idh, grade, None, pq)
                except Exception as e:
                    print(f"处理区域 {i} 时出错: {e}")
                    raise

            # 存储结果
            temp.append(seg_output)
            idh_temp.append(idh_out)
            grade_temp.append(grade_out)
            pq_temp.append(pq_out)
    except Exception as e:
        print(f"区域切分或处理过程中出错: {e}")
        raise

    # 创建结果张量，与输入保持相同的batch size，但通道数固定为4（分割类别数）
    result = torch.zeros(x.shape[0], 4, 240, 240, 155, device=x.device)
    
    # 将分割结果拼接到结果中
    result[..., :128, :128, :128] = temp[0]
    result[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    result[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    result[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    result[..., :128, :128, 128:155] = temp[4][..., 96:123]
    result[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    result[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    result[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    # 平均所有区域的分类结果
    idh_out = torch.mean(torch.stack(idh_temp), dim=0)
    grade_out = torch.mean(torch.stack(grade_temp), dim=0)
    pq_out = torch.mean(torch.stack(pq_temp), dim=0)
    
    # 返回结果，只保留有效的155个切片
    return result[..., :155], idh_out, grade_out, pq_out

def dice_coeff(pred, target, label):
    smooth = 1.
    pred_2 = pred.copy()
    target_2 = target.copy()
    pred_2[np.where(pred != label)] = 0
    pred_2[np.where(pred == label)] = 1
    target_2[np.where(target != label)] = 0
    target_2[np.where(target == label)] = 1
    y_true_f = target_2.flatten()
    y_pred_f = pred_2.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coeff_all(pred, target):
    smooth = 1.
    pred_1 = pred.copy()
    target_1 = target.copy()
    pred_1[np.where(pred == 2)] = 1
    pred_1[np.where(pred == 4)] = 1
    target_1[np.where(target == 2)] = 1
    target_1[np.where(target == 4)] = 1
    y_true_f = target_1.flatten()
    y_pred_f = pred_1.flatten()
    print(y_true_f.shape, y_pred_f.shape)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coeff_TC(pred, target):
    smooth = 1.
    pred_3 = pred.copy()
    target_3 = target.copy()
    pred_3[np.where(pred == 4)] = 1
    target_3[np.where(target == 4)] = 1
    pred_3[np.where(pred == 2)] = 0
    target_3[np.where(target == 2)] = 0
    y_true_f = target_3.flatten()
    y_pred_f = pred_3.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def validate_softmax(
        valid_loader,
        model,
        load_file,
        multimodel,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly。   shuffle = false所以直接用i提取dataset中的name即可！
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=False,  # if you are valid when train
        task='brats_test',  # 选择测试的分类任务
        generate_gradcam=False,  # Added for Grad-CAM
        gradcam_layer=None  # Added for Grad-CAM
        ):
    from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
    import pandas as pd
    H, W, T = 240, 240, 155

    # 初始化结果DataFrame
    results_df = pd.DataFrame()

    model['en'].eval()
    model['seg'].eval()
    model['idh'].eval()
    
    # 设置head模块为eval模式
    model['head'].eval()
    print('Head module is set to eval mode')

    runtimes = []
    ET_voxels_pred_list = []

    grade_prob1 = []
    grade_conf = []
    grade_class = []
    grade_truth = []
    grade_error_case = []

    idh_prob1 = []
    idh_conf = []
    idh_class = []
    idh_truth = []
    idh_error_case = []

    pq_prob1 = []
    pq_conf = []
    pq_class = []
    pq_truth = []
    pq_error_case = []

    ids = []

    NCR_ratio =[]
    ED_ratio =[]
    ET_ratio = []
    dice_scores = []    
    dice_scores_label_2 = []
    dice_scores_label_1_4 = []
    postprocess = True

    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))

        # 使用测试数据自带的nii文件
        # t1ce_Path = '/public/home/hpc226511030/Archive/BraTS2021_TrainingData/MICCAI_BraTS2021_TrainingData/BraTS2021_01061/BraTS2021_01061_t1ce.nii.gz'
        # t1ce_image = nib.load(t1ce_Path)
        # print(len(data))
        # print("data[0]:",data[0].shape,'data[1]',data[1],'data[2]',data[2] )
        data = [t.cuda(non_blocking=True) for t in data]
        x_head, x_head_label, x, target, weight_seg, idh, grade, mgmt, pq = data  # 修改这里，添加mgmt变量
        # NOTE x 与 x_head分别为主干网络与head网络的输入


        # else:
        #     x = data
        #     x.cuda()

        x = x[..., :155]

        # 根据use_TTA参数决定是否使用测试时增强
        if use_TTA:
            print("使用测试时增强(TTA)...")
            TTA_1,TTA_2,TTA_3,TTA_4,TTA_5,TTA_6,TTA_7,TTA_8 = tailor_and_concat(x, x_head, model, idh, grade, pq),tailor_and_concat(x.flip(dims=(2,)), x_head, model, idh, grade, pq),\
                                                            tailor_and_concat(x.flip(dims=(3,)), x_head, model, idh, grade, pq),tailor_and_concat(x.flip(dims=(4,)), x_head, model, idh, grade, pq),\
                                                            tailor_and_concat(x.flip(dims=(2, 3)), x_head, model, idh, grade, pq),tailor_and_concat(x.flip(dims=(2, 4)), x_head, model, idh, grade, pq),\
                                                            tailor_and_concat(x.flip(dims=(3, 4)), x_head, model, idh, grade, pq),tailor_and_concat(x.flip(dims=(2, 3, 4)), x_head, model, idh, grade, pq)

            logit = F.softmax(TTA_1[0], 1)  # no flip
            logit += F.softmax(TTA_2[0].flip(dims=(2,)), 1)  # flip H
            logit += F.softmax(TTA_3[0].flip(dims=(3,)), 1)  # flip W
            logit += F.softmax(TTA_4[0].flip(dims=(4,)), 1)  # flip D
            logit += F.softmax(TTA_5[0].flip(dims=(2, 3)), 1)  # flip H, W
            logit += F.softmax(TTA_6[0].flip(dims=(2, 4)), 1)  # flip H, D
            logit += F.softmax(TTA_7[0].flip(dims=(3, 4)), 1)  # flip W, D
            logit += F.softmax(TTA_8[0].flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            output = logit / 8.0   #TTA
            
            # 处理分类任务的预测结果
            idh_probs = []
            grade_probs = []
            pq_probs = []
            for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
                idh_probs.append(F.softmax(pred[1],1))
                grade_probs.append(F.softmax(pred[2],1))
                pq_probs.append(F.softmax(pred[3],1))

            idh_pred = torch.mean(torch.stack(idh_probs),dim=0)
            grade_pred = torch.mean(torch.stack(grade_probs),dim=0)
            pq_pred = torch.mean(torch.stack(pq_probs),dim=0)
        else:
            print("不使用测试时增强，仅进行一次推理，也不分区域处理...")
            result = plain_inference(x, x_head, model, idh, grade, pq)
            output = F.softmax(result[0], 1)
            idh_pred = F.softmax(result[1], 1)
            grade_pred = F.softmax(result[2], 1)
            pq_pred = F.softmax(result[3], 1)
            print("idh_pred:{}, idh_truth:{}".format(idh_pred,idh))
            print("grade_pred:{}, grade_truth:{}".format(grade_pred,grade))
            print("pq_pred:{}, pq_truth:{}".format(pq_pred,pq))
            

    #         print("不使用测试时增强，仅进行一次推理...")
    #         # 只进行一次tailor_and_concat操作
    #         result = tailor_and_concat(x, x_head, model, idh, grade, pq)
    #         output = F.softmax(result[0], 1)
            
    #         # 分类任务的预测直接使用单次推理结果
    #         idh_pred = F.softmax(result[1], 1)
    #         grade_pred = F.softmax(result[2], 1)
    #         pq_pred = F.softmax(result[3], 1)

    #     # 为所有样本收集预测结果，不论任务类型
    #     # 处理IDH预测结果
    #     idh_pred_class = torch.argmax(idh_pred, dim=1)
    #     idh_class.append(idh_pred_class.item())
    #     idh_prob1.append(idh_pred[0][1].item())
    #     idh_conf.append(idh_pred[0][idh_pred_class.item()].item())
    #     idh_truth.append(idh.item())  # 即使是-1也记录下来
    #     print('id:', names[i], 'IDH_truth:', idh.item(), 'IDH_pred:', idh_pred_class.item())
    #     if idh.item() != -1 and idh_pred_class.item() != idh.item():
    #         idh_error_case.append({'id': names[i], 'truth': idh.item(), 'pred': idh_pred_class.item()})

    #     # 处理Grade预测结果
    #     grade_pred_class = torch.argmax(grade_pred, dim=1)
    #     grade_class.append(grade_pred_class.item())
    #     grade_prob1.append(grade_pred[0][1].item())
    #     grade_conf.append(grade_pred[0][grade_pred_class.item()].item())
    #     grade_truth.append(grade.item())  # 即使是-1也记录下来
    #     print('id:', names[i], 'Grade_truth:', grade.item(), 'Grade_pred:', grade_pred_class.item())
    #     if grade.item() != -1 and grade_pred_class.item() != grade.item():
    #         grade_error_case.append({'id': names[i], 'truth': grade.item(), 'pred': grade_pred_class.item()})

    #     # 处理1p19q预测结果
    #     pq_pred_class = torch.argmax(pq_pred, dim=1)
    #     pq_class.append(pq_pred_class.item())
    #     pq_prob1.append(pq_pred[0][1].item())
    #     pq_conf.append(pq_pred[0][pq_pred_class.item()].item())
    #     pq_truth.append(pq.item())  # 即使是-1也记录下来
    #     print('id:', names[i], '1p19q_truth:', pq.item(), '1p19q_pred:', pq_pred_class.item())
    #     if pq.item() != -1 and pq_pred_class.item() != pq.item():
    #         pq_error_case.append({'id': names[i], 'truth': pq.item(), 'pred': pq_pred_class.item()})

    #     ids.append(names[i])
        

    #     output = output[0, :, :H, :W, :T].cpu().detach().numpy()
    #     output = output.argmax(0)   #分割结果的四张特征图的最大概率值所在标签（第二个维度-0/1/2/3）

    #     name = str(i)
    #     if names:
    #         name = names[i]
    #         msg += '{:>20}, '.format(name)

    #     print(msg)

    #     if savepath:
    #         # .npy for further model ensemble
    #         # .nii for directly model submission
    #         assert save_format in ['npy', 'nii']
    #         if save_format == 'npy':
    #             np.save(os.path.join(savepath, name + '_preds'), output)
    #         if save_format == 'nii':
    #             # raise NotImplementedError
    #             oname = os.path.join(savepath, name, name + '_loss_t1.nii.gz')
    #             seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

    #             seg_img[np.where(output == 1)] = 1
    #             seg_img[np.where(output == 2)] = 2
    #             seg_img[np.where(output == 3)] = 4

    #             if postprocess:
    #                 image = seg_img.copy()       
    #                 image[np.where(seg_img == 2)] = 1
    #                 image[np.where(seg_img == 4)] = 1
    #                 zuobiao = np.where(image != 0)
    #                 zuobiao = np.array(zuobiao)
    #                 WT = caijian(image, image, zuobiao)
    #                 itkimage = sitk.GetImageFromArray(WT)  
    #                 itkimgResampled = resize_image_itk(itkimage, (128,128,128),
    #                                             resamplemethod= sitk.sitkLinear) ## resample使用线性插值
    #                 WT_img = sitk.GetArrayFromImage(itkimgResampled)           
    #                 # Calculate the pixel density
    #                 A, B, C = WT_img.shape
    #                 pixel_density = np.sum(WT_img) / (A * B * C)
    #                 print(f"The pixel density is: {pixel_density}")

    #                 mask = []
    #                 # Execute the functions as per the updated requirement
    #                 if pixel_density < 0.1:
    #                     print('I am entered.')
    #                     seg_img_1 = seg_img.copy()
    #                     pre_sum = np.count_nonzero(seg_img_1)
    #                     mask.append(largest_block_to_100(image))
    #                     print('pre_sum:', pre_sum)
    #                     seg_img_1[np.where(mask[-1] != 100)] = 0
    #                     post_sum = np.count_nonzero(seg_img_1)
    #                     print('post_sum——1:', post_sum)
    #                     if post_sum/pre_sum < 0.5:
    #                         while post_sum/pre_sum < 0.5:
    #                             print('do it again.')
    #                             image[np.where(mask[-1] == 100)] = 0
    #                             mask.append(largest_block_to_100(image))

    #                             seg_img_2 = seg_img.copy()
    #                             seg_img_3 = seg_img.copy()

    #                             for i in mask:
    #                                 seg_img_2[np.where(i == 100)] = 100   #todo 此处值需要在0-255之间，否则会无法统计

    #                             seg_img_3[np.where(seg_img_2 != 100)] = 0
    #                             post_sum = np.count_nonzero(seg_img_3)
    #                             print('post_sum——2:', post_sum)
    #                             # break
    #                         else:
    #                             seg_img = seg_img_3
    #                     else:
    #                         seg_img = seg_img_1

    #             print('NCR:', np.sum(seg_img == 1), ' | ED:', np.sum(seg_img == 2), ' | ET:', np.sum(seg_img == 4))
    #             WT_vol = np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4))
    #             # print('WT:', WT_vol , ' | TC:', np.sum((seg_img == 1) | (seg_img == 4)))

    #             NCR_ratio.append(round(np.sum(seg_img == 1) / WT_vol, 2)) 
    #             ED_ratio.append(round(np.sum(seg_img == 2) / WT_vol, 2))
    #             ET_ratio.append(round(np.sum(seg_img == 4) / WT_vol, 2))
                
    #             if torch.max(target) > 0.5:
    #                 #todo 计算dice
    #                 target = target.cpu().detach().numpy()
    #                 dice_score = dice_coeff_all(seg_img, target)
    #                 dice_scores.append(dice_score.item())  # 将 DICE 值添加到列表中
    #                 print('DICE:', dice_score.item())

    #                 # 计算标签为2的DICE值
    #                 dice_score_label_2 = dice_coeff(seg_img, target, label=2)
    #                 dice_scores_label_2.append(dice_score_label_2.item())  # 将标签为2的DICE值添加到列表中
    #                 print('DICE_label_2:', dice_score_label_2.item())

    #                 # 计算标签为1和4的DICE值
    #                 dice_score_label_1_4 = dice_coeff_TC(seg_img, target)
    #                 dice_scores_label_1_4.append(dice_score_label_1_4.item())  # 将标签为1和4的DICE值添加到列表中
    #                 print('DICE_label_1_4:', dice_score_label_1_4.item())

    #             # 使用默认的仿射矩阵和头部信息，避免使用t1ce_image
    #             # nib.save(nib.Nifti1Image(seg_img, affine=t1ce_image.affine,header=t1ce_image.header), oname)
    #             nib.save(nib.Nifti1Image(seg_img, affine=np.eye(4)), oname)
    #             print('Successfully save {}'.format(oname))

    #             if snapshot:
    #                 """ --- grey figure---"""
    #                 # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
    #                 # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
    #                 # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
    #                 # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
    #                 """ --- colorful figure--- """
    #                 Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
    #                 Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
    #                 Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
    #                 Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

    #                 for frame in range(T):   #每一个切片保存一张彩色图像
    #                     if not os.path.exists(os.path.join(visual, name)):
    #                         os.makedirs(os.path.join(visual, name))
    #                     # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
    #                     imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])

    if generate_gradcam and i < 5 and task == 'brats_test':  # 只对前5个样本生成Grad-CAM，且仅当task为brats_test时
        try:
            # 获取样本名称
            sample_name = str(i)
            if names and i < len(names):
                sample_name = names[i]
            
            print(f"为样本 {sample_name} 生成 {task} 任务的Grad-CAM可视化...")
            # 创建只包含当前样本的数据加载器
            single_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(valid_loader.dataset, [i]), 
                batch_size=1, 
                shuffle=False
            )
            # 传递完整的model字典
            generate_gradcam_visualizations(
                valid_loader=single_loader,
                model=model,  # 传递完整的模型字典
                task=task,
                save_dir=visual,
                target_layer=gradcam_layer
            )
            print(f"成功为样本 {sample_name} 生成Grad-CAM可视化")
        except Exception as e:
            print(f"生成Grad-CAM时出错: {e}")
    
    # print("------ 测试结果统计 ------")
    
    # # 创建指定的结果保存目录
    # results_dir = 'C:/Users/86137/Desktop/GMMAS_last/test_results'
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    
    # # 创建绘图保存目录
    # plots_dir = os.path.join(results_dir, 'plots')
    # if not os.path.exists(plots_dir):
    #     os.makedirs(plots_dir)
        
    # # 为所有样本创建DataFrame
    # results_df = pd.DataFrame({
    #     'ID': ids,
    #     'IDH_Truth': idh_truth,
    #     'IDH_Pred': idh_class,
    #     'IDH_Prob': idh_prob1,
    #     'Grade_Truth': grade_truth,
    #     'Grade_Pred': grade_class,
    #     'Grade_Prob': grade_prob1,
    #     '1p19q_Truth': pq_truth,
    #     '1p19q_Pred': pq_class,
    #     '1p19q_Prob': pq_prob1
    # })
    
    # # 保存所有样本的预测结果
    # results_file = os.path.join(results_dir, 'all_prediction_results.csv')
    # results_df.to_csv(results_file, index=False)
    # print(f"\n所有样本的预测结果保存到: {results_file}")
     
    # if task == 'all' or task == 'IDH_status' or task == 'brats_test':
    #     print("\n------ IDH Status Classification Results ------")
    #     # 筛选出有IDH标签的样本（仅在绘图和计算指标时筛选）
    #     valid_idh_samples = results_df[results_df['IDH_Truth'] != -1].copy()
        
    #     # 为IDH任务单独保存有效样本的CSV文件
    #     idh_file = os.path.join(results_dir, 'IDH_results.csv')
    #     valid_idh_samples.to_csv(idh_file, index=False)
    #     print(f"IDH有效样本分类结果保存到: {idh_file}")
        
    #     # 使用筛选后的有效样本进行指标计算和绘图
    #     if len(valid_idh_samples) > 1 and len(valid_idh_samples['IDH_Truth'].unique()) > 1:
    #         try:
    #             idh_auc = roc_auc_score(valid_idh_samples['IDH_Truth'], valid_idh_samples['IDH_Prob'])
    #             print("IDH AUC:", idh_auc)
    #             plot_roc_curve(valid_idh_samples['IDH_Truth'].values, valid_idh_samples['IDH_Prob'].values, 'IDH', plots_dir)
    #             plot_dca_curve(valid_idh_samples['IDH_Truth'].values, valid_idh_samples['IDH_Prob'].values, 'IDH', plots_dir)
                
    #             print("IDH Accuracy:", accuracy_score(valid_idh_samples['IDH_Truth'], valid_idh_samples['IDH_Pred']))
    #             plot_confusion_matrix(valid_idh_samples['IDH_Truth'].values, valid_idh_samples['IDH_Pred'].values, 'IDH', plots_dir)
    #         except Exception as e:
    #             print(f"计算IDH指标或绘图时出错: {e}")
    #             print(f"数据详情: 样本数={len(valid_idh_samples)}, 类别数={len(valid_idh_samples['IDH_Truth'].unique())}")
    #             print(f"类别分布: {valid_idh_samples['IDH_Truth'].value_counts()}")
    #     else:
    #         print(f"无法为IDH生成ROC/DCA曲线: 样本数={len(valid_idh_samples)}, 类别数={len(valid_idh_samples['IDH_Truth'].unique())}")
    
    # if task == 'all' or task == 'Grade' or task == 'brats_test':
    #     print("\n------ Grade Classification Results ------")
    #     # 筛选出有Grade标签的样本（仅在绘图和计算指标时筛选）
    #     valid_grade_samples = results_df[results_df['Grade_Truth'] != -1].copy()
        
    #     # 为Grade任务单独保存有效样本的CSV文件
    #     grade_file = os.path.join(results_dir, 'Grade_results.csv')
    #     valid_grade_samples.to_csv(grade_file, index=False)
    #     print(f"Grade有效样本分类结果保存到: {grade_file}")
        
    #     # 使用筛选后的有效样本进行指标计算和绘图
    #     if len(valid_grade_samples) > 1 and len(valid_grade_samples['Grade_Truth'].unique()) > 1:
    #         try:
    #             grade_auc = roc_auc_score(valid_grade_samples['Grade_Truth'], valid_grade_samples['Grade_Prob'])
    #             print("Grade AUC:", grade_auc)
    #             plot_roc_curve(valid_grade_samples['Grade_Truth'].values, valid_grade_samples['Grade_Prob'].values, 'Grade', plots_dir)
    #             plot_dca_curve(valid_grade_samples['Grade_Truth'].values, valid_grade_samples['Grade_Prob'].values, 'Grade', plots_dir)
                
    #             print("Grade Accuracy:", accuracy_score(valid_grade_samples['Grade_Truth'], valid_grade_samples['Grade_Pred']))
    #             plot_confusion_matrix(valid_grade_samples['Grade_Truth'].values, valid_grade_samples['Grade_Pred'].values, 'Grade', plots_dir)
    #         except Exception as e:
    #             print(f"计算Grade指标或绘图时出错: {e}")
    #             print(f"数据详情: 样本数={len(valid_grade_samples)}, 类别数={len(valid_grade_samples['Grade_Truth'].unique())}")
    #             print(f"类别分布: {valid_grade_samples['Grade_Truth'].value_counts()}")
    #     else:
    #         print(f"无法为Grade生成ROC/DCA曲线: 样本数={len(valid_grade_samples)}, 类别数={len(valid_grade_samples['Grade_Truth'].unique())}")
    
    # if task == 'all' or task == '1p19q' or task == 'brats_test':
    #     print("\n------ 1p19q Classification Results ------")
    #     # 筛选出有1p19q标签的样本（仅在绘图和计算指标时筛选）
    #     valid_pq_samples = results_df[results_df['1p19q_Truth'] != -1].copy()
        
    #     # 为1p19q任务单独保存有效样本的CSV文件
    #     pq_file = os.path.join(results_dir, 'PQ_results.csv')
    #     valid_pq_samples.to_csv(pq_file, index=False)
    #     print(f"1p19q有效样本分类结果保存到: {pq_file}")
        
    #     # 使用筛选后的有效样本进行指标计算和绘图
    #     if len(valid_pq_samples) > 1 and len(valid_pq_samples['1p19q_Truth'].unique()) > 1:
    #         try:
    #             pq_auc = roc_auc_score(valid_pq_samples['1p19q_Truth'], valid_pq_samples['1p19q_Prob'])
    #             print("1p19q AUC:", pq_auc)
    #             plot_roc_curve(valid_pq_samples['1p19q_Truth'].values, valid_pq_samples['1p19q_Prob'].values, '1p19q', plots_dir)
    #             plot_dca_curve(valid_pq_samples['1p19q_Truth'].values, valid_pq_samples['1p19q_Prob'].values, '1p19q', plots_dir)
                
    #             print("1p19q Accuracy:", accuracy_score(valid_pq_samples['1p19q_Truth'], valid_pq_samples['1p19q_Pred']))
    #             plot_confusion_matrix(valid_pq_samples['1p19q_Truth'].values, valid_pq_samples['1p19q_Pred'].values, '1p19q', plots_dir)
    #         except Exception as e:
    #             print(f"计算1p19q指标或绘图时出错: {e}")
    #             print(f"数据详情: 样本数={len(valid_pq_samples)}, 类别数={len(valid_pq_samples['1p19q_Truth'].unique())}")
    #             print(f"类别分布: {valid_pq_samples['1p19q_Truth'].value_counts()}")
    #     else:
    #         print(f"无法为1p19q生成ROC/DCA曲线: 样本数={len(valid_pq_samples)}, 类别数={len(valid_pq_samples['1p19q_Truth'].unique())}")
                
    # # 在测试完成后调用后处理函数重新绘制DCA曲线
    # try:
    #     print("\n开始处理DCA曲线...")
    #     post_process_dca(results_dir)
    # except Exception as e:
    #     print(f"处理DCA曲线时出错: {e}")

# 修改plot_roc_curve函数，使用英文而非中文，并保存原始预测结果
def plot_roc_curve(y_true, y_pred_prob, task_name, save_dir='plots'):
    """
    绘制ROC曲线并保存，同时保存ROC数据
    
    参数:
    y_true: 真实标签列表
    y_pred_prob: 预测为正类的概率列表
    task_name: 任务名称（如'IDH', 'Grade'等）
    save_dir: 保存图像的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 计算ROC曲线的各个点
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    
    # 保存ROC数据到CSV文件
    roc_data_dir = os.path.join(os.path.dirname(save_dir), 'roc_data')
    if not os.path.exists(roc_data_dir):
        os.makedirs(roc_data_dir)
        
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thresholds
    })
    roc_data_file = os.path.join(roc_data_dir, f'{task_name}_ROC_data.csv')
    roc_data.to_csv(roc_data_file, index=False)
    print(f"ROC data saved to {roc_data_file}")
    
    # 保存原始预测结果和真实标签
    pred_data_dir = os.path.join(os.path.dirname(save_dir), 'prediction_data')
    if not os.path.exists(pred_data_dir):
        os.makedirs(pred_data_dir)
        
    pred_data = pd.DataFrame({
        'True_Label': y_true,
        'Prediction_Probability': y_pred_prob
    })
    pred_data_file = os.path.join(pred_data_dir, f'{task_name}_predictions.csv')
    pred_data.to_csv(pred_data_file, index=False)
    print(f"Prediction data saved to {pred_data_file}")

    # 设置图表字体大小
    plt.rcParams['font.size'] = 14  # 基础字体大小
    plt.rcParams['axes.titlesize'] = 25  # 标题字体大小
    plt.rcParams['axes.labelsize'] = 20  # 轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
    plt.rcParams['legend.fontsize'] = 20  # 图例字体大小
    plt.rcParams['font.weight'] = 'bold'  # 设置字体加粗

    # 绘制ROC曲线 - 使用英文而非中文
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {task_name} Classification')
    plt.legend(loc="lower right")
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, f'{task_name}_ROC.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return auc

# 使用与idh_dca_plot.py一致的净收益计算函数
def calculate_net_benefit(y_true, y_pred_prob, threshold):
    """
    净收益计算函数，确保返回非负值
    
    参数:
    y_true: 真实标签列表
    y_pred_prob: 预测为正类的概率列表
    threshold: 阈值
    
    返回:
    net_benefit: 净收益值，如果是负值则返回0
    """
    # 计算阈值下的预测
    y_pred = np.array(y_pred_prob) >= threshold
    
    # 计算真阳性和假阳性
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    # 计算净收益
    n = len(y_true)
    
    # 处理没有预测为阳性的情况
    if tp + fp == 0:
        return 0.0
    
    # 标准净收益公式
    net_benefit = (tp/n) - (fp/n) * (threshold/(1-threshold))
    
    # 确保返回非负值
    return max(0, net_benefit)

# 修改post_process_dca函数，确保其使用改进后的plot_dca_curve和calculate_net_benefit函数
def post_process_dca(results_dir='test_results'):
    """
    根据测试结果CSV文件重新计算和绘制DCA曲线
    
    参数:
    results_dir: 测试结果目录
    """
    print("\n===== 开始处理DCA曲线 =====")
    
    # 使用原始目录
    dca_plots_dir = os.path.join(results_dir, 'plots')
    dca_data_dir = os.path.join(results_dir, 'dca_data')
    
    if not os.path.exists(dca_plots_dir):
        os.makedirs(dca_plots_dir)
    if not os.path.exists(dca_data_dir):
        os.makedirs(dca_data_dir)
    
    # 任务列表 - 移除MGMT
    tasks = ['IDH', 'Grade', '1p19q']
    
    for task in tasks:
        # 对于IDH任务，CSV文件名为IDH_results.csv
        # 对于1p19q任务，CSV文件名为PQ_results.csv
        file_name = f"{task}_results.csv"
        if task == '1p19q':
            file_name = "PQ_results.csv"
        
        csv_file = os.path.join(results_dir, file_name)
        
        if not os.path.exists(csv_file):
            print(f"警告: 找不到 {csv_file}, 跳过该任务")
            continue
        
        print(f"\n处理 {task} 任务的DCA曲线...")
        
        try:
            # 读取预测数据
            pred_data = pd.read_csv(csv_file)
            
            # 根据任务名称确定列名
            if task == '1p19q':
                truth_col = '1p19q_Truth'
                prob_col = '1p19q_Prob'
            else:
                truth_col = f'{task}_Truth'
                prob_col = f'{task}_Prob'
            
            # 提取有效样本(排除-1标签)
            valid_data = pred_data[pred_data[truth_col] != -1]
            
            if len(valid_data) < 2:
                print(f"警告: {task} 任务有效样本不足 ({len(valid_data)}), 跳过")
                continue
                
            # 检查类别数量
            if len(np.unique(valid_data[truth_col])) < 2:
                print(f"警告: {task} 任务只有一种类别, 跳过")
                continue
            
            # 提取真实标签和预测概率
            y_true = valid_data[truth_col].values
            y_pred_prob = valid_data[prob_col].values
            
            # 使用修改后的plot_dca_curve函数来绘制DCA曲线
            plot_dca_curve(y_true, y_pred_prob, task, dca_plots_dir)
            
        except Exception as e:
            print(f"处理 {task} 任务时出错: {e}")
            # 输出详细错误信息帮助调试
            import traceback
            traceback.print_exc()
    
    print("\n===== DCA曲线处理完成 =====")

# 修改plot_dca_curve函数，保持与idh_dca_plot.py中一致
def plot_dca_curve(y_true, y_pred_prob, task_name, save_dir='plots'):
    """
    绘制决策曲线分析(DCA)图并保存，同时保存DCA数据
    
    参数:
    y_true: 真实标签列表
    y_pred_prob: 预测为正类的概率列表
    task_name: 任务名称（如'IDH', 'Grade'等）
    save_dir: 保存图像的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 分析数据集基本情况
    print("===== 数据集分析 =====")
    print(f"样本总数: {len(y_true)}")
    print(f"阳性样本数: {np.sum(y_true == 1)}")
    print(f"阴性样本数: {np.sum(y_true == 0)}")
    print(f"阳性比例: {np.mean(y_true):.4f}")
    
    print("\n预测概率分布:")
    print(f"最小值: {np.min(y_pred_prob):.4f}")
    print(f"最大值: {np.max(y_pred_prob):.4f}")
    print(f"平均值: {np.mean(y_pred_prob):.4f}")
    print(f"中位数: {np.median(y_pred_prob):.4f}")
    
    # 生成更细的阈值序列
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    # 重新计算每个阈值的净收益
    nb_model = []
    for t in thresholds:
        nb = calculate_net_benefit(y_true, y_pred_prob, t)
        nb_model.append(nb)
        print(f"阈值 {t:.2f}: 净收益 = {nb:.6f}")
    
    # 计算"全部治疗"的净收益（假设全部为阳性）
    nb_all = []
    for t in thresholds:
        # 避免除零错误
        if t >= 1.0:
            nb_all.append(0)
        else:
            all_benefit = np.mean(y_true) - (1-np.mean(y_true)) * t/(1-t)
            # 强制将负值净收益设为0
            nb_all.append(max(0, all_benefit))
    
    # 计算"全部不治疗"的净收益（假设全部为阴性）
    nb_none = [0] * len(thresholds)
    
    # 保存DCA数据到CSV文件
    dca_data_dir = os.path.join(os.path.dirname(save_dir), 'dca_data')
    if not os.path.exists(dca_data_dir):
        os.makedirs(dca_data_dir)
        
    dca_data = pd.DataFrame({
        'Thresholds': thresholds,
        'Net_Benefit_Model': nb_model,
        'Net_Benefit_All': nb_all,
        'Net_Benefit_None': nb_none
    })
    dca_data_file = os.path.join(dca_data_dir, f'{task_name}_DCA_data.csv')
    dca_data.to_csv(dca_data_file, index=False)
    print(f"\nDCA数据已保存到: {dca_data_file}")

    # 设置字体和图表样式
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 24
    
    # 创建高质量图表
    plt.figure(figsize=(12, 10), dpi=300)
    
    # 设置图例字体大小
    plt.rcParams['legend.fontsize'] = 20
    
    # 绘制曲线
    plt.plot(thresholds, nb_model, 'r-', linewidth=4, label='GMMAS')
    plt.plot(thresholds, nb_all, 'g-', linewidth=4, label='Treat All')
    plt.plot(thresholds, nb_none, 'b-', linewidth=4, label='Treat None')
    
    # 设置坐标轴和标题
    plt.xlim([0.0, 1.0])
    # 强制Y轴下限为0，确保没有负值
    y_max = max(max(nb_model) if nb_model else 0, max(nb_all) if nb_all else 0) + 0.05
    plt.ylim([0.0, y_max])
    
    # 放大字体
    plt.xlabel('Threshold Probability', fontweight='bold', fontsize=25)
    plt.ylabel('Net Benefit', fontweight='bold', fontsize=25)
    plt.title(f'Decision Curve Analysis for {task_name} Classification', fontweight='bold', fontsize=25)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # 美化图表
    plt.tight_layout()
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=2)
    # 确保只填充正值部分
    plt.fill_between(thresholds, nb_model, 0, where=(np.array(nb_model) > 0), alpha=0.2, color='r')
    
    # 增大刻度线宽度
    plt.tick_params(width=2)
    
    # 增大坐标轴线宽
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(2)
    
    # 保存图表
    output_file = os.path.join(save_dir, f'{task_name}_DCA.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"DCA曲线已保存到: {output_file}")

# 修改绘制混淆矩阵的函数，使用英文而非中文
def plot_confusion_matrix(y_true, y_pred, task_name, save_dir='plots'):
    """
    绘制混淆矩阵图像并保存
    
    参数:
    y_true: 真实标签列表
    y_pred: 预测标签列表
    task_name: 任务名称（如'IDH', 'Grade'等）
    save_dir: 保存图像的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 设置标签名称
    if task_name == 'IDH':
        class_names = ['Wild Type', 'Mutant']
    elif task_name == 'Grade':
        class_names = ['LGG', 'GBM']
    elif task_name == 'MGMT':
        class_names = ['Unmethylated', 'Methylated']
    elif task_name == '1p19q':
        class_names = ['Intact', 'Co-deleted']
    else:
        class_names = ['Class 0', 'Class 1']
    
    # 绘制混淆矩阵 - 使用英文而非中文
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {task_name} Classification')
    plt.colorbar()
    
    # 添加刻度
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, f'{task_name}_Confusion_Matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 添加Grad-CAM生成函数
def generate_gradcam_visualizations(valid_loader, model, task='IDH_status', save_dir='visualization', target_layer=None):
    """
    为指定模型和任务生成Grad-CAM可视化图像
    
    参数:
    valid_loader: 数据加载器，包含测试样本
    model: 模型字典，包含编码器、分割器等组件
    task: 分类任务，可选项: IDH_status, Grade, MGMT, 1p19q 
    save_dir: 保存Grad-CAM图像的目录
    target_layer: 目标层，如果为None则使用默认层
    """
    print(f"\n开始生成 {task} 任务的Grad-CAM可视化...")
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 设置为评估模式
    model['en'].eval()
    model['seg'].eval()
    model['idh'].eval()
    model['head'].eval()
    
    # 创建Grad-CAM目录
    gradcam_dir = os.path.join(save_dir, 'gradcam_visualizations')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # 定义Hook类来获取输出和梯度 - 使用full_backward_hook解决警告
    class SaveFeatures():
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)
            # 使用register_full_backward_hook替代旧的backward_hook
            self.hook_grad = module.register_full_backward_hook(self.hook_grad_fn)
            self.features = None
            self.grad_features = None
        
        def hook_fn(self, module, input, output):
            # 保存输出
            self.features = output
            # 确保需要梯度
            self.features.requires_grad_(True)
            # 保留梯度以便在非叶子张量上访问
            self.features.retain_grad()
        
        def hook_grad_fn(self, module, grad_input, grad_output):
            # 保存梯度 - full_backward_hook返回的grad_output是一个元组
            self.grad_features = grad_output[0]
        
        def remove(self):
            self.hook.remove()
            self.hook_grad.remove()
    
    # 如果target_layer为None，根据任务设置默认目标层
    if target_layer is None:
        # 列出可能的默认层候选 - 优先选择浅层卷积网络
        candidate_layers = [
            # 浅层卷积网络 - 首选，包含更多空间细节
            ('en', 'module.Unet.EnBlock1.conv2'),  # 第一个编码块的最后一层
            ('en', 'module.Unet.EnBlock2_1.conv2'),  # 第二个编码块
            ('en', 'module.Unet.EnBlock3_1.conv2'),  # 第三个编码块
            
            # 中层特征
            ('en', 'module.conv_x'),
            
            # 深层特征 - 最后选择
            ('idh', 'module.conv_en_idh'),
            ('idh', 'module.conv1_idh'),
            ('en', 'module.ChannelAttention.conv_1')
        ]
        
        # 尝试所有候选层，找到第一个有效的
        target_found = False
        for model_key, layer_path in candidate_layers:
            if model_key in model:
                try:
                    current = model[model_key]
                    parts = layer_path.split('.')
                    for part in parts:
                        if hasattr(current, part):
                            current = getattr(current, part)
                        else:
                            raise AttributeError(f"无效路径: {layer_path}")
                    
                    # 如果到这里没报错，说明找到了目标层
                    target_layer = current
                    print(f"找到默认目标层: {model_key}.{layer_path}")
                    target_found = True
                    break
                except Exception as e:
                    print(f"尝试使用 {model_key}.{layer_path} 失败: {e}")
        
        if not target_found:
            # 如果所有候选都失败，使用模型的一个主要组件作为目标
            print("无法找到任何候选层，尝试使用整个模型组件")
            for key in ['en', 'idh', 'seg', 'head']:
                if key in model:
                    try:
                        target_layer = model[key]
                        print(f"使用 {key} 组件作为目标层")
                        target_found = True
                        break
                    except:
                        continue
            
            if not target_found:
                print("无法找到任何有效目标层，Grad-CAM可能无法正常工作")
                return
    else:
        # 如果提供了字符串形式的层路径，解析它
        if isinstance(target_layer, str):
            try:
                # 分解层路径
                parts = target_layer.split('.')
                current = model
                
                # 第一部分可能是字典键
                if parts[0] in model:
                    current = model[parts[0]]
                    parts = parts[1:]
                    
                # 遍历剩余路径获取指定模块
                for part in parts:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        raise AttributeError(f"对象 {type(current).__name__} 没有属性 {part}")
                    
                target_layer = current
                print(f"成功获取指定层: {target_layer.__class__.__name__}")
            except Exception as e:
                print(f"获取指定层时出错: {e}")
                # 回退到默认层选择逻辑
                print("回退到默认层选择逻辑...")
                # 设置target_layer为None并再次调用相同的函数
                return generate_gradcam_visualizations(
                    valid_loader=valid_loader,
                    model=model,
                    task=task,
                    save_dir=save_dir,
                    target_layer=None
                )
    
    # 注册钩子
    layer_hook = SaveFeatures(target_layer)
    
    # 创建自定义的颜色映射集合
    def create_colormaps():
        # 基于jet颜色映射
        colormaps = {}
        
        # 1. 轻量化jet颜色映射（减淡蓝色）
        jet_colors = plt.cm.jet(np.arange(256))
        jet_colors[:128, :3] = jet_colors[:128, :3] * 0.7 + 0.3  # 使蓝色变浅
        colormaps['light_jet'] = plt.matplotlib.colors.ListedColormap(jet_colors)
        
        # 2. 热力图颜色映射
        colormaps['hot'] = plt.cm.hot
        
        # 3. 彩虹颜色映射
        colormaps['rainbow'] = plt.cm.rainbow
        
        # 4. 紫红色颜色映射
        colormaps['plasma'] = plt.cm.plasma
        
        # 5. 蓝紫红色颜色映射
        colormaps['cool'] = plt.cm.cool
        
        # 6. 红蓝色颜色映射
        colormaps['RdBu_r'] = plt.cm.RdBu_r
        
        # 7. 红黄蓝色颜色映射
        colormaps['RdYlBu_r'] = plt.cm.RdYlBu_r
        
        # 8. 棕紫色颜色映射
        colormaps['viridis'] = plt.cm.viridis
        
        # 9. 青黄色颜色映射
        colormaps['YlGnBu'] = plt.cm.YlGnBu
        
        # 10. 火焰色颜色映射
        colormaps['inferno'] = plt.cm.inferno
        
        return colormaps
    
    # 创建自定义颜色映射
    colormaps = create_colormaps()
    
    # 遍历数据集
    for i, data in enumerate(valid_loader):
        # 只处理少量样本以节省时间
        if i >= 5:  # 只处理前5个样本
            break
            
        print(f"处理样本 {i+1}/5...")
        data = [t.cuda(non_blocking=True) for t in data]
        x_head, x_head_label, x, target, weight_seg, idh, grade, mgmt, pq = data
        
        # 获取样本名称
        name = valid_loader.dataset.names[i] if hasattr(valid_loader.dataset, 'names') else f"sample_{i}"
        
        # 创建样本特定目录
        sample_dir = os.path.join(gradcam_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 选择对应任务的目标
        if task == 'IDH_status':
            target_category = idh.item()
        elif task == 'Grade':
            target_category = grade.item()
        elif task == 'MGMT':
            target_category = mgmt.item()
        elif task == '1p19q':
            target_category = pq.item()
        else:
            target_category = 1  # 默认为阳性类别
            
        # 如果标签是-1（缺失），则默认为阳性类别
        if target_category == -1:
            target_category = 1
        
        # 前向传播
        try:
            with torch.enable_grad():  # 需要梯度用于GradCAM
                # Head模块处理
                y_ROI, x_middle, y_output, y4_1, y3_1, y2_1 = model['head'](x_head)
                
                # 编码器处理
                x1_1, x2_1, x3_1, x4_1, encoder_output, weights_x, decoder_input = model['en'](x, y_ROI, x_middle)
                
                # 分类任务
                if task == 'IDH_status':
                    pred, _, _, _ = model['idh'](x4_1, decoder_input, idh, grade, None, pq)
                elif task == 'Grade':
                    _, pred, _, _ = model['idh'](x4_1, decoder_input, idh, grade, None, pq)
                elif task == 'MGMT':
                    _, _, pred, _ = model['idh'](x4_1, decoder_input, idh, grade, mgmt, pq)
                elif task == '1p19q':
                    _, _, _, pred = model['idh'](x4_1, decoder_input, idh, grade, None, pq)
                else:
                    pred, _, _, _ = model['idh'](x4_1, decoder_input, idh, grade, None, pq)
                
                # 选择目标类别的输出进行反向传播
                pred = F.softmax(pred, dim=1)
                score = pred[:, target_category]
                
                # 梯度清零
                model['en'].zero_grad()
                model['idh'].zero_grad()
                
                # 反向传播
                score.backward(retain_graph=True)
                
                # 获取梯度
                gradients = layer_hook.grad_features
                
                # 获取激活值
                activations = layer_hook.features
                
                # 检查梯度和激活值是否有效
                if gradients is None:
                    print("警告：梯度为None，使用随机梯度代替")
                    # 创建随机梯度代替（仅用于演示）
                    gradients = torch.randn_like(activations)
                
                if activations is None:
                    print("错误：激活值为None，无法继续")
                    raise ValueError("激活值为None，请检查目标层")
                
                # 计算GradCAM
                print(f"梯度形状: {gradients.shape}, 激活值形状: {activations.shape}")
                weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True) if len(gradients.shape) > 4 else torch.mean(gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * activations, dim=1)
                cam = F.relu(cam)  # ReLU激活
                
                # 归一化
                cam = cam - torch.min(cam)
                cam = cam / (torch.max(cam) + 1e-8)
                
                # 转换为NumPy数组并调整大小
                cam_np = cam.cpu().detach().numpy()[0]  # [D, H, W]
                
                # 将原始图像转换为可显示格式
                x_np = x.cpu().detach().numpy()[0, 0]  # 假设通道0是T1或T1CE图像
                
                # 检查维度以确保匹配
                print(f"热力图维度: {cam_np.shape}, 原始图像维度: {x_np.shape}")
                
                # 使用三线性插值将热力图上采样到原始图像尺寸
                if cam_np.shape != x_np.shape:
                    print("检测到维度不匹配，使用三线性插值进行上采样...")
                    try:
                        # 创建原始坐标
                        orig_shape = cam_np.shape
                        target_shape = x_np.shape
                        
                        # 进行3D上采样
                        from scipy.ndimage import zoom
                        zoom_factors = (
                            target_shape[0] / orig_shape[0],
                            target_shape[1] / orig_shape[1], 
                            target_shape[2] / orig_shape[2]
                        )
                        print(f"应用缩放因子: {zoom_factors}")
                        
                        # 使用三线性插值进行上采样
                        cam_np = zoom(cam_np, zoom_factors, order=1)  # order=1为线性插值
                        print(f"上采样后热力图维度: {cam_np.shape}")
                    except Exception as e:
                        print(f"热力图上采样出错: {e}")
                        print("无法生成有效的热力图，终止处理")
                        return
                
                # 对原始图像进行归一化，使其范围在[0, 1]之间
                x_np = (x_np - np.min(x_np)) / (np.max(x_np) - np.min(x_np) + 1e-8)
                
                # 假设MRI数据格式为[D, H, W] = [z, y, x]，则：
                # Axial(轴状面): 沿z轴切片 (xy平面)
                # Coronal(冠状面): 沿y轴切片 (xz平面)
                # Sagittal(矢状面): 沿x轴切片 (yz平面)
                print(f"原始图像形状: {x_np.shape}, 格式为[D, H, W] = [z, y, x]")
                
                # 为三个不同的视图创建目录
                axial_dir = os.path.join(sample_dir, 'axial')
                coronal_dir = os.path.join(sample_dir, 'coronal')
                sagittal_dir = os.path.join(sample_dir, 'sagittal')
                
                os.makedirs(axial_dir, exist_ok=True)
                os.makedirs(coronal_dir, exist_ok=True)
                os.makedirs(sagittal_dir, exist_ok=True)
                
                # 处理轴状面(Axial)视图 - 沿z轴切片
                print("生成轴状面(Axial)视图...")
                axial_nonzero_counts = []
                for slice_idx in range(x_np.shape[0]):
                    # 计算每个切片的非零像素数量
                    non_zero_count = np.count_nonzero(x_np[slice_idx] > 0.1)
                    axial_nonzero_counts.append((slice_idx, non_zero_count))
                
                # 按非零像素数排序
                axial_slices = sorted(axial_nonzero_counts, key=lambda x: x[1], reverse=True)
                
                # 选择有显著内容的切片(非零像素数大于阈值)
                significant_axial_slices = [(idx, count) for idx, count in axial_slices if count > 1000]
                
                if significant_axial_slices:
                    print(f"找到 {len(significant_axial_slices)} 个有显著内容的Axial切片")
                    # 选择中间40%~80%的切片范围，避开太前面和太后面的切片
                    start_idx = int(len(significant_axial_slices) * 0.4)
                    end_idx = int(len(significant_axial_slices) * 0.8)
                    middle_axial_slices = [idx for idx, _ in significant_axial_slices[start_idx:end_idx]]
                    
                    if not middle_axial_slices:
                        middle_axial_slices = [idx for idx, _ in significant_axial_slices]
                else:
                    print("未找到有显著内容的Axial切片，使用中间范围切片")
                    # 使用中间1/3的切片
                    middle_axial_slices = list(range(x_np.shape[0]//3, x_np.shape[0]*2//3))
                
                # 在这些切片中寻找热力图激活最强的区域
                axial_activations = []
                for slice_idx in middle_axial_slices:
                    if slice_idx < cam_np.shape[0]:
                        # 计算该切片热力图的最大值
                        max_activation = np.max(cam_np[slice_idx])
                        if max_activation > 0:  # 只考虑有正激活的切片
                            axial_activations.append((slice_idx, max_activation))
                
                # 如果找到了有正激活的切片
                if axial_activations:
                    # 按激活值排序，选择前5个
                    top_axial_slices = sorted(axial_activations, key=lambda x: x[1], reverse=True)[:5]
                    print(f"选择的高激活Axial切片索引: {[idx for idx, _ in top_axial_slices]}")
                    
                    # 只保存这些高激活区域的切片
                    for slice_idx, activation in top_axial_slices:
                        # 获取单个切片
                        img_slice = x_np[slice_idx]
                        cam_slice = cam_np[slice_idx]
                        
                        # 确保每个切片的形状匹配
                        if img_slice.shape != cam_slice.shape:
                            # 如果仍然不匹配，调整热力图尺寸以匹配当前切片
                            cam_slice = cv2.resize(cam_slice, (img_slice.shape[1], img_slice.shape[0]))
                        
                        # 调整热图尺寸以匹配原图
                        cam_resized = cam_slice
                        
                        # 确保热力图正确归一化，避免空热力图
                        if np.max(cam_resized) > 0:
                            cam_resized = cam_resized / np.max(cam_resized)
                        else:
                            print(f"警告: Axial切片 {slice_idx} 的热力图全为零，跳过该切片")
                            continue
                            
                        # 为每种颜色映射创建可视化
                        for cmap_name, cmap in colormaps.items():
                            # 使用当前颜色映射转换为彩色图
                            heatmap_rgb = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
                            
                            # 将灰度图转为RGB以便叠加
                            img_rgb = np.stack([img_slice]*3, axis=2)
                            # 确保转换为uint8类型以匹配heatmap_rgb
                            img_rgb = (img_rgb * 255).astype(np.uint8)
                            
                            # 叠加原图和热图
                            overlay = cv2.addWeighted(img_rgb, 0.4, heatmap_rgb, 0.6, 0)
                            
                            # 创建并保存可视化结果
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            # 原始图像
                            axes[0].imshow(img_slice, cmap='gray')
                            axes[0].set_title('')
                            axes[0].axis('off')
                            
                            # GradCAM热图
                            axes[1].imshow(cam_resized, cmap=cmap)
                            axes[1].set_title('')
                            axes[1].axis('off')
                            
                            # 叠加结果
                            axes[2].imshow(overlay)
                            axes[2].set_title('')
                            axes[2].axis('off')
                            
                            # 移除所有边框和额外的空白
                            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0)
                            plt.tight_layout(pad=0)
                            
                            # 保存无文字的图像
                            plt.savefig(os.path.join(axial_dir, f'{task}_axial_slice_{slice_idx}_activation_{activation:.4f}_colormap_{cmap_name}.png'), 
                                      dpi=300, bbox_inches='tight', pad_inches=0)
                            plt.close()
                
                # 处理冠状面(Coronal)视图 - 沿y轴切片
                print("生成冠状面(Coronal)视图...")
                coronal_nonzero_counts = []
                for slice_idx in range(x_np.shape[1]):
                    # 计算每个切片的非零像素数量
                    non_zero_count = np.count_nonzero(x_np[:, slice_idx, :] > 0.1)
                    coronal_nonzero_counts.append((slice_idx, non_zero_count))
                
                # 按非零像素数排序
                coronal_slices = sorted(coronal_nonzero_counts, key=lambda x: x[1], reverse=True)
                
                # 选择有显著内容的切片(非零像素数大于阈值)
                significant_coronal_slices = [(idx, count) for idx, count in coronal_slices if count > 1000]
                
                if significant_coronal_slices:
                    print(f"找到 {len(significant_coronal_slices)} 个有显著内容的Coronal切片")
                    # 选择中间40%~80%的切片范围
                    start_idx = int(len(significant_coronal_slices) * 0.4)
                    end_idx = int(len(significant_coronal_slices) * 0.8)
                    middle_coronal_slices = [idx for idx, _ in significant_coronal_slices[start_idx:end_idx]]
                    
                    if not middle_coronal_slices:
                        middle_coronal_slices = [idx for idx, _ in significant_coronal_slices]
                else:
                    print("未找到有显著内容的Coronal切片，使用中间范围切片")
                    # 使用中间1/3的切片
                    middle_coronal_slices = list(range(x_np.shape[1]//3, x_np.shape[1]*2//3))
                
                # 在这些切片中寻找热力图激活最强的区域
                coronal_activations = []
                for slice_idx in middle_coronal_slices:
                    if slice_idx < cam_np.shape[1]:
                        # 计算该切片热力图的最大值
                        max_activation = np.max(cam_np[:, slice_idx, :])
                        if max_activation > 0:  # 只考虑有正激活的切片
                            coronal_activations.append((slice_idx, max_activation))
                
                # 如果找到了有正激活的切片
                if coronal_activations:
                    # 按激活值排序，选择前5个
                    top_coronal_slices = sorted(coronal_activations, key=lambda x: x[1], reverse=True)[:5]
                    print(f"选择的高激活Coronal切片索引: {[idx for idx, _ in top_coronal_slices]}")
                    
                    # 只保存这些高激活区域的切片
                    for slice_idx, activation in top_coronal_slices:
                        # 获取单个切片
                        img_slice = x_np[:, slice_idx, :]
                        cam_slice = cam_np[:, slice_idx, :]
                        
                        # 确保每个切片的形状匹配
                        if img_slice.shape != cam_slice.shape:
                            # 如果仍然不匹配，调整热力图尺寸以匹配当前切片
                            cam_slice = cv2.resize(cam_slice, (img_slice.shape[1], img_slice.shape[0]))
                        
                        # 调整热图尺寸以匹配原图
                        cam_resized = cam_slice
                        
                        # 确保热力图正确归一化，避免空热力图
                        if np.max(cam_resized) > 0:
                            cam_resized = cam_resized / np.max(cam_resized)
                        else:
                            print(f"警告: Coronal切片 {slice_idx} 的热力图全为零，跳过该切片")
                            continue
                            
                        # 为每种颜色映射创建可视化
                        for cmap_name, cmap in colormaps.items():
                            # 使用当前颜色映射转换为彩色图
                            heatmap_rgb = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
                            
                            # 将灰度图转为RGB以便叠加
                            img_rgb = np.stack([img_slice]*3, axis=2)
                            # 确保转换为uint8类型以匹配heatmap_rgb
                            img_rgb = (img_rgb * 255).astype(np.uint8)
                            
                            # 叠加原图和热图
                            overlay = cv2.addWeighted(img_rgb, 0.4, heatmap_rgb, 0.6, 0)
                            
                            # 创建并保存可视化结果
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            # 原始图像
                            axes[0].imshow(img_slice, cmap='gray')
                            axes[0].set_title('')
                            axes[0].axis('off')
                            
                            # GradCAM热图
                            axes[1].imshow(cam_resized, cmap=cmap)
                            axes[1].set_title('')
                            axes[1].axis('off')
                            
                            # 叠加结果
                            axes[2].imshow(overlay)
                            axes[2].set_title('')
                            axes[2].axis('off')
                            
                            # 移除所有边框和额外的空白
                            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0)
                            plt.tight_layout(pad=0)
                            
                            # 保存无文字的图像
                            plt.savefig(os.path.join(coronal_dir, f'{task}_coronal_slice_{slice_idx}_activation_{activation:.4f}_colormap_{cmap_name}.png'), 
                                      dpi=300, bbox_inches='tight', pad_inches=0)
                            plt.close()
                
                # 处理矢状面(Sagittal)视图 - 沿x轴切片
                print("生成矢状面(Sagittal)视图...")
                sagittal_nonzero_counts = []
                for slice_idx in range(x_np.shape[2]):
                    # 计算每个切片的非零像素数量
                    non_zero_count = np.count_nonzero(x_np[:, :, slice_idx] > 0.1)
                    sagittal_nonzero_counts.append((slice_idx, non_zero_count))
                
                # 按非零像素数排序
                sagittal_slices = sorted(sagittal_nonzero_counts, key=lambda x: x[1], reverse=True)
                
                # 选择有显著内容的切片(非零像素数大于阈值)
                significant_sagittal_slices = [(idx, count) for idx, count in sagittal_slices if count > 1000]
                
                if significant_sagittal_slices:
                    print(f"找到 {len(significant_sagittal_slices)} 个有显著内容的Sagittal切片")
                    # 选择中间40%~80%的切片范围
                    start_idx = int(len(significant_sagittal_slices) * 0.4)
                    end_idx = int(len(significant_sagittal_slices) * 0.8)
                    middle_sagittal_slices = [idx for idx, _ in significant_sagittal_slices[start_idx:end_idx]]
                    
                    if not middle_sagittal_slices:
                        middle_sagittal_slices = [idx for idx, _ in significant_sagittal_slices]
                else:
                    print("未找到有显著内容的Sagittal切片，使用中间范围切片")
                    # 使用中间1/3的切片
                    middle_sagittal_slices = list(range(x_np.shape[2]//3, x_np.shape[2]*2//3))
                
                # 在这些切片中寻找热力图激活最强的区域
                sagittal_activations = []
                for slice_idx in middle_sagittal_slices:
                    if slice_idx < cam_np.shape[2]:
                        # 计算该切片热力图的最大值
                        max_activation = np.max(cam_np[:, :, slice_idx])
                        if max_activation > 0:  # 只考虑有正激活的切片
                            sagittal_activations.append((slice_idx, max_activation))
                
                # 如果找到了有正激活的切片
                if sagittal_activations:
                    # 按激活值排序，选择前5个
                    top_sagittal_slices = sorted(sagittal_activations, key=lambda x: x[1], reverse=True)[:5]
                    print(f"选择的高激活Sagittal切片索引: {[idx for idx, _ in top_sagittal_slices]}")
                    
                    # 只保存这些高激活区域的切片
                    for slice_idx, activation in top_sagittal_slices:
                        # 获取单个切片
                        img_slice = x_np[:, :, slice_idx]
                        cam_slice = cam_np[:, :, slice_idx]
                        
                        # 确保每个切片的形状匹配
                        if img_slice.shape != cam_slice.shape:
                            # 如果仍然不匹配，调整热力图尺寸以匹配当前切片
                            cam_slice = cv2.resize(cam_slice, (img_slice.shape[1], img_slice.shape[0]))
                        
                        # 调整热图尺寸以匹配原图
                        cam_resized = cam_slice
                        
                        # 确保热力图正确归一化，避免空热力图
                        if np.max(cam_resized) > 0:
                            cam_resized = cam_resized / np.max(cam_resized)
                        else:
                            print(f"警告: Sagittal切片 {slice_idx} 的热力图全为零，跳过该切片")
                            continue
                            
                        # 为每种颜色映射创建可视化
                        for cmap_name, cmap in colormaps.items():
                            # 使用当前颜色映射转换为彩色图
                            heatmap_rgb = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
                            
                            # 将灰度图转为RGB以便叠加
                            img_rgb = np.stack([img_slice]*3, axis=2)
                            # 确保转换为uint8类型以匹配heatmap_rgb
                            img_rgb = (img_rgb * 255).astype(np.uint8)
                            
                            # 叠加原图和热图
                            overlay = cv2.addWeighted(img_rgb, 0.4, heatmap_rgb, 0.6, 0)
                            
                            # 创建并保存可视化结果
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            # 原始图像
                            axes[0].imshow(img_slice, cmap='gray')
                            axes[0].set_title('')
                            axes[0].axis('off')
                            
                            # GradCAM热图
                            axes[1].imshow(cam_resized, cmap=cmap)
                            axes[1].set_title('')
                            axes[1].axis('off')
                            
                            # 叠加结果
                            axes[2].imshow(overlay)
                            axes[2].set_title('')
                            axes[2].axis('off')
                            
                            # 移除所有边框和额外的空白
                            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0)
                            plt.tight_layout(pad=0)
                            
                            # 保存无文字的图像
                            plt.savefig(os.path.join(sagittal_dir, f'{task}_sagittal_slice_{slice_idx}_activation_{activation:.4f}_colormap_{cmap_name}.png'), 
                                      dpi=300, bbox_inches='tight', pad_inches=0)
                            plt.close()
                
                print(f"已为样本 {name} 生成Grad-CAM可视化（三个平面视图）")
                
        except Exception as e:
            print(f"为样本 {name} 生成Grad-CAM时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 移除钩子
    layer_hook.remove()
    print(f"\n完成 {task} 任务的Grad-CAM可视化生成（三个平面视图）")

if __name__ == '__main__':
    # 添加统计函数
    pass