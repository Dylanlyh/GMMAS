import torch,time
import torch.nn.functional as F
import setproctitle
from utils.tools import all_reduce_tensor
import logging
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np
from data.BraTS_IDH import BraTS
from torch.utils.data import DataLoader

def tailor_and_concat(x, model, idh, grade, mgmt, pq):
    temp = []
    idh_temp=[]
    grade_temp=[]
    mgmt_temp=[]
    pq_temp=[]
    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()
    #y = torch.cat((x,x[:,[0],:,:,:]),dim=1).clone()
    for i in range(len(temp)): 

        x1_1, x2_1, x3_1,x4_1, encoder_output = model['en'](temp[i])

        seg_output = model['seg'](x1_1, x2_1, x3_1,encoder_output)
        
        idh_out, grade_out, mgmt_out, pq_out = model['idh'](x4_1, encoder_output, idh, grade, mgmt ,pq) 

        temp[i] = seg_output
        idh_temp.append(idh_out)
        grade_temp.append(grade_out)
        mgmt_temp.append(mgmt_out)
        pq_temp.append(pq_out)

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    idh_out = torch.mean(torch.stack(idh_temp), dim=0)
    grade_out = torch.mean(torch.stack(grade_temp), dim=0)
    mgmt_out = torch.mean(torch.stack(mgmt_temp), dim=0)
    pq_out = torch.mean(torch.stack(pq_temp), dim=0)
    return y[..., :155], idh_out, grade_out, mgmt_out, pq_out

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
def ECE_score(y_pred, y_truth, n_bins=10):
    y_pred = np.array(y_pred)
    y_truth = np.array(y_truth)

    if y_truth.ndim > 1:
        y_truth = np.argmax(y_truth, axis=1)
    py_index = np.argmax(y_pred, axis=1)

    py_value = []
    for i in range(y_pred.shape[0]):
        py_value.append(y_pred[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(y_pred.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_truth[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def grade_uncertainty_ECE(pseudo_loader, DDP_model, names, args):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt

    DDP_model['en'].eval()
    DDP_model['seg'].eval()
    DDP_model['idh'].eval()

    # 新增代码 - 初始化用于收集不同uncertainty区间的样本
    uncertainty_bins = np.linspace(0, 1, 21)  # 创建0到1的11个点，包含0和1
    bin_samples = {bin_idx: [] for bin_idx in range(len(uncertainty_bins) - 1)}  # 每个区间的样本列表
    bin_truths = {bin_idx: [] for bin_idx in range(len(uncertainty_bins) - 1)}  # 每个区间的真实标签列表

    select_mask =[]
    grade_p_label =[]
    grade_n_label = []
    patients =[]
    grade_class = []
    confidence = []
    uncertainty = []
    grade_truth = []
    y_pred = []
    y_truth = []
    grade_pred_class= []

    f_pass = 10
    enable_dropout(DDP_model['en'])
    enable_dropout(DDP_model['seg'])
    enable_dropout(DDP_model['idh'])

    predicted_probs = []
    true_labels = []

    with torch.no_grad():
        # batchsize = 1
        for batch_id, (x, target, weight_seg, grade, idh, mgmt, _1p19q) in enumerate(pseudo_loader):
            x = x.cuda(non_blocking=True)
            grade = grade.cuda(non_blocking=True)

            out_prob = [] ## return to zero for each batch

            # true_labels.append(grade_.tolist())
            true_labels.append(grade.item())

            for _ in range(f_pass):
                x1_1, x2_1, x3_1,x4_1, encoder_output = DDP_model['en'](x)
                # seg_out = DDP_model['seg'](encoder_outs[0], encoder_outs[1], encoder_outs[2], encoder_outs[4])
                idh_out, grade_out,mgmt_out,pq_out = DDP_model['idh'](x4_1, encoder_output, idh, grade, mgmt, _1p19q)   
                grade_prob = F.softmax(grade_out, 1)  ## for selecting positive pseudo-labels
                out_prob.append(grade_prob) # ten times' probablities [one-dimension tensor containing each class] in out_prob list
            out_prob = torch.stack(out_prob) 

            out_std = torch.std(out_prob,dim=0)
            out_std = out_std.view(1,2)    # two classes' standard deviation

            out_prob = torch.mean(out_prob,dim=0)
            out_prob = out_prob.view(1,2)
            out_prob_1 = out_prob[0][1]
            predicted_probs.append(out_prob_1.cpu().numpy())
            class_pred = torch.argmax(out_prob, 1)

            max_value,max_idx = torch.max(out_prob,dim=1)
            max_std = out_std.gather(1,max_idx.view(-1,1))

            #selecting positive pseduo-labels

            selected_idx = (max_value>=args.tao_c)*(max_std.squeeze(1)<args.tao_u)

            # print('selected_idx:',selected_idx,"class_pred:",class_pred.item())
            if selected_idx and class_pred.item()==1:    # first if condition: idx value (if + n)
                label = 1
                grade_p_label.append(class_pred.item())
            elif selected_idx and class_pred.item()==0:   # "and" in here adds "if" condition
                label = 0
                grade_n_label.append(class_pred.item())
            else:
                label = -1

            if selected_idx:
                y_pred.append(out_prob[0].cpu().numpy())   # [0] here is the same use as squeeze(1)
                y_truth.append(grade.item())
                grade_pred_class.append(class_pred.item())


            grade_class.append(label)
            patients.append(names[batch_id])
            confidence.append(max_value.item())
            uncertainty.append(max_std.item())
            print("name:", names[batch_id], "selected_idx:", selected_idx, "max_value:", max_value.item(), "uncertainty:",
                  max_std.item(), "pred_grade:", class_pred.item(), "grade:", grade.item())
            grade_truth.append(grade.item())

            # 新增代码 - 将样本按uncertainty分组
            bin_idx = np.digitize(max_std.item(), uncertainty_bins) - 1  # 找到uncertainty所在的区间
            if 0 <= bin_idx < len(uncertainty_bins) - 1:
                bin_samples[bin_idx].append(out_prob[0].cpu().numpy())
                bin_truths[bin_idx].append(grade.item())
    
    # 新增代码 - 计算并绘制每个区间的ECE值
    ece_values = []
    for bin_idx in range(len(uncertainty_bins) - 1):
        if bin_samples[bin_idx]:
            ece = ECE_score(bin_samples[bin_idx], bin_truths[bin_idx])
            ece_values.append(ece)
        else:
            ece_values.append(None)

   # Plotting the ECE values
    plt.figure(figsize=(10, 8))
    plt.plot(uncertainty_bins[:4], ece_values[:4], marker='o')
    ece_data = []
    for x,y in zip(uncertainty_bins[:4], ece_values[:4]):
        ece_data.append([x,y])
    # Save ece_data as a text file
    np.savetxt('/public/home/hpc226511030/GMMAS/output/ece_grade_data.txt', ece_data)

    plt.xlim(-0.01, 0.16)
    plt.xlabel('Uncertainty value')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('Grade ECE vs Uncertainty')
    plt.savefig('/public/home/hpc226511030/GMMAS/output/Grade ece_values.png')

    import numpy as np
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    predicted_probs = np.array(predicted_probs)  # Replace with your model's predicted probabilities
    true_labels = np.array(true_labels)      # Replace with the actual labels
    print("predicted_probs:",predicted_probs.shape,"true_labels:",true_labels.shape)
    # The calibration curve function returns the mean predicted probability and the fraction of positives.
    mean_predicted_value, fraction_of_positives = calibration_curve(true_labels, predicted_probs, n_bins=10, strategy='quantile')

    # Plotting the calibration curve
    plt.figure(figsize=(10, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel('Observed Probability')
    plt.xlabel('Predicted Probability')
    plt.legend()
    plt.savefig('/public/home/hpc226511030/GMMAS/output/Grade cali_curve.png')
    for x,y in zip(mean_predicted_value, fraction_of_positives):
        ece_data.append([x,y])
    np.savetxt('/public/home/hpc226511030/GMMAS/output/cali_grade_data.txt', ece_data)

def IDH_uncertainty_ECE(pseudo_loader, DDP_model, names, args):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt

    DDP_model['en'].eval()
    DDP_model['seg'].eval()
    DDP_model['idh'].eval()

    # 新增代码 - 初始化用于收集不同uncertainty区间的样本
    uncertainty_bins = np.linspace(0, 1, 21)  # 创建0到1的11个点，包含0和1
    bin_samples = {bin_idx: [] for bin_idx in range(len(uncertainty_bins) - 1)}  # 每个区间的样本列表
    bin_truths = {bin_idx: [] for bin_idx in range(len(uncertainty_bins) - 1)}  # 每个区间的真实标签列表

    select_mask =[]
    IDH_p_label =[]
    IDH_n_label = []
    patients =[]
    IDH_class = []
    confidence = []
    uncertainty = []
    IDH_truth = []
    y_pred = []
    y_truth = []
    IDH_pred_class= []

    f_pass = 10
    enable_dropout(DDP_model['en'])
    enable_dropout(DDP_model['seg'])
    enable_dropout(DDP_model['idh'])

    predicted_probs = []
    true_labels = []

    with torch.no_grad():
        # batchsize = 1
        for batch_id, (x, target, weight_seg, grade, idh, mgmt, _1p19q) in enumerate(pseudo_loader):
            x = x.cuda(non_blocking=True)
            # target = target.cuda(args.local_rank,non_blocking=True)
            idh = idh.cuda(non_blocking=True)
            grade = grade.cuda(non_blocking=True)
            mgmt = mgmt.cuda(non_blocking=True)
            pq = _1p19q.cuda(non_blocking=True)
            model = DDP_model
            out_prob = [] ## return to zero for each batch

            true_labels.append(idh.item())

            for _ in range(f_pass):
                # x1_1, x2_1, x3_1,x4_1, encoder_output = DDP_model['en'](x)
                # seg_out = DDP_model['seg'](encoder_outs[0], encoder_outs[1], encoder_outs[2], encoder_outs[4])
                # x = x[..., :155]

                TTA_1,TTA_2,TTA_3,TTA_4,TTA_5,TTA_6,TTA_7,TTA_8 = tailor_and_concat(x, model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(2,)), model, idh, grade, mgmt, pq),\
                                                                    tailor_and_concat(x.flip(dims=(3,)), model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(4,)), model, idh, grade, mgmt, pq),\
                                                                    tailor_and_concat(x.flip(dims=(2, 3)), model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(2, 4)), model, idh, grade, mgmt, pq),\
                                                                    tailor_and_concat(x.flip(dims=(3, 4)), model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(2, 3, 4)), model, idh, grade, mgmt, pq)


                logit = F.softmax(TTA_1[0], 1)  # no flip
                logit += F.softmax(TTA_2[0].flip(dims=(2,)), 1)  # flip H
                logit += F.softmax(TTA_3[0].flip(dims=(3,)), 1)  # flip W
                logit += F.softmax(TTA_4[0].flip(dims=(4,)), 1)  # flip D
                logit += F.softmax(TTA_5[0].flip(dims=(2, 3)), 1)  # flip H, W
                logit += F.softmax(TTA_6[0].flip(dims=(2, 4)), 1)  # flip H, D
                logit += F.softmax(TTA_7[0].flip(dims=(3, 4)), 1)  # flip W, D
                logit += F.softmax(TTA_8[0].flip(dims=(2, 3, 4)), 1)  # flip H, W, D
                output = logit / 8.0   #TTA
                idh_probs = []
                grade_probs = []
                mgmt_probs = []
                pq_probs = []
                for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
                    idh_probs.append(F.softmax(pred[1],1))
                    grade_probs.append(F.softmax(pred[2],1))
                    mgmt_probs.append(F.softmax(pred[3],1))
                    pq_probs.append(F.softmax(pred[4],1))

                idh_pred = torch.mean(torch.stack(idh_probs),dim=0)
                grade_pred = torch.mean(torch.stack(grade_probs),dim=0)
                mgmt_pred = torch.mean(torch.stack(mgmt_probs),dim=0)
                pq_pred = torch.mean(torch.stack(pq_probs),dim=0)

                # IDH_out, grade_out,mgmt_out,pq_out = DDP_model['idh'](x4_1, encoder_output, idh, grade, mgmt, _1p19q)   
                # IDH_prob = F.softmax(IDH_out, 1)  ## for selecting positive pseudo-labels
                out_prob.append(idh_pred) # ten times' probablities [one-dimension tensor containing each class] in out_prob list
            out_prob = torch.stack(out_prob) 

            out_std = torch.std(out_prob,dim=0)
            out_std = out_std.view(1,2)    # two classes' standard deviation

            out_prob = torch.mean(out_prob,dim=0)
            out_prob = out_prob.view(1,2)
            out_prob_1 = out_prob[0][1]
            predicted_probs.append(out_prob_1.cpu().numpy())
            class_pred = torch.argmax(out_prob, 1)

            max_value,max_idx = torch.max(out_prob,dim=1)
            max_std = out_std.gather(1,max_idx.view(-1,1))

            #selecting positive pseduo-labels

            selected_idx = (max_value>=args.tao_c)*(max_std.squeeze(1)<args.tao_u)

            # print('selected_idx:',selected_idx,"class_pred:",class_pred.item())
            if selected_idx and class_pred.item()==1:    # first if condition: idx value (if + n)
                label = 1
                IDH_p_label.append(class_pred.item())
            elif selected_idx and class_pred.item()==0:   # "and" in here adds "if" condition
                label = 0
                IDH_n_label.append(class_pred.item())
            else:
                label = -1

            if selected_idx:
                y_pred.append(out_prob[0].cpu().numpy())   # [0] here is the same use as squeeze(1)
                y_truth.append(IDH.item())
                IDH_pred_class.append(class_pred.item())


            IDH_class.append(label)
            patients.append(names[batch_id])
            confidence.append(max_value.item())
            uncertainty.append(max_std.item())
            print("name:", names[batch_id], "selected_idx:", selected_idx, "max_value:", max_value.item(), "uncertainty:",
                  max_std.item(), "pred_IDH:", class_pred.item(), "IDH:", IDH.item())
            IDH_truth.append(IDH.item())

            # 新增代码 - 将样本按uncertainty分组
            bin_idx = np.digitize(max_std.item(), uncertainty_bins) - 1  # 找到uncertainty所在的区间
            if 0 <= bin_idx < len(uncertainty_bins) - 1:
                bin_samples[bin_idx].append(out_prob[0].cpu().numpy())
                bin_truths[bin_idx].append(IDH.item())
    
    # 新增代码 - 计算并绘制每个区间的ECE值
    ece_values = []
    for bin_idx in range(len(uncertainty_bins) - 1):
        if bin_samples[bin_idx]:
            ece = ECE_score(bin_samples[bin_idx], bin_truths[bin_idx])
            ece_values.append(ece)
        else:
            ece_values.append(None)

   # Plotting the ECE values
    plt.figure(figsize=(10, 8))
    plt.plot(uncertainty_bins[:4], ece_values[:4], marker='o')
    plt.xlim(-0.01, 0.16)
    plt.xlabel('Uncertainty value')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('IDH ECE vs Uncertainty')
    plt.savefig('/public/home/hpc226511030/GMMAS/output/IDH ece_values.png')
    ece_data = []
    for x,y in zip(uncertainty_bins[:4], ece_values[:4]):
        ece_data.append([x,y])
    # Save ece_data as a text file
    np.savetxt('/public/home/hpc226511030/GMMAS/output/idh ece_grade_data.txt', ece_data)


    import numpy as np
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    predicted_probs = np.array(predicted_probs)  # Replace with your model's predicted probabilities
    true_labels = np.array(true_labels)      # Replace with the actual labels
    print("predicted_probs:",predicted_probs.shape,"true_labels:",true_labels.shape)
    # The calibration curve function returns the mean predicted probability and the fraction of positives.
    mean_predicted_value, fraction_of_positives = calibration_curve(true_labels, predicted_probs, n_bins=10, strategy='quantile')

    # Plotting the calibration curve
    plt.figure(figsize=(10, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel('Observed Probability')
    plt.xlabel('Predicted Probability')
    plt.legend()
    plt.savefig('/public/home/hpc226511030/GMMAS/output/IDH cali_curve.png')
    for x,y in zip(mean_predicted_value, fraction_of_positives):
        ece_data.append([x,y])
    np.savetxt('/public/home/hpc226511030/GMMAS/output/idh cali_grade_data.txt', ece_data)

def pq_uncertainty_ECE(pseudo_loader, DDP_model, names, args):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt

    DDP_model['en'].eval()
    DDP_model['seg'].eval()
    DDP_model['idh'].eval()

    # 新增代码 - 初始化用于收集不同uncertainty区间的样本
    uncertainty_bins = np.linspace(0, 1, 21)  # 创建0到1的11个点，包含0和1
    bin_samples = {bin_idx: [] for bin_idx in range(len(uncertainty_bins) - 1)}  # 每个区间的样本列表
    bin_truths = {bin_idx: [] for bin_idx in range(len(uncertainty_bins) - 1)}  # 每个区间的真实标签列表

    select_mask =[]
    pq_p_label =[]
    pq_n_label = []
    patients =[]
    pq_class = []
    confidence = []
    uncertainty = []
    pq_truth = []
    y_pred = []
    y_truth = []
    pq_pred_class= []

    f_pass = 10
    enable_dropout(DDP_model['en'])
    enable_dropout(DDP_model['seg'])
    enable_dropout(DDP_model['idh'])

    predicted_probs = []
    true_labels = []

    with torch.no_grad():
        # batchsize = 1
        for batch_id, (x, target, weight_seg, grade, idh, mgmt, _1p19q) in enumerate(pseudo_loader):
            x = x.cuda(non_blocking=True)
            # target = target.cuda(args.local_rank,non_blocking=True)
            pq = _1p19q.cuda(non_blocking=True)

            out_prob = [] ## return to zero for each batch

            true_labels.append(pq.item())

            for _ in range(f_pass):
                x1_1, x2_1, x3_1,x4_1, encoder_output = DDP_model['en'](x)
                # seg_out = DDP_model['seg'](encoder_outs[0], encoder_outs[1], encoder_outs[2], encoder_outs[4])
                IDH_out, grade_out,mgmt_out,pq_out = DDP_model['idh'](x4_1, encoder_output, idh, grade, mgmt, _1p19q)   
                pq_prob = F.softmax(pq_out, 1, 2.186)  ## for selecting positive pseudo-labels
                out_prob.append(pq_prob) # ten times' probablities [one-dimension tensor containing each class] in out_prob list
            out_prob = torch.stack(out_prob) 

            out_std = torch.std(out_prob,dim=0)
            out_std = out_std.view(1,2)    # two classes' standard deviation

            out_prob = torch.mean(out_prob,dim=0)
            out_prob = out_prob.view(1,2)
            out_prob_1 = out_prob[0][1]
            predicted_probs.append(out_prob_1.cpu().numpy())
            class_pred = torch.argmax(out_prob, 1)

            max_value,max_idx = torch.max(out_prob,dim=1)
            max_std = out_std.gather(1,max_idx.view(-1,1))

            #selecting positive pseduo-labels

            selected_idx = (max_value>=args.tao_c)*(max_std.squeeze(1)<args.tao_u)

            # print('selected_idx:',selected_idx,"class_pred:",class_pred.item())
            if selected_idx and class_pred.item()==1:    # first if condition: idx value (if + n)
                label = 1
                pq_p_label.append(class_pred.item())
            elif selected_idx and class_pred.item()==0:   # "and" in here adds "if" condition
                label = 0
                pq_n_label.append(class_pred.item())
            else:
                label = -1

            if selected_idx:
                y_pred.append(out_prob[0].cpu().numpy())   # [0] here is the same use as squeeze(1)
                y_truth.append(pq.item())
                pq_pred_class.append(class_pred.item())


            pq_class.append(label)
            patients.append(names[batch_id])
            confidence.append(max_value.item())
            uncertainty.append(max_std.item())
            print("name:", names[batch_id], "selected_idx:", selected_idx, "max_value:", max_value.item(), "uncertainty:",
                  max_std.item(), "pred_pq:", class_pred.item(), "pq:", pq.item())
            pq_truth.append(pq.item())

            # 新增代码 - 将样本按uncertainty分组
            bin_idx = np.digitize(max_std.item(), uncertainty_bins) - 1  # 找到uncertainty所在的区间
            if 0 <= bin_idx < len(uncertainty_bins) - 1:
                bin_samples[bin_idx].append(out_prob[0].cpu().numpy())
                bin_truths[bin_idx].append(pq.item())
    
    # 新增代码 - 计算并绘制每个区间的ECE值
    ece_values = []
    for bin_idx in range(len(uncertainty_bins) - 1):
        if bin_samples[bin_idx]:
            ece = ECE_score(bin_samples[bin_idx], bin_truths[bin_idx])
            ece_values.append(ece)
        else:
            ece_values.append(None)

   # Plotting the ECE values
    plt.figure(figsize=(10, 8))
    plt.plot(uncertainty_bins[:4], ece_values[:4], marker='o')
    plt.xlim(-0.01, 0.16)
    plt.xlabel('Uncertainty value')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('1p/19q ECE vs Uncertainty')
    plt.savefig('/public/home/hpc226511030/GMMAS/output/pq ece_values.png')

    import numpy as np
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    predicted_probs = np.array(predicted_probs)  # Replace with your model's predicted probabilities
    true_labels = np.array(true_labels)      # Replace with the actual labels
    print("predicted_probs:",predicted_probs.shape,"true_labels:",true_labels.shape)
    # The calibration curve function returns the mean predicted probability and the fraction of positives.
    mean_predicted_value, fraction_of_positives = calibration_curve(true_labels, predicted_probs, n_bins=10, strategy='quantile')

    # Plotting the calibration curve
    plt.figure(figsize=(10, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel('Observed Probability')
    plt.xlabel('Predicted Probability')
    plt.legend()
    plt.savefig('/public/home/hpc226511030/GMMAS/output/pq cali_curve.png')

