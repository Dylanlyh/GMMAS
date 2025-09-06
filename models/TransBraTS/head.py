import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet_head(nn.Module):
    def __init__(self):
        super(Unet_head, self).__init__()
        self.UNET = Unet()
        self.MIDDLE_ONE = middle_one()
        self.DECODER = Decoder_modual()
    def forward(self, x):
        x1_1, x2_1, x3_1, x8, InitConv_vars = self.UNET(x)
        x = self.MIDDLE_ONE(x8)
        y, y4_1, y3_1, y2_1 = self.DECODER(x1_1, x2_1, x3_1, x)
        ROI = y.argmax(1)   #分割结果的四张特征图的最大概率值所在标签（第二个维度-0/1/2/3）
        return ROI, x, y, y4_1, y3_1, y2_1

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='bn'): #gn
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        return y



class Unet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=4):
        super(Unet, self).__init__()
        self.InitConv_vars = nn.Parameter(torch.tensor((1.0, 1.0, 1.0, 1.0),requires_grad=True))

        self.InitConv_flair = InitConv(in_channels=1, out_channels=base_channels, dropout=0.2)
        self.InitConv_ce = InitConv(in_channels=1, out_channels=base_channels, dropout=0.2)
        self.InitConv_t1 = InitConv(in_channels=1, out_channels=base_channels, dropout=0.2)
        self.InitConv_t2 = InitConv(in_channels=1, out_channels=base_channels, dropout=0.2)

        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1_flair = EnDown(in_channels=base_channels, out_channels=base_channels*2)
        self.EnDown1_ce = EnDown(in_channels=base_channels, out_channels=base_channels*2)
        self.EnDown1_t1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)
        self.EnDown1_t2 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2_flair = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)
        self.EnDown2_ce = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)
        self.EnDown2_t1 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)
        self.EnDown2_t2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3_flair = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)
        self.EnDown3_ce = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)
        self.EnDown3_t1 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)
        self.EnDown3_t2 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

    def forward(self, x):
        # print('input.shape:', x.shape)
        if x.shape[1] == 4:          # todo 在前向传播网络中加入了序列数目判断
            x_flair = x[:,0,:,:,:]    # 取出某一个序列自动降维
            x_flair = x_flair.unsqueeze(1)
            x_ce = x[:,1,:,:,:]
            x_ce = x_ce.unsqueeze(1)
            x_t1 = x[:,2,:,:,:]
            x_t1 = x_t1.unsqueeze(1)
            x_t2 = x[:,3,:,:,:]
            x_t2 = x_t2.unsqueeze(1)
            FLAIR, CE, T1, T2 = True, True, True, True
        elif x.shape[1] == 3:
            x_flair = x[:,0,:,:,:]
            x_flair = x_flair.unsqueeze(1)
            x_ce = x[:,1,:,:,:]
            x_ce = x_ce.unsqueeze(1)
            x_t1 = x[:,2,:,:,:]
            x_t1 = x_t1.unsqueeze(1)    
            FLAIR, CE, T1, T2 = True, True, True, False
        elif x.shape[1] == 2:
            x_flair = x[:,0,:,:,:]
            x_flair = x_flair.unsqueeze(1)
            x_ce = x[:,1,:,:,:]
            x_ce = x_ce.unsqueeze(1)
            FLAIR, CE, T1, T2 = True, True, False, False
        elif x.shape[1] == 1:
            x_flair = x[:,0,:,:,:]
            x_flair = x_flair.unsqueeze(1)
            FLAIR, CE, T1, T2 = True, False, False, False
        if FLAIR:
            x_flair = self.InitConv_flair(x_flair)       # (1, 16, 128, 128, 128)
        if CE:
            x_ce = self.InitConv_ce(x_ce)
        if T1:
            x_t1 = self.InitConv_t1(x_t1)
        if T2:
            x_t2 = self.InitConv_t2(x_t2)

        Init_flair_weight = torch.exp(self.InitConv_vars[0]-1) 
        Init_ce_weight = torch.exp(self.InitConv_vars[1]-1) 
        Init_t1_weight = torch.exp(self.InitConv_vars[2]-1) 
        Init_t2_weight = torch.exp(self.InitConv_vars[3]-1) 

        weights = [Init_flair_weight, Init_ce_weight, Init_t1_weight, Init_t2_weight]
        weights_sum = Init_flair_weight + Init_ce_weight + Init_t1_weight + Init_t2_weight

        # todo 概率随机将一个权重设置为0
        # if x.shape[1] == 4:
        #     if 0.2 < random.random() < 0.5:
        #         random_index = random.randint(0, len(weights) - 1)
        #         weights[random_index] = 0
        #     elif random.random() < 0.2:
        #         random_indices = random.sample(range(len(weights)), 2)
        #         for i in random_indices:
        #             weights[i] = 0
        
        # todo test time
        # weights[0] = 0  # todo缺失flair序列
        # weights[1] = 0     # todo缺失ce序列
        # weights[2] = 0     # todo缺失t1序列
        # weights[3] = 0     # todo缺失t2序列

        # todo 根据序列数目将对应通道相加，再求平均
        if T2:
            x = (weights[0]* x_flair + weights[1]* x_ce + weights[2]* x_t1 + weights[3]* x_t2) / (weights[0] + weights[1] + weights[2] + weights[3])
        elif T1 and not T2:
            x = (weights[0]* x_flair + weights[1]* x_ce + weights[2]* x_t1) / (weights[0] + weights[1] + weights[2])
        elif CE and not T1 and not T2:
            x = (weights[0]* x_flair + weights[1]* x_ce) / (weights[0] + weights[1])
        elif FLAIR and not CE and not T1 and not T2:
            x = x_flair

        x1_1 = self.EnBlock1(x)

        x1_2_flair = self.EnDown1_flair(x1_1)  # (1, 32, 64, 64, 64)
        x1_2_ce = self.EnDown1_ce(x1_1)
        x1_2_t1 = self.EnDown1_t1(x1_1)
        x1_2_t2 = self.EnDown1_t2(x1_1)
        x1_2 = (x1_2_flair + x1_2_ce + x1_2_t1 + x1_2_t2) / 4

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2_flair = self.EnDown2_flair(x2_1)  # (1, 64, 32, 32, 32)
        x2_2_ce = self.EnDown2_ce(x2_1)
        x2_2_t1 = self.EnDown2_t1(x2_1)
        x2_2_t2 = self.EnDown2_t2(x2_1)
        x2_2 = (x2_2_flair + x2_2_ce + x2_2_t1 + x2_2_t2) / 4

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2_flair = self.EnDown3_flair(x3_1)  # (1, 128, 16, 16, 16)
        x3_2_ce = self.EnDown3_ce(x3_1)
        x3_2_t1 = self.EnDown3_t1(x3_1)
        x3_2_t2 = self.EnDown3_t2(x3_1)
        x3_2 = (x3_2_flair + x3_2_ce + x3_2_t1 + x3_2_t2) / 4
        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)

        output = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        return x1_1,x2_1,x3_1,output,self.InitConv_vars


class middle_one(nn.Module):
    def __init__(self, in_channels=512, out_channels=128):
        super(middle_one, self).__init__()
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv_1 = nn.Conv3d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1
            )
        self.conv_2 = nn.Conv3d(
                256,
                512,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def forward(self, x):
        output = self.bn(x)
        output = self.relu(output)
        output = self.conv_1(output)    #（2，256，16，16，16）
        output = self.conv_2(output)    #（2，512，16，16，16）
        return output


class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()
        self.bn1 = nn.BatchNorm3d(512)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x
        return x1

class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x
        return x1

class Decoder_modual(nn.Module):
    def __init__(self):
        super(Decoder_modual, self).__init__()

        self.embedding_dim = 512

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)

        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 16) #32

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 16, out_channels=self.embedding_dim // 32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 32)  #16

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)

        self.Softmax = nn.Softmax(dim=1)

    def forward(self,x1_1, x2_1, x3_1, x8):
        return self.decode(x1_1, x2_1, x3_1, x8)

    def decode(self, x1_1, x2_1, x3_1, x8):

        x8 = self.Enblock8_1(x8)
        y4_1 = self.Enblock8_2(x8)

        y4 = self.DeUp4(y4_1, x3_1)  # (1, 64, 32, 32, 32)
        y3_1 = self.DeBlock4(y4)

        y3 = self.DeUp3(y3_1, x2_1)  # (1, 32, 64, 64, 64)
        y2_1 = self.DeBlock3(y3)

        y2 = self.DeUp2(y2_1, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        y = self.endconv(y2)      # (1, 4, 128, 128, 128)

        y = self.Softmax(y)

        return y, y4_1, y3_1, y2_1