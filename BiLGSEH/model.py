import torch
import torch.nn as nn
import math
import torch.nn.init as init
import torch.nn.functional as F

########## The first phase of hash code learning ##########

class MyNet(nn.Module):
    def __init__(self, code_len, ori_featI, ori_featT, ori_featL):
        super(MyNet, self).__init__()
        self.code_len = code_len

        self.EncoderImg = nn.Sequential(nn.Linear(ori_featI, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.AttentionLayerImg = AttentionLayer(512, ori_featL, 512)
        self.FcImg = nn.Linear(2 * 512, 512)

        self.EncoderTxt = nn.Sequential(nn.Linear(ori_featT, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.AttentionLayerTxt = AttentionLayer(512, ori_featL, 512)
        self.FcTxt = nn.Linear(2 * 512, 512)

        self.gcn = nn.Linear(512, 512)
        self.BNgcn = nn.BatchNorm1d(512)
        self.actgcn = nn.ReLU(inplace=True)

        self.hashimg = nn.Linear(2 * 512, code_len)
        self.hashtxt = nn.Linear(2 * 512, code_len)
        self.HIBN = nn.BatchNorm1d(code_len)
        self.HTBN = nn.BatchNorm1d(code_len)

    def forward(self, img, txt, label, affinity_A):
        self.batch_size = img.size(0)

        img_coarse = self.EncoderImg(img)
        img_fine = self.AttentionLayerImg(img, label)
        img_feature = self.FcImg(torch.cat((img_coarse, img_fine), 1))   

        txt_coarse = self.EncoderTxt(txt)
        txt_fine = self.AttentionLayerTxt(txt, label)
        txt_feature = self.FcTxt(torch.cat((txt_coarse, txt_fine), 1))  


        img_feature_norm = F.normalize(img_feature, dim=1)    
        txt_feature_norm = F.normalize(txt_feature, dim=1)    
        VC = torch.cat((img_feature_norm, txt_feature_norm), 0)    
        II = torch.eye(affinity_A.shape[0], affinity_A.shape[1]).cuda()   
        S_cma = torch.cat((torch.cat((affinity_A, II), 1), torch.cat((II, affinity_A), 1)), 0)    

        VJ = self.gcn(VC)
        VJ = S_cma.mm(VJ)
        VJ = self.BNgcn(VJ)
        VJ = VJ[:self.batch_size, :] + VJ[self.batch_size:, :]
        VJ = self.actgcn(VJ)    

        HI = self.HIBN(self.hashimg(torch.cat((VJ, img_feature), 1)))
        HT = self.HTBN(self.hashimg(torch.cat((VJ, txt_feature), 1)))

        return  HI, HT



########## Hash function learning in the second phase ###########

class Img_Net(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(Img_Net, self).__init__()

        self.fc1 = nn.Linear(img_feat_len, img_feat_len // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(img_feat_len // 2, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HI = self.tanh(hid)

        return HI

class Txt_Net(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(Txt_Net, self).__init__()

        self.fc1 = nn.Linear(txt_feat_len, txt_feat_len // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(txt_feat_len // 2, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HT = self.tanh(hid)

        return HT


########## The labels guide the attention module ###########

class AttentionLayer(nn.Module):
    def __init__(self, data_dim, label_dim, hidden_dim, n_heads=4):
        super(AttentionLayer, self).__init__()

        assert hidden_dim % n_heads == 0

        self.data_dim = data_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(label_dim, hidden_dim)
        self.fc_k = nn.Linear(data_dim, hidden_dim)
        self.fc_v = nn.Linear(data_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        self.dense = nn.Linear(hidden_dim, data_dim)    
        self.bn = nn.BatchNorm1d(data_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data_tensor, label_tensor):
        batch_size = data_tensor.shape[0]

        Q = self.fc_q(label_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        K = self.fc_k(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        V = self.fc_v(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()

        att_map = torch.softmax((torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale), dim=-1)
        output = torch.matmul(att_map, V).view(batch_size, -1)

        output = self.dense(output)
        output = self.bn(output)
        output = self.relu(output)

        return output


