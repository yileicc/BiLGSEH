import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
import logging
import os.path as osp
import pandas as pd

def bi_directional_similarity(label):
    n = label.size(0)
    k = label.size(1)
    xor_result = torch.zeros(n, n)

    for i in range(n):
        for j in range(i + 1, n):
            xor_result[i][j] = torch.bitwise_xor(label[i].int(), label[j].int()).sum()
            xor_result[j][i] = xor_result[i][j]    
    numerator = k - xor_result
    pos_result = numerator / k    
    neg_result = - xor_result / k

    indicator = (label.mm(label.t()) > 0) * 1
    sim_result = indicator * pos_result.cuda() + (1 - indicator) * neg_result.cuda()

    return sim_result

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def p_topK(qB, rB, query_label, retrieval_label, K=None):
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p

def p_topK2(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def compress(database_loader, test_loader, model_I, model_T):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(database_loader):
        with torch.no_grad():
            # var_data_I = Variable(data_I.cuda())
            var_data_I = Variable(F.normalize(data_I.cuda()))
            code_I = model_I(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        with torch.no_grad():
            # var_data_T = Variable(data_T.cuda())
            var_data_T = Variable(F.normalize(torch.FloatTensor(data_T.numpy()).cuda()))
            code_T = model_T(var_data_T.to(torch.float))
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
        with torch.no_grad():
            # var_data_I = Variable(data_I.cuda())
            var_data_I = Variable(F.normalize(data_I.cuda()))
            code_I = model_I(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        with torch.no_grad():
            # var_data_T = Variable(data_T.cuda())
            var_data_T = Variable(F.normalize(torch.FloatTensor(data_T.numpy()).cuda()))
            code_T = model_T(var_data_T.to(torch.float))
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.array(qu_L)
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def calculate_hamming(B1, B2):
    leng = B2.shape[1] 
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def calculate_map(qu_B, re_B, qu_L, re_L):
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, int(tsum)) 
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map

def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    qB = torch.tensor(qB)
    rB = torch.tensor(rB)
    query_label = torch.tensor(query_label)
    retrieval_label = torch.tensor(retrieval_label)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R

def write_tensors_to_excel(tensor1, tensor2, tensor3, tensor4, file_name):
    series1 = pd.Series(tensor1, name='precision(i2t)')
    series2 = pd.Series(tensor2, name='precision(t2i)')
    series3 = pd.Series(tensor3, name='recall(i2t)')
    series4 = pd.Series(tensor4, name='recall(t2i)')

    df = pd.concat([series1, series2, series3, series4], axis=1)
    df.to_excel(file_name, index=False, engine='openpyxl')

def logger(fileName='log'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = str(fileName) + '.txt'
    log_dir = './logs'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger

class TripletLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.reduction = reduction

    def cos_distance(self, source, target):
        cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)
        distances = torch.clamp(1 - cos_sim, 0)
        return distances

    def get_triplet_mask(self, s_labels, t_labels):
        flag = (1 - 0.1) * 1
        batch_size = s_labels.shape[0]
        sim_origin = s_labels.mm(t_labels.t())
        sim = (sim_origin > 0).float()
        ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
        ph = torch.arange(0., batch_size) + 2
        ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
        th = torch.log2(ph).cuda()
        Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
        sim_origin = 2 ** sim_origin - 1
        sim_origin = sim_origin / Z

        i_equal_j = sim.unsqueeze(2)
        i_equal_k = sim.unsqueeze(1)
        sim_pos = sim_origin.unsqueeze(2)
        sim_neg = sim_origin.unsqueeze(1)
        weight = (sim_pos - sim_neg) * (flag + 0.1)
        mask = i_equal_j * (1 - i_equal_k) * (flag + 0.1)

        return mask, weight

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0.2):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        pairwise_dist = self.cos_distance(source, target)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        mask, weight = self.get_triplet_mask(s_labels, t_labels)
        triplet_loss = mask * triplet_loss

        triplet_loss = triplet_loss.clamp(0)
        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).cuda())
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).cuda()).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)     
        z_j = F.normalize(emb_j, dim=1)     
        representations = torch.cat([z_i, z_j], dim=0)         
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  
        nominator = torch.exp(positives / self.temperature)            
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)       
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


