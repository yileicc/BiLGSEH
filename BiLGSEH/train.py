import torch
import torch.nn.functional as F
from load_data import load_dataset
from torch.autograd import Variable
from model import MyNet, Img_Net, Txt_Net, Discriminator
from utils import compress, calculate_top_map, bi_directional_similarity, ContrastiveLoss, pr_curve
import numpy as np
import os.path as osp
from torch import autograd

class train_net:
    def __init__(self, log, config):
        self.config = config
        self.log = log

        dataloader, data_train = load_dataset(config.DATASET, config.BATCH_SIZE)

        self.I_tr, self.T_tr, self.L_tr = data_train

        self.train_num = self.I_tr.shape[0]
        self.img_feat_len = self.I_tr.shape[1]
        self.txt_feat_len = self.T_tr.shape[1]
        self.lab_feat_len = self.L_tr.shape[1]

        self.train_loader = dataloader['train']
        self.test_loader = dataloader['query']
        self.database_loader = dataloader['retrieval']

        self.mynet = MyNet(code_len=self.config.HASH_BIT, ori_featI=self.img_feat_len, ori_featT=self.txt_feat_len, ori_featL=self.lab_feat_len).cuda()
        self.imgnet = Img_Net(code_len=self.config.HASH_BIT, img_feat_len=self.img_feat_len).cuda()
        self.txtnet = Txt_Net(code_len=self.config.HASH_BIT, txt_feat_len=self.txt_feat_len).cuda()

        self.opt_mynet = torch.optim.Adam(self.mynet.parameters(), lr=config.LR_MyNet)
        self.opt_imgnet = torch.optim.Adam(self.imgnet.parameters(), lr=config.LR_IMG)
        self.opt_txtnet = torch.optim.Adam(self.txtnet.parameters(), lr=config.LR_TXT)

        self.record_Lsmodel = []
        self.record_Lshfunc = []

        self.ContrastiveLoss = ContrastiveLoss(batch_size=config.BATCH_SIZE, temperature=config.t)

    def train_hashcode(self, epoch):
        coll_BI = list([])
        coll_BT = list([])
        coll_sim = list([])
        record_index = list([])
        Ls_method = 0

        self.mynet.train()

        for No, (img, txt, lab, index_) in enumerate(self.train_loader):

            img = Variable(img.cuda().to(torch.float))
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            lab = Variable(torch.FloatTensor(lab.numpy()).cuda())

            S_batch = bi_directional_similarity(lab)

            HI, HT = self.mynet(img, txt, lab, S_batch)
            BI = torch.sign(HI)
            BT = torch.sign(HT)

            coll_BI.extend(BI.cpu().data.numpy())
            coll_BT.extend(BT.cpu().data.numpy())
            coll_sim.extend(S_batch.cpu().data.numpy())
            record_index.extend(index_)

            ################ phase 1 ##################

            self.opt_mynet.zero_grad()

            loss_qua = F.mse_loss(HI, BI) + F.mse_loss(HT, BT)
            HI_norm = F.normalize(HI)
            HT_norm = F.normalize(HT)
            HI_HI = HI_norm.mm(HI_norm.t())
            HT_HT = HT_norm.mm(HT_norm.t())
            loss_sim = self.config.lambda2 * (F.mse_loss(S_batch, HI_HI) + F.mse_loss(S_batch, HT_HT))    
            loss_con = self.config.lambda1 * self.ContrastiveLoss(HI, HT)   

            loss_model = loss_qua + loss_sim + loss_con
            Ls_method = (Ls_method + loss_model).item()

            loss_model.backward()
            self.opt_mynet.step()

            if (No + 1) % (self.train_num // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL) == 0:
                self.log.info('Epoch [%d/%d], Iter [%d/%d] loss_qua=%.4f, loss_sim=%.4f, loss_con=%.4f, Ls_model: %.4f'
                                 % (epoch + 1, self.config.NUM_EPOCH, No + 1, self.train_num // self.config.BATCH_SIZE,
                                     loss_qua, loss_sim, loss_con, loss_model.item()))

        coll_BI = np.array(coll_BI)
        coll_BT = np.array(coll_BT)
        coll_sim = np.array(coll_sim)
        self.record_Lsmodel.append(Ls_method)

        return coll_BI, coll_BT, coll_sim, record_index

    def train_Hashfunc(self, coll_BI, coll_BT, coll_sim, record_index, epoch):

        self.imgnet.train()
        self.txtnet.train()

        Ls_hfunc = 0

        BI = torch.from_numpy(coll_BI).cuda()
        BT = torch.from_numpy(coll_BT).cuda()
        S_sim = torch.from_numpy(coll_sim).cuda()

        img = torch.Tensor(self.I_tr[record_index,:]).cuda()
        txt = torch.Tensor(self.T_tr[record_index,:]).cuda()

        img_norm = F.normalize(img)
        txt_norm = F.normalize(txt)

        num_cyc = img_norm.shape[0] / self.config.BATCH_SIZE
        num_cyc = int(num_cyc+1) if num_cyc-int(num_cyc)>0 else int(num_cyc)

        for kk in range(num_cyc):
            if kk+1 < num_cyc:
                img_batch = img_norm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
                txt_batch = txt_norm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
                BI_batch = BI[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
                BT_batch = BT[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
                sim_batch = S_sim[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
            else:
                img_batch = img_norm[kk * self.config.BATCH_SIZE:, :]
                txt_batch = txt_norm[kk * self.config.BATCH_SIZE:, :]
                BI_batch = BI[kk * self.config.BATCH_SIZE:, :]
                BT_batch = BT[kk * self.config.BATCH_SIZE:, :]
                sim_batch = S_sim[kk * self.config.BATCH_SIZE:, :]

            hfunc_BI = self.imgnet(img_batch)
            hfunc_BT = self.txtnet(txt_batch)

            self.opt_imgnet.zero_grad()
            self.opt_txtnet.zero_grad()

            loss_f1 = F.mse_loss(hfunc_BI, BI_batch) + F.mse_loss(hfunc_BT, BT_batch) + F.mse_loss(hfunc_BI, hfunc_BT)

            S_BI_BT = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BT).t())
            S_BI_BI = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BI).t())
            S_BT_BT = F.normalize(hfunc_BT).mm(F.normalize(hfunc_BT).t())

            loss_f2 = F.mse_loss(S_BI_BT, sim_batch) + F.mse_loss(S_BI_BI, sim_batch) + F.mse_loss(S_BT_BT, sim_batch)

            loss_hfunc = loss_f1 + self.config.beta * loss_f2
            Ls_hfunc = (Ls_hfunc + loss_hfunc).item()

            loss_hfunc.backward()

            self.opt_imgnet.step()
            self.opt_txtnet.step()

        self.log.info('Epoch [%d/%d], Ls_hfunc: %.4f' % (epoch + 1, self.config.NUM_EPOCH, loss_hfunc.item()))

        self.record_Lshfunc.append(Ls_hfunc)

        return self.imgnet, self.txtnet

    def performance_eval(self):

        self.log.info('--------------------Evaluation: mAP@50-------------------')
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.imgnet, self.txtnet)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        return MAP_I2T, MAP_T2I

    def performance_eval_PR(self):

        self.log.info('--------------------Evaluation:PR covers_mAP@50-------------------')
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.imgnet, self.txtnet)
        P_I2T, R_I2T = pr_curve(qu_BI, re_BT, qu_L, re_L)
        P_T2I, R_T2I = pr_curve(qu_BT, re_BI, qu_L, re_L)
        return P_I2T, R_I2T, P_T2I, R_T2I

    def save_checkpoints(self):
        file_name = self.config.DATASET + '_' + str(self.config.HASH_BIT) + 'bits.pth'
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.imgnet.state_dict(),
            'TxtNet': self.txtnet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.log.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.log.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError

        self.imgnet.load_state_dict(obj['ImgNet'])
        self.txtnet.load_state_dict(obj['TxtNet'])

