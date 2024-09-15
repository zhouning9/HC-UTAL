import torch
import torch.nn as nn
import numpy as np

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        # q:[B,2048]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['NA'], 1), 
            torch.mean(contrast_pairs['PA'], 1), 
            contrast_pairs['PB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['NB'], 1), 
            torch.mean(contrast_pairs['PB'], 1), 
            contrast_pairs['PA']
        )

        loss = HA_refinement + HB_refinement
        return loss
        
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.ce_criterion = nn.BCELoss()
        # 150 最好
        self.margin = 150

    def get_absloss(self,contrast_pairs):
        feat_act = contrast_pairs["PA"]
        feat_bkg = contrast_pairs['PB']
        
        loss_act = self.margin - \
                   torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um

    def get_tripletloss(self,label, contrast_pair):
        n = label.size(0) # get batch_size
        class_sim_idx =[]
        action_list = contrast_pair['PA']

        # 遍历20个类
        for i in range(label.size(1)):
            class_sim_idx.append(set())

        # get the same label
        # for i in range(n):
        #     for j in range(i+1,n):
        #         label0 = label[i].cpu().numpy()
        #         label1 = label[j].cpu().numpy()
        #         if (label0 == label1).all():
        #             l = label0.tolist()
        #             # idx = l.index(1)
        #             for idx in range(len(l)):
        #                 if l[idx]==1: 
        #                     class_sim_idx[idx].add(i)
        #                     class_sim_idx[idx].add(j)


        # 获得每个类别有那些视频
        for i in range(n):
            label0 = label[i].cpu().numpy()
            for j in range(len(label0)):
                if label0[j]==1:
                    class_sim_idx[j].add(i)

        triplet_loss = torch.FloatTensor([0.])
        
        # 计算距离
        distence = torch.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                action0 = torch.mean(action_list[i],dim=1)
                action1 = torch.mean(action_list[j],dim=1)
                d = 1- torch.sum(action0*action1,dim=0)/(torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0)) 
                distence[i][j] = d
                distence[j][i] = d

        # 遍历 20 个类别
        for i in range(label.size(1)):
            if len(class_sim_idx[i])<=1:
                # 如果第i个类别为0
                continue
            for idx in class_sim_idx[i]:
                max_d = torch.FloatTensor([0.])
                min_d = torch.max(distence[idx])
                # 寻找相同类别视频最大距离
                for j in class_sim_idx[i]:
                    if j!=idx:
                        # 不是视频自己
                        max_d = torch.max(distence[idx][j],max_d)
                # 寻找不同类别当中视频最小距离
                for j in range(n):
                    if j!=idx and j not in class_sim_idx[i]:
                        min_d = torch.min(distence[idx][j],min_d)
            
            triplet_loss = triplet_loss+ torch.max(max_d-min_d+0.8,torch.FloatTensor([0.]))

        return triplet_loss
    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)

        loss_abs = self.get_absloss(contrast_pairs)

        loss_triplet = self.get_tripletloss(label,contrast_pairs)[0]
        loss_total = loss_cls + 0.01 * loss_snico + 0.0005 * loss_abs + 0.005*loss_triplet

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Abs': loss_abs,
            'Loss/Triplet': loss_triplet,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class Totalloss6(nn.Module):
    def __init__(self):
        super(Totalloss6, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        # 150 最好
        self.margin = 150

    def NCE(self, q, k, neg, T=0.1):
        # q:[1,2048]
        # neg:[1,2048,61]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)  # k[1,2048]
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [1,1]
        l_neg = torch.einsum('nc,nck->nk', [q, neg])  # [1,c,2048]
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def get_separation_loss(self, contrast_pairs):
        feat_act = contrast_pairs["EA"]
        feat_bkg = contrast_pairs['EB']

        loss_act = self.margin - \
            torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_separation = torch.mean((loss_act + loss_bkg) ** 2)

        return loss_separation
    
    def get_ins_contrast_loss(self, label, contrast_pair):
        embeddings = contrast_pair['embeddings']
        n = label.size(0)  # get batch_size
        class_sim_idx = []
        # 遍历20个类
        for i in range(label.size(1)):
            class_sim_idx.append(set())

        # 获得每个类别有那些视频
        for i in range(n):
            label0 = label[i].cpu().numpy()
            for j in range(len(label0)):
                if label0[j] == 1:
                    class_sim_idx[j].add(i)

        # 计算距离
        distence = torch.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                action0 = torch.mean(embeddings[i], dim=1)
                action1 = torch.mean(embeddings[j], dim=1)
                d = 1 - torch.sum(action0*action1, dim=0) / \
                    (torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0))
                distence[i][j] = d
                distence[j][i] = d

        contrast_loss = torch.FloatTensor([0.]).cuda()
        # 遍历 20 个类别
        for i in range(label.size(1)):
            if len(class_sim_idx[i]) <= 1:
                # 如果第i个类别为0
                continue
            for idx in class_sim_idx[i]:
                min_d = torch.max(distence[idx])
                max_d = torch.FloatTensor([0.])

                q = torch.mean(embeddings[idx], dim=0)
                k = torch.mean(embeddings[idx], dim=0)

                neg = []
                # 寻找相同类别视频
                for j in class_sim_idx[i]:
                    if j != idx and distence[idx][j] < min_d:
                        # 不是视频本身
                        k = torch.mean(embeddings[j], dim=0)
                        min_d = distence[idx][j]

                    # if j != idx and distence_ha[idx][j] < min_d_ha:
                    #     # 不是视频本身
                    #     k_ha = torch.mean(ha[j], dim=0)
                    #     min_d_ha = distence_ha[idx][j]

                # 寻找不同类别
                for j in range(n):
                    if j != idx and j not in class_sim_idx[i]:
                        neg.append(torch.mean(embeddings[j], dim=0))

                neg = torch.stack(neg, 0)
                q = torch.unsqueeze(q, 0)
                k = torch.unsqueeze(k, 0)
                neg = torch.unsqueeze(neg, 0)
                neg = neg.permute(0, 2, 1)

                loss = self.NCE(q, k, neg)
                # loss_hb = self.NCE(q_hb, k_hb, neg_hb)
                contrast_loss = contrast_loss + loss

        return contrast_loss/n
    
    def get_feat_contrast_loss(self, label, contrast_pair):
        n = label.size(0)  # get batch_size
        class_sim_idx = []
        ea = contrast_pair['EA']
        ha = contrast_pair['HA']
        eb = contrast_pair['EB']
        hb = contrast_pair['HB']

        # 遍历20个类
        for i in range(label.size(1)):
            class_sim_idx.append(set())

        # 获得每个类别有那些视频
        for i in range(n):
            label0 = label[i].cpu().numpy()
            for j in range(len(label0)):
                if label0[j] == 1:
                    class_sim_idx[j].add(i)

        # 计算距离
        distence = torch.zeros((n, n))
        distence_ha = torch.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                action0 = torch.mean(ea[i], dim=1)
                action1 = torch.mean(ea[j], dim=1)
                d = 1 - torch.sum(action0*action1, dim=0) / \
                    (torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0))
                distence[i][j] = d
                distence[j][i] = d

                action0 = torch.mean(ha[i], dim=1)
                action1 = torch.mean(ha[j], dim=1)
                d = 1 - torch.sum(action0*action1, dim=0) / \
                    (torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0))
                distence_ha[i][j] = d
                distence_ha[j][i] = d

        contrast_loss = torch.FloatTensor([0.]).cuda()
        # 遍历 20 个类别
        for i in range(label.size(1)):
            if len(class_sim_idx[i]) <= 1:
                # 如果第i个类别为0
                continue
            for idx in class_sim_idx[i]:
                min_d = torch.max(distence[idx])
                max_d = torch.FloatTensor([0.])

                # min_d_ha = torch.max(distence_ha[idx])
                # max_d_ha = torch.FloatTensor([0.])

                q_ea = torch.mean(ea[idx], dim=0)
                k_ea = torch.mean(ha[idx], dim=0)

                q_ha = torch.mean(ha[idx], dim=0)
                k_ha = torch.mean(ea[idx], dim=0)
                neg_ea = []

                q_hb = torch.mean(hb[idx], dim=0)
                k_hb = torch.mean(eb[idx], dim=0)
                neg_hb = torch.cat([ea[idx], ha[idx]], dim=0)

                # 寻找相同类别视频
                for j in class_sim_idx[i]:
                    neg_ea.append(torch.mean(eb[j], dim=0))
                    neg_ea.append(torch.mean(hb[j], dim=0))

                    # if j != idx and distence[idx][j] > max_d:
                    #     # 不是视频本身
                    #     k_ea = torch.mean(ea[j], dim=0)
                    #     max_d = distence[idx][j]

                    if j != idx and distence[idx][j] < min_d:
                        # 不是视频本身
                        k_ea = torch.mean(ea[j], dim=0)
                        min_d = distence[idx][j]

                    # if j != idx and distence_ha[idx][j] < min_d_ha:
                    #     # 不是视频本身
                    #     k_ha = torch.mean(ha[j], dim=0)
                    #     min_d_ha = distence_ha[idx][j]

                # 寻找不同类别
                for j in range(n):
                    if j != idx and j not in class_sim_idx[i]:
                        neg_ea.append(torch.mean(ea[j], dim=0))
                        neg_ea.append(torch.mean(ha[j], dim=0))
                        neg_ea.append(torch.mean(eb[j], dim=0))
                        neg_ea.append(torch.mean(hb[j], dim=0))

                neg_ea = torch.stack(neg_ea, 0)
                q_ea = torch.unsqueeze(q_ea, 0)
                k_ea = torch.unsqueeze(k_ea, 0)
                neg_ea = torch.unsqueeze(neg_ea, 0)
                neg_ea = neg_ea.permute(0, 2, 1)

                q_ha = torch.unsqueeze(q_ha, 0)
                k_ha = torch.unsqueeze(k_ha, 0)

                q_hb = torch.unsqueeze(q_hb, 0)
                k_hb = torch.unsqueeze(k_hb, 0)
                neg_hb = torch.unsqueeze(neg_hb, 0)
                neg_hb = neg_hb.permute(0, 2, 1)

                loss_ea = self.NCE(q_ea, k_ea, neg_ea)
                loss_ha = self.NCE(q_ha, k_ha, neg_ea)
                # loss_hb = self.NCE(q_hb, k_hb, neg_hb)
                contrast_loss = contrast_loss + loss_ea + loss_ha

        return contrast_loss/n

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        # loss_snico = self.snico_criterion(contrast_pairs)

        loss_separation = self.get_separation_loss(contrast_pairs)

        loss_ins = self.get_ins_contrast_loss(label, contrast_pairs)[0]
        loss_feat = self.get_feat_contrast_loss(label, contrast_pairs)[0]
        # loss_total = loss_cls + 0.001 * loss_separation + 0.015 * loss_feat + 0.01* loss_ins
        loss_total = loss_cls + 0. * loss_separation + 0. * loss_feat + 0.01 * loss_ins
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/Separation': loss_separation,
            'Loss/Ins': loss_ins,
            'Loss/Feat': loss_feat,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict


class Totalloss7(nn.Module):
    def __init__(self):
        super(Totalloss7, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        # 150 最好
        self.margin = 150
        # dwa
        self.avg_cost = torch.zeros([10000,4]).float()
        self.dwa_t = 3.0

        self.cls_weight = torch.FloatTensor([1.0])
        self.ins_weight = torch.FloatTensor([0.01])
        self.feat_weight = torch.FloatTensor([0.015])
        self.abs_weight = torch.FloatTensor([0.001])
        
        self.step = 0

    def NCE(self, q, k, neg, T=0.1):
        # q:[1,2048]
        # neg:[1,2048,61]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)  # k[1,2048]
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [1,1]
        l_neg = torch.einsum('nc,nck->nk', [q, neg])  # [1,c,2048]
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def get_separation_loss(self, contrast_pairs):
        feat_act = contrast_pairs["EA"]
        feat_bkg = contrast_pairs['EB']

        loss_act = self.margin - \
            torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_separation = torch.mean((loss_act + loss_bkg) ** 2)

        return loss_separation
    
    def get_ins_contrast_loss(self, label, contrast_pair):
        embeddings = contrast_pair['embeddings']
        n = label.size(0)  # get batch_size
        class_sim_idx = []
        # 遍历20个类
        for i in range(label.size(1)):
            class_sim_idx.append(set())

        # 获得每个类别有那些视频
        for i in range(n):
            label0 = label[i].cpu().numpy()
            for j in range(len(label0)):
                if label0[j] == 1:
                    class_sim_idx[j].add(i)

        # 计算距离
        distence = torch.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                action0 = torch.mean(embeddings[i], dim=1)
                action1 = torch.mean(embeddings[j], dim=1)
                d = 1 - torch.sum(action0*action1, dim=0) / \
                    (torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0))
                distence[i][j] = d
                distence[j][i] = d

        contrast_loss = torch.FloatTensor([0.]).cuda()
        # 遍历 20 个类别
        for i in range(label.size(1)):
            if len(class_sim_idx[i]) <= 1:
                # 如果第i个类别为0
                continue
            for idx in class_sim_idx[i]:
                min_d = torch.max(distence[idx])
                max_d = torch.FloatTensor([0.])

                q = torch.mean(embeddings[idx], dim=0)
                k = torch.mean(embeddings[idx], dim=0)

                neg = []
                # 寻找相同类别视频
                for j in class_sim_idx[i]:
                    if j != idx and distence[idx][j] < min_d:
                        # 不是视频本身
                        k = torch.mean(embeddings[j], dim=0)
                        min_d = distence[idx][j]

                    # if j != idx and distence_ha[idx][j] < min_d_ha:
                    #     # 不是视频本身
                    #     k_ha = torch.mean(ha[j], dim=0)
                    #     min_d_ha = distence_ha[idx][j]

                # 寻找不同类别
                for j in range(n):
                    if j != idx and j not in class_sim_idx[i]:
                        neg.append(torch.mean(embeddings[j], dim=0))

                neg = torch.stack(neg, 0)
                q = torch.unsqueeze(q, 0)
                k = torch.unsqueeze(k, 0)
                neg = torch.unsqueeze(neg, 0)
                neg = neg.permute(0, 2, 1)

                loss = self.NCE(q, k, neg)
                # loss_hb = self.NCE(q_hb, k_hb, neg_hb)
                contrast_loss = contrast_loss + loss

        return contrast_loss/n
    
    def get_feat_contrast_loss(self, label, contrast_pair):
        n = label.size(0)  # get batch_size
        class_sim_idx = []
        ea = contrast_pair['EA']
        ha = contrast_pair['HA']
        eb = contrast_pair['EB']
        hb = contrast_pair['HB']

        # 遍历20个类
        for i in range(label.size(1)):
            class_sim_idx.append(set())

        # 获得每个类别有那些视频
        for i in range(n):
            label0 = label[i].cpu().numpy()
            for j in range(len(label0)):
                if label0[j] == 1:
                    class_sim_idx[j].add(i)

        # 计算距离
        distence = torch.zeros((n, n))
        distence_ha = torch.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                action0 = torch.mean(ea[i], dim=1)
                action1 = torch.mean(ea[j], dim=1)
                d = 1 - torch.sum(action0*action1, dim=0) / \
                    (torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0))
                distence[i][j] = d
                distence[j][i] = d

                action0 = torch.mean(ha[i], dim=1)
                action1 = torch.mean(ha[j], dim=1)
                d = 1 - torch.sum(action0*action1, dim=0) / \
                    (torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0))
                distence_ha[i][j] = d
                distence_ha[j][i] = d

        contrast_loss = torch.FloatTensor([0.]).cuda()
        # 遍历 20 个类别
        for i in range(label.size(1)):
            if len(class_sim_idx[i]) <= 1:
                # 如果第i个类别为0
                continue
            for idx in class_sim_idx[i]:
                min_d = torch.max(distence[idx])
                max_d = torch.FloatTensor([0.])

                # min_d_ha = torch.max(distence_ha[idx])
                # max_d_ha = torch.FloatTensor([0.])

                q_ea = torch.mean(ea[idx], dim=0)
                k_ea = torch.mean(ha[idx], dim=0)

                q_ha = torch.mean(ha[idx], dim=0)
                k_ha = torch.mean(ea[idx], dim=0)
                neg_ea = []

                q_hb = torch.mean(hb[idx], dim=0)
                k_hb = torch.mean(eb[idx], dim=0)
                neg_hb = torch.cat([ea[idx], ha[idx]], dim=0)

                # 寻找相同类别视频
                for j in class_sim_idx[i]:
                    neg_ea.append(torch.mean(eb[j], dim=0))
                    neg_ea.append(torch.mean(hb[j], dim=0))

                    # if j != idx and distence[idx][j] > max_d:
                    #     # 不是视频本身
                    #     k_ea = torch.mean(ea[j], dim=0)
                    #     max_d = distence[idx][j]

                    if j != idx and distence[idx][j] < min_d:
                        # 不是视频本身
                        k_ea = torch.mean(ea[j], dim=0)
                        min_d = distence[idx][j]

                    # if j != idx and distence_ha[idx][j] < min_d_ha:
                    #     # 不是视频本身
                    #     k_ha = torch.mean(ha[j], dim=0)
                    #     min_d_ha = distence_ha[idx][j]

                # 寻找不同类别
                for j in range(n):
                    if j != idx and j not in class_sim_idx[i]:
                        neg_ea.append(torch.mean(ea[j], dim=0))
                        neg_ea.append(torch.mean(ha[j], dim=0))
                        neg_ea.append(torch.mean(eb[j], dim=0))
                        neg_ea.append(torch.mean(hb[j], dim=0))

                neg_ea = torch.stack(neg_ea, 0)
                q_ea = torch.unsqueeze(q_ea, 0)
                k_ea = torch.unsqueeze(k_ea, 0)
                neg_ea = torch.unsqueeze(neg_ea, 0)
                neg_ea = neg_ea.permute(0, 2, 1)

                q_ha = torch.unsqueeze(q_ha, 0)
                k_ha = torch.unsqueeze(k_ha, 0)

                q_hb = torch.unsqueeze(q_hb, 0)
                k_hb = torch.unsqueeze(k_hb, 0)
                neg_hb = torch.unsqueeze(neg_hb, 0)
                neg_hb = neg_hb.permute(0, 2, 1)

                loss_ea = self.NCE(q_ea, k_ea, neg_ea)
                loss_ha = self.NCE(q_ha, k_ha, neg_ea)
                # loss_hb = self.NCE(q_hb, k_hb, neg_hb)
                contrast_loss = contrast_loss + loss_ea + loss_ha

        return contrast_loss/n

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        # loss_snico = self.snico_criterion(contrast_pairs)

        loss_separation = self.get_separation_loss(contrast_pairs)

        loss_ins = self.get_ins_contrast_loss(label, contrast_pairs)[0]
        loss_feat = self.get_feat_contrast_loss(label, contrast_pairs)[0]
        #loss_total = loss_cls + 0.001 * loss_separation + 0.015 * loss_feat + 0.01* loss_ins
        if self.step>2:
            cls_w = self.avg_cost[self.step - 1, 0] / self.avg_cost[self.step - 2, 0]
            ins_w = self.avg_cost[self.step - 1, 1] / self.avg_cost[self.step - 2, 1]
            feat_w = self.avg_cost[self.step - 1, 2] / self.avg_cost[self.step - 2, 2]
            # abs_w = self.avg_cost[self.step - 1, 3] / self.avg_cost[self.step - 2, 3]
            # self.cls_weight = 2 * torch.exp(cls_w / self.dwa_t) / (torch.exp(cls_w / self.dwa_t) + torch.exp(ins_w / self.dwa_t)+ torch.exp(feat_w / self.dwa_t)+torch.exp(abs_w / self.dwa_t))
            self.cls_weight = 2 * torch.exp(cls_w / self.dwa_t) / (torch.exp(cls_w / self.dwa_t) + torch.exp(ins_w / self.dwa_t)+ torch.exp(feat_w / self.dwa_t))
            self.ins_weight = 2 * torch.exp(ins_w / self.dwa_t) / (torch.exp(cls_w / self.dwa_t) + torch.exp(ins_w / self.dwa_t)+ torch.exp(feat_w / self.dwa_t))
            self.feat_weight = 2 * torch.exp(feat_w / self.dwa_t) / (torch.exp(cls_w / self.dwa_t) + torch.exp(ins_w / self.dwa_t)+ torch.exp(feat_w / self.dwa_t))
            # self.abs_weight = 2 * torch.exp(abs_w / self.dwa_t) / (torch.exp(cls_w / self.dwa_t) + torch.exp(ins_w / self.dwa_t)+ torch.exp(feat_w / self.dwa_t)+torch.exp(abs_w / self.dwa_t))

        self.avg_cost[self.step, 0] = loss_cls.detach()
        self.avg_cost[self.step, 1] = loss_ins.detach()
        self.avg_cost[self.step, 2] = loss_feat.detach()
        self.avg_cost[self.step, 3] = loss_separation.detach()

        loss_total = self.cls_weight.cuda() * loss_cls + self.ins_weight.cuda() * loss_ins +  self.feat_weight.cuda() * loss_feat + self.abs_weight.cuda() * loss_separation  
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/Separation': loss_separation,
            'Loss/Ins': loss_ins,
            'Loss/Feat': loss_feat,
            'Loss/Total': loss_total
        }
        self.step+=1;
        return loss_total, loss_dict