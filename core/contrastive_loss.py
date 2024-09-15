import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.tau = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def forward(self, z_i, z_j):
        # z_i [8,128] ,z_j [8,128]
        # N = 2 * self.batch_size
        # z = torch.cat((z_i, z_j), dim=0)

        # sim = torch.matmul(z, z.T) / self.temperature
        # sim_i_j = torch.diag(sim, self.batch_size)
        # sim_j_i = torch.diag(sim, -self.batch_size)

        # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative_samples = sim[self.mask].reshape(N, -1)

        # labels = torch.zeros(N).to(positive_samples.device).long()
        # logits = torch.cat((positive_samples, negative_samples), dim=1)
        # loss = self.criterion(logits, labels)
        # loss /= N

        loss = self.semi_loss(z_i, z_j)
        loss = loss.sum()
        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.tau = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def forward(self, c_i, c_j):
        # c_i[8,10] c_j[8,10]
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        # c_i = c_i.t()
        # c_j = c_j.t()
        # N = 2 * self.class_num
        # c = torch.cat((c_i, c_j), dim=0)

        # sim = self.similarity_f(c.unsqueeze(
        #     1), c.unsqueeze(0)) / self.temperature
        # sim_i_j = torch.diag(sim, self.class_num)
        # sim_j_i = torch.diag(sim, -self.class_num)

        # positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative_clusters = sim[self.mask].reshape(N, -1)

        # labels = torch.zeros(N).to(positive_clusters.device).long()
        # logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        # loss = self.criterion(logits, labels)
        # loss /= N
        loss = self.semi_loss(c_i, c_j)
        loss = loss.sum()
        return loss + ne_loss
