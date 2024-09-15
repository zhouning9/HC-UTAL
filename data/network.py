import torch.nn as nn
import torch
from torch.nn.functional import normalize
# from torch_geometric.nn import GCNConv

class Network(nn.Module):
    def __init__(self, len_feature, feature_dim, class_num):
        super(Network, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        self.f_embed2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.len_feature, self.len_feature),
            nn.ReLU(),
            nn.Linear(self.len_feature, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.len_feature, self.len_feature),
            nn.ReLU(),
            nn.Linear(self.len_feature, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        x_i = x_i.permute(0, 2, 1)
        x_j = x_j.permute(0, 2, 1)

        embeddings_i = self.f_embed(x_i)
        embeddings_j = self.f_embed(x_j)

        embeddings_i = embeddings_i.permute(0, 2, 1)
        embeddings_j = embeddings_j.permute(0, 2, 1)

        # h_i [8,2048]
        h_i = torch.mean(embeddings_i, dim=1)
        h_j = torch.mean(embeddings_j, dim=1)
        h_i = normalize(h_i,dim=1)
        h_j = normalize(h_j,dim=1)

        z_i = normalize(self.instance_projector(h_i), dim=1) #[8,1024]
        z_j = normalize(self.instance_projector(h_j), dim=1) #[8,1024]

        c_i = self.cluster_projector(h_i) #[8,20]
        c_j = self.cluster_projector(h_j) #[8,20]

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        x = x.permute(0, 2, 1)
        embeddings = self.f_embed(x)
        embeddings = embeddings.permute(0, 2, 1)

        h = torch.mean(embeddings, dim=1)

        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

    def forward_instance(self, x):
        x = x.permute(0, 2, 1)
        embeddings = self.f_embed(x)
        embeddings = embeddings.permute(0, 2, 1)
        h = torch.mean(embeddings, dim=1)
        z = normalize(self.instance_projector(h), dim=1) #[8,1024]
        return z