import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim).cuda())
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adjacency_matrix):
        x = torch.matmul(adjacency_matrix.cuda(), x.cuda())
        x = torch.matmul(x, self.weight)
        return x

class GraphConvolutionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionNetwork, self).__init__()
        self.adjacency_matrix = None
        self.graph_conv1 = GraphConvolution(input_dim, output_dim)
        self.BN1 = nn.BatchNorm1d(output_dim)
        self.re1 = nn.ReLU(True)

    def forward(self, x, affinity_A):
        affinity_A = torch.where(affinity_A > 0, torch.ones_like(affinity_A), torch.zeros_like(affinity_A))   
        II = torch.eye(affinity_A.shape[0], affinity_A.shape[1]).cuda()
        adjacency_matrix = torch.cat((torch.cat((affinity_A, II), 1), torch.cat((II, affinity_A), 1)), 0)

        D = torch.diag(adjacency_matrix.sum(dim=1)).cuda()
        D_hat_inv_sqrt = torch.pow(D, -0.5).cuda()
        norm_A_hat = torch.mm(torch.mm(D_hat_inv_sqrt, adjacency_matrix), D_hat_inv_sqrt).cuda()
        x1 = self.graph_conv1(x, norm_A_hat).cuda()

        # x1 = self.graph_conv1(x, adjacency_matrix)
        x1 = self.BN1(x1)
        x1 = self.re1(x1)

        return x1
