# -*- coding: utf-8 -*-
"""GINE-based Siamese RankNet and helpers.

Uses nn.Dropout modules (instead of functional dropout) so Monte Carlo dropout
can keep BatchNorm in eval mode while sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class GINEEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, dropout=0.3):
        super().__init__()
        self.node_lin = nn.Linear(node_in, hidden_dim)
        self.edge_lin = nn.Linear(edge_in, hidden_dim)

        def make_gine():
            return GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)))

        self.conv1 = make_gine()
        self.conv2 = make_gine()
        self.conv3 = make_gine()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_lin(x)
        if edge_attr.numel() > 0:
            edge_attr = self.edge_lin(edge_attr)
        else:
            edge_attr = torch.zeros((edge_index.size(1), x.size(1)), device=x.device)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        return global_add_pool(x, batch)


class SiameseRankNet(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, dropout=0.3):
        super().__init__()
        self.encoder = GINEEncoder(node_in, edge_in, hidden_dim, dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.head_drop = nn.Dropout(dropout)

    def forward_one(self, data):
        x = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        x = self.head_drop(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, data1, data2):
        return self.forward_one(data1), self.forward_one(data2)


class WeightedRankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, s1, s2, target, weights):
        loss = self.bce(s1 - s2, target)
        return (loss * weights).mean()


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-enable only Dropout layers.

    This keeps BatchNorm using running statistics while dropout stays
    stochastic, which is the correct setting for Monte Carlo dropout.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
