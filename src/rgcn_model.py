# src/rgcn_model.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv


class RGCN(

    torch.nn.Module

):

    def __init__(

        self,
        in_channels

    ):

        super().__init__()

        self.conv1=RGCNConv(
            in_channels,
            32,
            1
        )

        self.conv2=RGCNConv(
            32,
            2,
            1
        )

    def forward(

        self,
        x,
        edge_index,
        edge_type

    ):

        x=self.conv1(
            x,
            edge_index,
            edge_type
        )

        x=F.relu(x)

        x=self.conv2(
            x,
            edge_index,
            edge_type
        )

        return x