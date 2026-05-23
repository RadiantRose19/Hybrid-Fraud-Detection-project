# src/vgae_model.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import VGAE
from torch_geometric.nn import GCNConv


class Encoder(torch.nn.Module):

    def __init__(

        self,
        in_channels,
        out_channels

    ):

        super().__init__()

        self.conv1=GCNConv(
            in_channels,
            32
        )

        self.mu=GCNConv(
            32,
            out_channels
        )

        self.logstd=GCNConv(
            32,
            out_channels
        )

    def forward(

        self,
        x,
        edge_index

    ):

        x=self.conv1(
            x,
            edge_index
        )

        x=F.relu(x)

        return (

            self.mu(
                x,
                edge_index
            ),

            self.logstd(
                x,
                edge_index
            )

        )