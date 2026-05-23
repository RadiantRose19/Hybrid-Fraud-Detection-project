# =====================================================
# src/fusion_model.py
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFusionModel(nn.Module):

    def __init__(

        self,

        vgae_dim,
        rgcn_dim,
        hidden_dim=64,
        output_dim=2

    ):

        super().__init__()

        # Combined embedding size
        total_dim = vgae_dim + rgcn_dim

        # Fusion layers
        self.fc1 = nn.Linear(
            total_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.output = nn.Linear(
            hidden_dim,
            output_dim
        )


    def forward(

        self,

        vgae_embeddings,
        rgcn_embeddings

    ):

        # ==========================
        # Concatenate embeddings
        # ==========================

        combined = torch.cat(

            [

                vgae_embeddings,
                rgcn_embeddings

            ],

            dim=1

        )

        # ==========================
        # Dense layers
        # ==========================

        x = self.fc1(
            combined
        )

        x = F.relu(x)

        x = F.dropout(

            x,

            p=0.3,

            training=self.training

        )

        x = self.fc2(
            x
        )

        x = F.relu(x)

        out = self.output(
            x
        )

        return out