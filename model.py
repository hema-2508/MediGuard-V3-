"""
model.py
--------
GraphSAGE encoder (2 layers, hidden=128, dropout=0.3) that produces a
node embedding per drug, plus a pair classifier head that consumes
[A, B, |A-B|, A*B] and predicts a single interaction logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from config import Config


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE node encoder.

    Layer 1: input_dim -> hidden_dim, ReLU, BatchNorm, Dropout
    Layer 2: hidden_dim -> hidden_dim (final node embedding)
    """

    def __init__(
        self,
        in_channels: int = Config.NODE_FEATURE_DIM,
        hidden_channels: int = Config.SAGE_HIDDEN_DIM,
        num_layers: int = Config.SAGE_NUM_LAYERS,
        dropout: float = Config.SAGE_DROPOUT,
        aggr: str = Config.SAGE_AGGR,
    ):
        super().__init__()
        assert num_layers >= 2, "This encoder is designed for >= 2 SAGEConv layers."

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # first layer: raw features -> hidden
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # middle layers (if num_layers > 2): hidden -> hidden
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # final layer: hidden -> hidden (embedding output, no activation after)
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_conv_layers = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            is_last_layer = i == num_conv_layers - 1
            if not is_last_layer:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # (num_nodes, hidden_channels) node embeddings


class PairClassifier(nn.Module):
    """MLP head operating on the concatenated pair representation
    [A, B, |A-B|, A*B], each of dimension `embed_dim`, giving an input
    of size 4 * embed_dim. Outputs a single raw logit (use
    BCEWithLogitsLoss / sigmoid downstream)."""

    def __init__(
        self,
        embed_dim: int = Config.SAGE_HIDDEN_DIM,
        hidden_dim: int = Config.PAIR_HIDDEN_DIM,
        dropout: float = Config.PAIR_DROPOUT,
    ):
        super().__init__()
        in_dim = embed_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def build_pair_features(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """[A, B, |A-B|, A*B] concatenation."""
        diff = torch.abs(z_a - z_b)
        prod = z_a * z_b
        return torch.cat([z_a, z_b, diff, prod], dim=-1)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        pair_repr = self.build_pair_features(z_a, z_b)
        logit = self.net(pair_repr).squeeze(-1)
        return logit


class DDIModel(nn.Module):
    """End-to-end GraphSAGE + pair-classifier DDI predictor.

    forward(x, edge_index, idx_a, idx_b) computes node embeddings for
    the ENTIRE graph once, then gathers the embeddings for the
    requested pair indices and classifies each pair.
    """

    def __init__(
        self,
        in_channels: int = Config.NODE_FEATURE_DIM,
        hidden_channels: int = Config.SAGE_HIDDEN_DIM,
        num_layers: int = Config.SAGE_NUM_LAYERS,
        sage_dropout: float = Config.SAGE_DROPOUT,
        pair_hidden_dim: int = Config.PAIR_HIDDEN_DIM,
        pair_dropout: float = Config.PAIR_DROPOUT,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=sage_dropout,
        )
        self.classifier = PairClassifier(
            embed_dim=hidden_channels,
            hidden_dim=pair_hidden_dim,
            dropout=pair_dropout,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        node_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """If `node_embeddings` is provided, skip re-encoding the graph
        (useful for evaluation where embeddings are computed once and
        reused across many pair mini-batches)."""
        z = node_embeddings if node_embeddings is not None else self.encode(x, edge_index)
        z_a = z[idx_a]
        z_b = z[idx_b]
        return self.classifier(z_a, z_b)