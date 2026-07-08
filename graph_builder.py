"""
graph_builder.py
----------------
Builds the drug-drug interaction graph consumed by GraphSAGE.

Design decision (documented so it's not a mystery later): the
message-passing edge_index is built ONLY from label==1 pairs in the
*training* split. This is the standard transductive link-prediction
setup: the GNN is allowed to see the training interaction network's
structure, but never structure derived from validation/test/external
labels, which would leak the answer into the node embeddings.

Drugs that only appear in test/external data (and never in a training
positive pair) simply become isolated nodes in the graph. SAGEConv
with root_weight=True still produces a valid embedding for isolated
nodes (via the self/root transformation), so cold-start drugs are
handled gracefully rather than crashing.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from config import Config
from dataset import PairSplit
from utils import get_logger

logger = get_logger("graph_builder")


def build_edge_index(num_nodes: int, train_pairs: PairSplit) -> torch.Tensor:
    """Construct the message-passing edge_index from training pairs
    whose label == 1 (i.e. known drug-drug interactions)."""
    pos_mask = train_pairs.labels == 1
    src = train_pairs.idx_a[pos_mask]
    dst = train_pairs.idx_b[pos_mask]

    if len(src) == 0:
        logger.warning(
            "No positive (label==1) training pairs found; graph will have "
            "no edges besides self-loops. Check your training data."
        )
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    if Config.MAKE_EDGES_BIDIRECTIONAL and edge_index.numel() > 0:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    edge_index, _ = remove_self_loops(edge_index)

    if Config.ADD_SELF_LOOPS:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    return edge_index


def build_graph(feature_matrix: np.ndarray, train_pairs: PairSplit) -> Data:
    """Return a torch_geometric.data.Data object:
        x          : (num_nodes, NODE_FEATURE_DIM) float tensor of drug features
        edge_index : (2, num_edges) long tensor, DDI network from training positives
    """
    num_nodes = feature_matrix.shape[0]
    x = torch.tensor(feature_matrix, dtype=torch.float32)
    edge_index = build_edge_index(num_nodes, train_pairs)

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    logger.info(
        f"Built DDI graph: {data.num_nodes} nodes, {data.edge_index.size(1)} directed edges "
        f"(includes self-loops={Config.ADD_SELF_LOOPS}, bidirectional={Config.MAKE_EDGES_BIDIRECTIONAL})."
    )
    return data


def extend_graph_with_new_nodes(
    graph: Data, new_feature_rows: np.ndarray
) -> Data:
    """Return a NEW Data object with `new_feature_rows` appended as
    isolated nodes (no incident edges). Used by predict.py when a
    SMILES not seen during training/graph-building needs an embedding.
    Original graph is left untouched.
    """
    new_x = torch.tensor(new_feature_rows, dtype=torch.float32)
    x = torch.cat([graph.x, new_x], dim=0)
    data = Data(x=x, edge_index=graph.edge_index.clone(), num_nodes=x.size(0))
    return data