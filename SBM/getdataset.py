from dataset import CustomDataset
import torch
from nodeli import li_node
from scipy import sparse
import numpy as np
import os
import networkx as nx
import logging

def load_and_process_dataset(args, dataset_name, logger):
    """Load synthetic Cora dataset and convert to PyG format"""
    # Construct full path to dataset
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "syn-cora")
    dataset = CustomDataset(root=root_dir, name=dataset_name, setting="gcn")
    
    # Convert to PyG format
    edge_index = torch.from_numpy(np.vstack(dataset.adj.nonzero())).long()
    
    # Convert sparse features to dense numpy array
    if sparse.issparse(dataset.features):
        x = torch.from_numpy(dataset.features.todense()).float()
    else:
        x = torch.from_numpy(dataset.features).float()
    
    y = torch.from_numpy(dataset.labels).long()
    
    # Create train/val/test masks
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    
    train_mask[dataset.idx_train] = True
    val_mask[dataset.idx_val] = True
    test_mask[dataset.idx_test] = True
    
    # Convert to networkx for informativeness calculation
    G = nx.Graph()
    G.add_nodes_from(range(len(y)))
    G.add_edges_from(edge_index.t().numpy())
    
    # Calculate label informativeness using existing function
    informativeness = li_node(G, dataset.labels)
    
    # Calculate homophily (edge homophily)
    edges = edge_index.t().numpy()
    same_label = dataset.labels[edges[:, 0]] == dataset.labels[edges[:, 1]]
    homophily = same_label.mean()
    
    # Create a data object
    data = type('Data', (), {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': len(y),
        'informativeness': informativeness,
        'homophily': homophily
    })()
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {len(edges)}")
    logger.info(f"Number of features: {x.shape[1]}")
    logger.info(f"Number of classes: {len(torch.unique(y))}")
    logger.info(f"Homophily: {homophily:.4f}")
    logger.info(f"Label Informativeness: {informativeness:.4f}")
    
    return data
