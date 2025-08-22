import numpy as np
import networkx as nx
from torch_geometric.utils import subgraph
import torch
from torch_geometric.utils import degree, to_networkx
import random


def get_graph_and_labels_from_pyg_dataset(dataset):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(dataset.x)))
    # Move the edge_index tensor to CPU before converting to numpy
    graph.add_edges_from(dataset.edge_index.cpu().T.numpy())

    labels = dataset.y.detach().cpu().numpy()
    return graph, labels


def make_labels_hashable(labels):
    """Convert labels to a hashable type."""
    if isinstance(labels, np.ndarray):
        return labels.astype(str)
    elif isinstance(labels, torch.Tensor):
        return labels.cpu().numpy().astype(str)
    elif isinstance(labels, dict):
        if isinstance(next(iter(labels.values())), (dict, list, np.ndarray)):
            return {node: str(label) for node, label in labels.items()}
    return labels


def convert_labels_to_consecutive_integers(labels):
    """Convert labels to consecutive integers starting from 0."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    if isinstance(labels, np.ndarray):
        unique_labels = np.unique(labels)
        labels_map = {label: i for i, label in enumerate(unique_labels)}
        return np.array([labels_map[label] for label in labels])
    else:
        labels = make_labels_hashable(labels)
        unique_labels = set(labels.values())
        labels_map = {label: i for i, label in enumerate(unique_labels)}
        return {node: labels_map[label] for node, label in labels.items()}


def make_labels_consecutive(labels):
    """Convert labels to consecutive integers starting from 0"""
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[l] for l in labels])

def li_node(G, node_labels, eps=1e-8):
    """
    Calculate node label informativeness for a graph
    
    Args:
        G: NetworkX graph
        node_labels: Node labels array
        eps: Small constant to prevent division by zero
    """
    # Convert labels to consecutive integers
    node_labels = make_labels_consecutive(node_labels)
    num_classes = len(np.unique(node_labels))
    
    if num_classes <= 1:
        return 0.0  # If only one class, informativeness is 0
        
    # Calculate label frequencies
    label_counts = np.bincount(node_labels)
    label_freqs = label_counts / len(node_labels)
    
    # Calculate entropy of label distribution
    entropy = -np.sum(label_freqs * np.log(label_freqs + eps))
    
    # Calculate conditional entropy
    cond_entropy = 0
    total_edges = G.number_of_edges()
    
    if total_edges == 0:
        return 0.0  # If no edges, informativeness is 0
    
    # Create edge label pairs
    edge_label_pairs = []
    for u, v in G.edges():
        if u < len(node_labels) and v < len(node_labels):  # Check if indices are valid
            edge_label_pairs.append((node_labels[u], node_labels[v]))
    
    if not edge_label_pairs:
        return 0.0
        
    edge_label_pairs = np.array(edge_label_pairs)
    
    # Calculate joint probabilities
    joint_counts = np.zeros((num_classes, num_classes))
    for i, j in edge_label_pairs:
        joint_counts[i, j] += 1
        joint_counts[j, i] += 1  # Since graph is undirected
    
    joint_probs = joint_counts / (2 * total_edges + eps)
    
    # Calculate marginal probabilities
    marginal_probs = joint_probs.sum(axis=1)
    
    # Calculate conditional entropy
    for i in range(num_classes):
        for j in range(num_classes):
            if joint_probs[i, j] > 0 and marginal_probs[i] > 0:
                cond_prob = joint_probs[i, j] / (marginal_probs[i] + eps)
                cond_entropy -= joint_probs[i, j] * np.log(cond_prob + eps)
    
    # Calculate normalized informativeness
    if entropy > eps:
        informativeness = 1 - (cond_entropy / entropy)
    else:
        informativeness = 0.0
        
    return float(informativeness)

def get_local_subgraph(node_idx, edge_index, num_hops, num_nodes):
    """Get k-hop subgraph around a node"""
    row, col = edge_index
    node_mask = row.new_zeros(num_nodes, dtype=torch.bool)
    edge_mask = row.new_zeros(row.size(0), dtype=torch.bool)
    
    nodes = {node_idx}
    for _ in range(num_hops):
        edge_mask = torch.logical_or(
            edge_mask,
            torch.logical_or(
                torch.isin(row, torch.tensor(list(nodes))),
                torch.isin(col, torch.tensor(list(nodes)))
            )
        )
        nodes.update(row[edge_mask].tolist() + col[edge_mask].tolist())
    
    node_mask[list(nodes)] = True
    edge_mask = edge_mask.nonzero().view(-1)
    
    return node_mask, edge_mask

def h_edge(graph, labels):
    """Compute edge homophily."""
    edges_with_same_label = 0
    for u, v in graph.edges:
        if labels[u] == labels[v]:
            edges_with_same_label += 1

    h_edge = edges_with_same_label / len(graph.edges)

    return h_edge


def h_edge_subgraph(edge_index, labels):
    """Compute edge homophily."""
    edges_with_same_label = (labels[edge_index[0]] == labels[edge_index[1]]).sum().item()
    h_edge = edges_with_same_label / edge_index.shape[1]
    return h_edge


def h_adj(graph, labels):
    """Compute adjusted homophily."""
    labels = convert_labels_to_consecutive_integers(labels)

    num_classes = len(np.unique(labels))

    degree_sums = np.zeros((num_classes,))
    for u in graph.nodes:
        label = labels[u]
        degree_sums[label] += graph.degree(u)

    adjust = (degree_sums ** 2 / (len(graph.edges) * 2) ** 2).sum()

    h_adj = (h_edge(graph, labels) - adjust) / (1 - adjust)

    return h_adj


def h_adj_subgraph(data):
    """Compute average adjusted homophily across all splits for the subgraph induced by train and validation nodes."""
    num_splits = data.train_mask.shape[1]
    total_h_adj = 0
    total_nodes = 0
    total_edges = 0
    valid_splits = 0

    for split in range(num_splits):
        # Create mask for train and validation nodes for this split
        train_val_mask = torch.logical_or(data.train_mask[:, split], data.val_mask[:, split])
        
        # Get the subgraph induced by train and validation nodes
        subset = train_val_mask.nonzero().reshape(-1)
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
        sub_y = data.y[subset]
        
        num_nodes = subset.shape[0]
        num_edges = sub_edge_index.shape[1]
        
        # Convert labels to consecutive integers
        labels = convert_labels_to_consecutive_integers(sub_y.numpy())
        
        # Compute edge homophily
        h_edge_value = h_edge_subgraph(sub_edge_index, labels)
        
        # Compute adjusted homophily
        num_classes = len(np.unique(labels))
        degree_sums = np.zeros((num_classes,))
        
        for i in range(num_nodes):
            label = labels[i]
            degree_sums[label] += degree(sub_edge_index[0], num_nodes=num_nodes)[i].item()
        
        adjust = (degree_sums ** 2 / (num_edges * 2) ** 2).sum()
        
        # Skip this split if adjust is 1 (would lead to division by zero)
        if adjust == 1:
            continue
        
        h_adj = (h_edge_value - adjust) / (1 - adjust)
        
        # Skip this split if h_adj is NaN
        if np.isnan(h_adj):
            continue
        
        total_h_adj += h_adj
        total_nodes += num_nodes
        total_edges += num_edges
        valid_splits += 1
    
    if valid_splits == 0:
        return None, 0, 0

    avg_h_adj = total_h_adj / valid_splits
    avg_nodes = total_nodes / valid_splits
    avg_edges = total_edges / valid_splits
    
    print(f"Average number of nodes in the subgraph: {avg_nodes:.2f}")
    print(f"Average number of edges in the subgraph: {avg_edges:.2f}")
    print(f"Number of valid splits: {valid_splits}")
    
    return avg_h_adj, avg_nodes, avg_edges

def li_edge(graph, labels, eps=1e-8):
    """Compute edge label informativeness."""
    labels = convert_labels_to_consecutive_integers(labels)

    num_classes = len(np.unique(labels))

    class_probs = np.array([0 for _ in range(num_classes)], dtype=float)
    class_degree_weighted_probs = np.array([0 for _ in range(num_classes)], dtype=float)
    for u in graph.nodes:
        label = labels[u]
        class_probs[label] += 1
        class_degree_weighted_probs[label] += graph.degree(u)

    class_probs /= class_probs.sum()
    class_degree_weighted_probs /= class_degree_weighted_probs.sum()

    edge_probs = np.zeros((num_classes, num_classes))
    for u, v in graph.edges:
        label_u = labels[u]
        label_v = labels[v]
        edge_probs[label_u, label_v] += 1
        edge_probs[label_v, label_u] += 1

    edge_probs /= edge_probs.sum()

    edge_probs += eps

    numerator = (edge_probs * np.log(edge_probs)).sum()
    denominator = (class_degree_weighted_probs * np.log(class_degree_weighted_probs)).sum()
    li_edge = 2 - numerator / denominator

    return li_edge


def community_h_adj_avg(G, community, data):
    """
    Calculate average adjusted homophily for a community using train and validation nodes across all splits.
    
    :param G: The NetworkX graph
    :param community: A set of nodes in the community
    :param data: The original PyTorch Geometric data object
    :return: Tuple containing:
        - Average adjusted homophily for the community subgraph (train+val nodes)
        - Average number of train/val nodes
        - Average number of edges in the subgraph
        - Adjusted homophily for the entire community
    """
    num_splits = 1
    total_h_adj = 0
    total_nodes = 0
    total_edges = 0
    valid_splits = 0

    # Calculate homophily for the entire community
    community_subgraph = G.subgraph(community)
    community_labels = data.y[list(community)].numpy()
    community_h_adj = calculate_h_adj(community_subgraph, community_labels)

    for split in range(num_splits):
        # Create mask for train and validation nodes for this split
        train_val_mask = torch.logical_or(data.train_mask[:, split], data.val_mask[:, split])
        
        # Filter for only train and validation nodes within the community
        community_train_val = set(community) & set(train_val_mask.nonzero().flatten().tolist())
        
        if len(community_train_val) <= 1:
            continue  # Not enough train/val nodes to calculate homophily for this split
        
        # Get subgraph of train/val nodes in the community
        subgraph = G.subgraph(community_train_val)
        
        # Get labels for train and validation nodes
        labels = data.y[list(community_train_val)].numpy()
        
        # Convert labels to consecutive integers
        unique_labels = np.unique(labels)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        int_labels = np.array([label_to_int[label] for label in labels])
        
        num_classes = len(unique_labels)
        num_nodes = len(community_train_val)
        num_edges = subgraph.number_of_edges()
        
        if num_edges == 0:
            continue  # No edges, can't calculate homophily for this split
        
        degree_sums = np.zeros(num_classes)
        edge_sums = np.zeros((num_classes, num_classes))
        
        # Count edges and degrees
        for u, v in subgraph.edges():
            lu = int_labels[list(community_train_val).index(u)]
            lv = int_labels[list(community_train_val).index(v)]
            degree_sums[lu] += 1
            degree_sums[lv] += 1
            edge_sums[lu, lv] += 1
            edge_sums[lv, lu] += 1
        
        # Calculate edge homophily
        h_edge = np.trace(edge_sums) / np.sum(edge_sums)
        
        # Calculate adjustment factor
        total_edges = num_edges
        adjust = np.sum((degree_sums / (2 * total_edges)) ** 2)
        
        # Calculate adjusted homophily
        if adjust == 1:
            continue  # Can't calculate adjusted homophily for this split
        
        h_adj = (h_edge - adjust) / (1 - adjust)
        
        if np.isnan(h_adj):
            continue  # Skip this split if h_adj is NaN
        
        total_h_adj += h_adj
        total_nodes += num_nodes
        total_edges += num_edges
        valid_splits += 1
    
    if valid_splits == 0:
        return None, 0, 0, community_h_adj

    avg_h_adj = total_h_adj / valid_splits
    avg_nodes = total_nodes / valid_splits
    avg_edges = total_edges / valid_splits
    
    return avg_h_adj, avg_nodes, avg_edges, community_h_adj

def calculate_h_adj(graph, labels):
    """Helper function to calculate adjusted homophily for a graph and its labels."""
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels])
    
    num_classes = len(unique_labels)
    num_edges = graph.number_of_edges()
    
    if num_edges == 0:
        return None  # Can't calculate homophily for a graph with no edges
    
    degree_sums = np.zeros(num_classes)
    edge_sums = np.zeros((num_classes, num_classes))
    
    # Count edges and degrees
    for u, v in graph.edges():
        lu = int_labels[list(graph.nodes()).index(u)]
        lv = int_labels[list(graph.nodes()).index(v)]
        degree_sums[lu] += 1
        degree_sums[lv] += 1
        edge_sums[lu, lv] += 1
        edge_sums[lv, lu] += 1
    
    # Calculate edge homophily
    h_edge = np.trace(edge_sums) / np.sum(edge_sums)
    
    # Calculate adjustment factor
    adjust = np.sum((degree_sums / (2 * num_edges)) ** 2)
    
    # Calculate adjusted homophily
    if adjust == 1:
        return None  # Can't calculate adjusted homophily
    
    h_adj = (h_edge - adjust) / (1 - adjust)
    
    return h_adj






