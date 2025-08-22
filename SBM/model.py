import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GraphConv as PyGGraphConv

class NodeGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Last layer for node-level prediction
        self.convs.append(GCNConv(hidden_channels, num_classes))
        
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False, embedding_layer=None):
        """
        Args:
            x: Input node features
            edge_index: Edge indices
            return_node_emb: Whether to return node embeddings
            embedding_layer: Which layer's embeddings to return (0-based index, None means last hidden layer)
        """
        embeddings = []
        
        # Process through all layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=0.0, training=self.training)
                embeddings.append(x)  # Store embeddings after activation
        
        if return_node_emb:
            if embedding_layer is not None:
                if embedding_layer >= len(embeddings):
                    raise ValueError(f"embedding_layer {embedding_layer} is too large. Max value is {len(embeddings)-1}")
                return x, embeddings[embedding_layer]
            return x, embeddings[-1]  # Default to last hidden layer
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class NodeGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(num_features, hidden_channels, heads=heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
        
        # Last layer with 1 head for final prediction
        self.convs.append(GATv2Conv(hidden_channels * heads, num_classes, heads=1))
        
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False, embedding_layer=None):
        embeddings = []
        
        # Process through all layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Not the last layer
                x = F.elu(x)
                x = F.dropout(x, p=0.0, training=self.training)
                embeddings.append(x)
        
        if return_node_emb:
            if embedding_layer is not None:
                if embedding_layer >= len(embeddings):
                    raise ValueError(f"embedding_layer {embedding_layer} is too large. Max value is {len(embeddings)-1}")
                return x, embeddings[embedding_layer]
            return x, embeddings[-1]
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class NodeGraphConv(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, aggr='add'):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        
        # First layer
        self.layers.append(PyGGraphConv(num_features, hidden_channels, aggr=aggr))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(PyGGraphConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # Last layer for node-level prediction
        self.layers.append(PyGGraphConv(hidden_channels, num_classes, aggr=aggr))
        
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False, embedding_layer=None):
        embeddings = []
        
        # Process through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  # Not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=0.0, training=self.training)
                embeddings.append(x)
        
        if return_node_emb:
            if embedding_layer is not None:
                if embedding_layer >= len(embeddings):
                    raise ValueError(f"embedding_layer {embedding_layer} is too large. Max value is {len(embeddings)-1}")
                return x, embeddings[embedding_layer]
            return x, embeddings[-1]
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()