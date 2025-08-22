from deeprobust.graph.data import Dataset
import os.path as osp
import numpy as np
import networkx as nx


class CustomDataset(Dataset):
    def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
  
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!"
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def get_train_val_test(self):
        """Get training, validation, and test indices"""
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
            return idx_train, idx_val, idx_test
        else:
            # Custom implementation with proper split ratios
            rng = np.random.RandomState(self.seed)
            idx = np.arange(len(self.labels))
            
            # Get indices for each class
            idx_train = []
            idx_val = []
            idx_test = []
            
            for i in range(max(self.labels) + 1):
                # Get indices for current class
                idx_i = idx[self.labels == i]
                
                # Shuffle indices
                rng.shuffle(idx_i)
                
                # Split indices using ratios (60% train, 20% val, 20% test)
                n_samples = len(idx_i)
                train_size = int(0.6 * n_samples)
                val_size = int(0.2 * n_samples)
                
                idx_train.extend(idx_i[:train_size])
                idx_val.extend(idx_i[train_size:train_size + val_size])
                idx_test.extend(idx_i[train_size + val_size:])
            
            # Convert to numpy arrays with explicit int64 type
            idx_train = np.array(idx_train, dtype=np.int64)
            idx_val = np.array(idx_val, dtype=np.int64)
            idx_test = np.array(idx_test, dtype=np.int64)
            
            # Shuffle the indices
            rng.shuffle(idx_train)
            rng.shuffle(idx_val)
            rng.shuffle(idx_test)
            
            return idx_train, idx_val, idx_test

if __name__ == '__main__':
    dataset = CustomDataset(root="syn-cora", name="h0.00-r2", setting="gcn", seed=15)

    adj = dataset.adj  # Access adjacency matrix
    features = dataset.features  # Access node features
    labels = dataset.labels
    idx_test, idx_train, idx_val = dataset.idx_test, dataset.idx_train, dataset.idx_val
    print(adj.shape)
    print(features.shape)
    print(labels)
    print(idx_train.shape)
    print(idx_val.shape)
    print(idx_test.shape)
