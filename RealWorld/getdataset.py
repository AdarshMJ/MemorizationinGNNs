import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork, WebKB, HeterophilousGraphDataset
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents
from torch_geometric.transforms import Compose



def load_dataset(args):
    transforms = Compose([
        LargestConnectedComponents(),
        RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    ])
    
    # Handle standard PyTorch Geometric datasets
    if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['computers', 'photo']:
        dataset = Amazon(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() == 'actor':
        dataset = Actor(root='data/Actor', transform=transforms)
    elif args.dataset.lower() in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['cornell', 'wisconsin','texas']:
        dataset = WebKB(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['roman-empire', 'amazon-ratings']:
        dataset = HeterophilousGraphDataset(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")    
    dataset._data_list = None
    return dataset


def get_node_splits(data, train_mask, swap_candidate_independent=False):
    """
    Create node splits without shuffling to preserve natural ordering.
    
    Args:
        data: PyG data object
        train_mask: Mask for train nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train indices in their original order
    train_indices = torch.where(train_mask)[0]
    
    # Calculate sizes
    num_nodes = len(train_indices)
    shared_size = int(0.50 * num_nodes)
    remaining = num_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_indices[:shared_size].tolist()
    original_candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    original_independent_idx = train_indices[shared_size + split_size:shared_size + split_size * 2].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, original_independent_idx, original_candidate_idx
    else:
        return shared_idx, original_candidate_idx, original_independent_idx

def verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, logger):
    """Verify there is no direct overlap between candidate and independent sets"""
    # Convert to sets for easy comparison
    candidate_set = set(candidate_idx)
    independent_set = set(independent_idx)
    
    # Check: No overlap between candidate and independent sets
    overlap = candidate_set.intersection(independent_set)
    if overlap:
        raise ValueError(f"Data leakage detected! Found {len(overlap)} nodes in both candidate and independent sets")
    
    logger.info("\nData Leakage Check:")
    logger.info(f"✓ No overlap between candidate and independent sets")