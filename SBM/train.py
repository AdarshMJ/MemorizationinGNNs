import os
import numpy as np
import torch
import random
from model import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, x, edge_index, train_mask, y, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(x.to(device), edge_index.to(device))
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, edge_index, mask, y, device):
    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device))
        pred = out[mask].max(1)[1]
        correct = pred.eq(y[mask]).sum().item()
        total = mask.sum().item()
    return correct / total

def get_model(model_type, num_features, num_classes, hidden_dim, num_layers, gat_heads=4):
    """Create a new model instance based on specified type"""
    if model_type.lower() == 'gcn':
        return NodeGCN(num_features, num_classes, hidden_dim, num_layers)
    elif model_type.lower() == 'gat':
        return NodeGAT(num_features, num_classes, hidden_dim, num_layers, heads=gat_heads)
    elif model_type.lower() == 'graphconv':
        return NodeGraphConv(num_features, num_classes, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def train_models(args, data, shared_idx, candidate_idx, independent_idx, device, logger, output_dir=None):
    """Train model f and g on their respective node sets"""
    
    # Only create directories if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create train masks for model f and g
    train_mask_f = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_f[shared_idx + candidate_idx] = True
    
    train_mask_g = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_g[shared_idx + independent_idx] = True
    
    # Print training set information
    logger.info("\nTraining Set Information:")
    logger.info(f"Model f training nodes: {train_mask_f.sum().item()}")
    logger.info(f"- Shared nodes: {len(shared_idx)}")
    logger.info(f"- Candidate nodes: {len(candidate_idx)}")
    
    logger.info(f"\nModel g training nodes: {train_mask_g.sum().item()}")
    logger.info(f"- Shared nodes: {len(shared_idx)}")
    logger.info(f"- Independent nodes: {len(independent_idx)}")
    
    # Get number of classes
    num_classes = data.y.max().item() + 1
    
    # Lists to store models and their accuracies
    f_models = []
    g_models = []
    f_val_accs = []
    g_val_accs = []
    f_test_accs = []  # Add list for test accuracies
    g_test_accs = []  # Add list for test accuracies
    
    # Seeds for multiple training runs
    training_seeds = [42, 123, 456]
    
    logger.info("\nModel Architecture Details:")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Input Features: {data.x.size(1)}")
    logger.info(f"Hidden Dimensions: {args.hidden_dim}")
    logger.info(f"Number of Layers: {args.num_layers}")
    if args.model_type == 'gat':
        logger.info(f"Number of Attention Heads: {args.gat_heads}")
    logger.info(f"Output Classes: {num_classes}")
    logger.info(f"Training with seeds: {training_seeds}")
    
    # Train multiple models with different seeds
    for seed in training_seeds:
        set_seed(seed)
        
        logger.info(f"\nTraining with seed {seed}")
        
        # Initialize models
        model_f = get_model(args.model_type, data.x.size(1), num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        model_g = get_model(args.model_type, data.x.size(1), num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        
        opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_f_val_acc = 0
        best_g_val_acc = 0
        best_f_state = None
        best_g_state = None
        best_f_test_acc = 0
        best_g_test_acc = 0

        for epoch in range(args.epochs):
            # Train model f on shared+candidate nodes
            f_loss = train(model_f, data.x, data.edge_index, 
                         train_mask_f, data.y, opt_f, device)
            f_val_acc = test(model_f, data.x, data.edge_index, 
                           data.val_mask, data.y, device)
            current_f_test_acc = test(model_f, data.x, data.edge_index, 
                                    data.test_mask, data.y, device)
            
            # Train model g on shared+independent nodes
            g_loss = train(model_g, data.x, data.edge_index, 
                         train_mask_g, data.y, opt_g, device)
            g_val_acc = test(model_g, data.x, data.edge_index, 
                           data.val_mask, data.y, device)
            current_g_test_acc = test(model_g, data.x, data.edge_index, 
                                    data.test_mask, data.y, device)
            
            # Save best models based on validation accuracy
            if f_val_acc > best_f_val_acc:
                best_f_val_acc = f_val_acc
                best_f_test_acc = current_f_test_acc
                best_f_state = model_f.state_dict()
            
            if g_val_acc > best_g_val_acc:
                best_g_val_acc = g_val_acc
                best_g_test_acc = current_g_test_acc
                best_g_state = model_g.state_dict()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f'Seed {seed}, Epoch {epoch+1}/{args.epochs}:')
                logger.info(f'Model f - Loss: {f_loss:.4f}, Val Acc: {f_val_acc:.4f}, Test Acc: {current_f_test_acc:.4f}')
                logger.info(f'Model g - Loss: {g_loss:.4f}, Val Acc: {g_val_acc:.4f}, Test Acc: {current_g_test_acc:.4f}')
        
        # Load best states
        model_f.load_state_dict(best_f_state)
        model_g.load_state_dict(best_g_state)
        
        # Store models and accuracies
        f_models.append(model_f.state_dict())
        g_models.append(model_g.state_dict())
        f_val_accs.append(best_f_val_acc)
        g_val_accs.append(best_g_val_acc)
        f_test_accs.append(best_f_test_acc)  # Store test accuracy
        g_test_accs.append(best_g_test_acc)  # Store test accuracy
        
        logger.info(f"\nSeed {seed} Results:")
        logger.info(f"Best Model f - Val Acc: {best_f_val_acc:.4f}, Test Acc: {best_f_test_acc:.4f}")
        logger.info(f"Best Model g - Val Acc: {best_g_val_acc:.4f}, Test Acc: {best_g_test_acc:.4f}")
    
    # Select models with best validation accuracy
    f_best_idx = np.argmax(f_val_accs)
    g_best_idx = np.argmax(g_val_accs)
    
    # Save best models
    save_dir = output_dir
    
    # Create model save directory if it doesn't exist
    #os.makedirs(save_dir, exist_ok=True)
    
    #torch.save(f_models[f_best_idx], os.path.join(save_dir, 'f_model.pt'))
    #torch.save(g_models[g_best_idx], os.path.join(save_dir, 'g_model.pt'))
    
    logger.info("\nSaved models with best validation accuracy:")
    logger.info(f"Model f - Val Acc: {f_val_accs[f_best_idx]:.4f}")
    logger.info(f"Model g - Val Acc: {g_val_accs[g_best_idx]:.4f}")
    
    # After training is complete, calculate and log average test accuracies
    avg_f_test_acc = np.mean(f_test_accs)
    avg_g_test_acc = np.mean(g_test_accs)
    std_f_test_acc = np.std(f_test_accs)
    std_g_test_acc = np.std(g_test_accs)
    
    if logger:
        logger.info("\nFinal Test Accuracies (averaged over {} seeds):".format(len(training_seeds)))
        logger.info(f"Model f - Test Acc: {avg_f_test_acc:.4f} ± {std_f_test_acc:.4f}")
        logger.info(f"Model g - Test Acc: {avg_g_test_acc:.4f} ± {std_g_test_acc:.4f}")
    
    return model_f, model_g, f_val_accs[f_best_idx], g_val_accs[g_best_idx]