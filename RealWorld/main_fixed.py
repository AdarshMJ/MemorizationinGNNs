import os
import sys
import csv
import time
import pickle
import logging
import numpy as np
import argparse
import torch
from getdataset import load_dataset, get_node_splits, verify_no_data_leakage
from train import *
from model import *
from datetime import datetime
from memorization import calculate_node_memorization_score, plot_node_memorization_analysis, plot_separate_node_memorization_plots
from scipy import stats
from nodeli import get_graph_and_labels_from_pyg_dataset, li_node, h_adj
from knn_label_disagreement import run_knn_label_disagreement_analysis




def setup_logging(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name with the structure ModelType_Datasetname_NumLayers_timestamp
    dir_name = f"{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}"
    
    # Create base results directory if it doesn't exist
    base_dir = 'results'
    os.makedirs(base_dir, exist_ok=True)
    
    # Create full directory path
    log_dir = os.path.join(base_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup main logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file = os.path.join(log_dir, f'{args.model_type}_{args.dataset}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger, log_dir, timestamp


def perform_memorization_statistical_tests(node_scores, logger):
    """
    Perform statistical tests to check if memorization scores are statistically significant.
    
    Args:
        node_scores: Dictionary of memorization scores by node type
        logger: Logger to output results
    """
    logger.info("\n===== Statistical Significance Tests =====")
    
    # Check if all required node types exist
    required_types = ['candidate', 'shared', 'independent', 'extra']
    for node_type in required_types:
        if node_type not in node_scores:
            logger.info(f"Skipping some statistical tests: '{node_type}' nodes not found")
    
    # 1. Candidate vs other node types
    if 'candidate' in node_scores:
        candidate_scores = node_scores['candidate']['mem_scores']
        
        # Test against each other node type
        for other_type in ['shared', 'independent', 'extra']:
            if other_type not in node_scores:
                continue
                
            other_scores = node_scores[other_type]['mem_scores']
            
            # Run t-test
            t_stat, p_val = stats.ttest_ind(candidate_scores, other_scores, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(candidate_scores) - np.mean(other_scores)
            pooled_std = np.sqrt((np.std(candidate_scores)**2 + np.std(other_scores)**2) / 2)
            effect_size = abs(mean_diff) / pooled_std
            
            # Interpret effect size
            if effect_size < 0.2:
                effect_size_interp = "negligible"
            elif effect_size < 0.5:
                effect_size_interp = "small"
            elif effect_size < 0.8:
                effect_size_interp = "medium"
            else:
                effect_size_interp = "large"
                
            # Interpret p-value
            significant = p_val < 0.01
            
            # Log results
            logger.info(f"\nCandidate vs {other_type} nodes:")
            logger.info(f"  T-statistic: {t_stat:.4f}")
            logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
            logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
            logger.info(f"  Mean difference: {mean_diff:.4f}")
    
    # 2. Shared vs Independent nodes
    if 'shared' in node_scores and 'independent' in node_scores:
        shared_scores = node_scores['shared']['mem_scores']
        independent_scores = node_scores['independent']['mem_scores']
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(shared_scores, independent_scores, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(shared_scores) - np.mean(independent_scores)
        pooled_std = np.sqrt((np.std(shared_scores)**2 + np.std(independent_scores)**2) / 2)
        effect_size = abs(mean_diff) / pooled_std
        
        # Interpret effect size
        if effect_size < 0.2:
            effect_size_interp = "negligible"
        elif effect_size < 0.5:
            effect_size_interp = "small"
        elif effect_size < 0.8:
            effect_size_interp = "medium"
        else:
            effect_size_interp = "large"
            
        # Interpret p-value
        significant = p_val < 0.01
        
        # Log results
        logger.info(f"\nShared vs Independent nodes:")
        logger.info(f"  T-statistic: {t_stat:.4f}")
        logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
        logger.info(f"  Mean difference: {mean_diff:.4f}")
    
    # 3. Extra vs Independent nodes
    if 'extra' in node_scores and 'independent' in node_scores:
        extra_scores = node_scores['extra']['mem_scores']
        independent_scores = node_scores['independent']['mem_scores']
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(extra_scores, independent_scores, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(extra_scores) - np.mean(independent_scores)
        pooled_std = np.sqrt((np.std(extra_scores)**2 + np.std(independent_scores)**2) / 2)
        effect_size = abs(mean_diff) / pooled_std
        
        # Interpret effect size
        if effect_size < 0.2:
            effect_size_interp = "negligible"
        elif effect_size < 0.5:
            effect_size_interp = "small"
        elif effect_size < 0.8:
            effect_size_interp = "medium"
        else:
            effect_size_interp = "large"
            
        # Interpret p-value
        significant = p_val < 0.01
        
        # Log results
        logger.info(f"\nExtra vs Independent nodes:")
        logger.info(f"  T-statistic: {t_stat:.4f}")
        logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
        logger.info(f"  Mean difference: {mean_diff:.4f}")

def calculate_graph_metrics(data):
    """Calculate adjusted homophily and node label informativeness for the graph"""
    # Get graph and labels in correct format
    graph, labels = get_graph_and_labels_from_pyg_dataset(data)
    
    # Calculate adjusted homophily
    adj_homophily = h_adj(graph, labels)
    
    # Calculate node label informativeness
    nli = li_node(graph, labels)
    
    return adj_homophily, nli

def log_results_to_csv(args, adj_homophily, nli, test_acc_mean, test_acc_std):
    """Log results to a CSV file"""
    csv_file = 'experiment_results.csv'
    file_exists = os.path.exists(csv_file)
    
    # Prepare the results row
    results = {
        'Dataset': args.dataset,
        'Model': args.model_type,
        'Test_Accuracy_Mean': f"{test_acc_mean:.4f}",
        'Test_Accuracy_Std': f"{test_acc_std:.4f}",
        'Learning_Rate': args.lr,
        'Weight_Decay': args.weight_decay,
        'Adjusted_Homophily': f"{adj_homophily:.4f}",
        'Node_Label_Informativeness': f"{nli:.4f}",
        'Num_Layers': args.num_layers
    }
    
    # Write to CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
    return csv_file

def save_memorization_scores(node_scores, log_dir, logger=None):
    """
    Save memorization scores to disk for later use.
    
    Args:
        node_scores: Dictionary containing node memorization scores
        log_dir: Directory to save the scores
        logger: Logger for status updates
    """
    # Save the entire node_scores dictionary as a pickle file
    pickle_path = os.path.join(log_dir, 'node_scores.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(node_scores, f)
    
    # Also save individual node types as CSV files for easier inspection
    for node_type, data in node_scores.items():
        if 'raw_data' in data:
            csv_path = os.path.join(log_dir, f'{node_type}_scores.csv')
            data['raw_data'].to_csv(csv_path, index=False)
    
    if logger:
        logger.info(f"Memorization scores saved to {log_dir}")
        logger.info(f"- Complete data: {os.path.basename(pickle_path)}")
        for node_type in node_scores:
            if 'raw_data' in node_scores[node_type]:
                logger.info(f"- {node_type} scores: {node_type}_scores.csv")

def main():
    parser = argparse.ArgumentParser()
    # Combine all dataset choices
    all_datasets = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Actor', 
                   'Chameleon', 'Squirrel', 'Cornell', 'Wisconsin', 'Texas',
                   'Roman-empire', 'Amazon-ratings']
                   
    parser.add_argument('--dataset', type=str, required=True,
                       choices=all_datasets,
                       help='Dataset to use for analysis')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv' , 'graphsage'],
                       help='Type of GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='Number of attention heads for GAT')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--swap_nodes', action='store_true', 
                       help='Swap candidate and independent nodes')
    parser.add_argument('--num_passes', type=int, default=1,
                       help='Number of forward passes to average for confidence scores')
  
    args = parser.parse_args()
    
    # Setup
    logger, log_dir, timestamp = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define training seeds here so they can be used throughout
    training_seeds = [42, 123, 456]
    
    # Load dataset
    dataset = load_dataset(args)
    data = dataset[0].to(device)
    
    # Calculate graph metrics
    adj_homophily, nli = calculate_graph_metrics(data)
    logger.info("\nGraph Metrics:")
    logger.info(f"Adjusted Homophily: {adj_homophily:.4f}")
    logger.info(f"Node Label Informativeness: {nli:.4f}")
    
    # Log dataset information
    logger.info(f"\nDataset Information:")
    logger.info(f"Dataset Name: {args.dataset}")
    logger.info(f"Number of Nodes: {data.num_nodes}")
    logger.info(f"Number of Edges: {data.edge_index.size(1)}")
    logger.info(f"Number of Features: {data.num_features}")
    logger.info(f"Number of Classes: {dataset.num_classes}")
    
    # Get node splits
    shared_idx, candidate_idx, independent_idx = get_node_splits(
        data, data.train_mask, swap_candidate_independent=args.swap_nodes
    )
    
    # Get extra indices from test set
    test_indices = torch.where(data.test_mask)[0]
    extra_size = len(candidate_idx)
    extra_indices = test_indices[:extra_size].tolist()  # Take first extra_size test indices

    logger.info("\nPartition Statistics:")
    if args.swap_nodes:
        logger.info("Note: Candidate and Independent nodes have been swapped!")
        logger.info("Original independent nodes are now being used as candidate nodes")
        logger.info("Original candidate nodes are now being used as independent nodes")
    logger.info(f"Total train nodes: {data.train_mask.sum().item()}")
    logger.info(f"Shared: {len(shared_idx)} nodes")
    logger.info(f"Candidate: {len(candidate_idx)} nodes")
    logger.info(f"Independent: {len(independent_idx)} nodes")
    logger.info(f"Extra test nodes: {len(extra_indices)} nodes")
    logger.info(f"Val set: {data.val_mask.sum().item()} nodes")
    logger.info(f"Test set: {data.test_mask.sum().item()} nodes")
    
    # Create nodes_dict
    nodes_dict = {
        'shared': shared_idx,
        'candidate': candidate_idx,
        'independent': independent_idx,
        'extra': extra_indices,
        'val': torch.where(data.val_mask)[0].tolist(),
        'test': torch.where(data.test_mask)[0].tolist()
    }
    
    # Verify no data leakage
    verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, logger)
    
    
    # Train models
    logger.info("\nTraining models...")
    model_f, model_g, f_val_acc, g_val_acc, f_test_accs, g_test_accs, f_models, g_models, model_training_times = train_models(
        args=args,
        data=data,
        shared_idx=shared_idx,
        candidate_idx=candidate_idx,
        independent_idx=independent_idx,
        device=device,
        logger=logger,
        output_dir=log_dir,
        seeds=training_seeds  # Pass seeds to train_models
    )
    
    
    # Calculate memorization scores for the best model
    logger.info("\nCalculating memorization scores for best model...")
    best_model_scores = calculate_node_memorization_score(
        model_f=model_f,
        model_g=model_g,
        data=data,
        nodes_dict=nodes_dict,
        device=device,
        logger=logger,
        num_passes=args.num_passes
    )
    
    # Save memorization scores to disk
    save_memorization_scores(best_model_scores, log_dir, logger)
    
    # Calculate and log average scores for each node type
    for node_type, scores_dict in best_model_scores.items():
        logger.info(f"Best model memorization score for {node_type} nodes: {scores_dict['avg_score']:.4f}")
        # Also log the number of nodes above threshold
        logger.info(f"Nodes with score > 0.5: {scores_dict['nodes_above_threshold']}/{len(scores_dict['mem_scores'])} ({scores_dict['percentage_above_threshold']:.1f}%)")
    
    # Perform statistical tests on best model's memorization scores
    perform_memorization_statistical_tests(best_model_scores, logger)
    
    # Create visualization for best model scores
    plot_filename = f'{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}_best.pdf'
    plot_path = os.path.join(log_dir, plot_filename)
    
    plot_node_memorization_analysis(
        node_scores=best_model_scores,
        save_path=plot_path,
        title_suffix=f"Best Model | Dataset: {args.dataset}, Model: {args.model_type}\nf_acc={f_val_acc:.3f}, g_acc={g_val_acc:.3f}",
        node_types_to_plot=['shared', 'candidate', 'independent', 'extra']
    )
    logger.info(f"Best model memorization score plot saved to: {plot_path}")
    
    # Add separate memorization score distribution plots for each node type
    separate_plots_filename = f'{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}_separate'
    separate_plots_path = os.path.join(log_dir, separate_plots_filename)
    plot_separate_node_memorization_plots(
        node_scores=best_model_scores,
        save_path=separate_plots_path
    )
    logger.info(f"Separate node type memorization score plots saved with base path: {separate_plots_path}")


    logger.info("\nRunning kNN label disagreement analysis with best model (for visualization)...")
    best_knn_results = run_knn_label_disagreement_analysis(
        data=data,
        nodes_dict=nodes_dict,
        node_scores=best_model_scores,
        save_dir=log_dir,
        model_type=args.model_type,
        dataset_name=args.dataset,
        timestamp=timestamp,
        k=3,  # Default k value
        threshold=0.5,  # Default threshold for memorization
        similarity_metric='euclidean',  # Default similarity metric
        device=device,
        logger=logger
    )
    
if __name__ == "__main__":
    main()
