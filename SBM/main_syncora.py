import argparse
import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from getdataset import load_and_process_dataset
from model import *
from memorization import calculate_node_memorization_score, plot_separate_memorization_analysis
from train import *



def create_visualization(results_df, save_path, args):
    # Plot 1: Original Homophily vs Informativeness with memorization rate as color
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=500,
        alpha=0.8
    )
    plt.xticks(fontsize=18) # Set x-tick font size to 20
    plt.yticks(fontsize=18) # Set y-tick font size to 20
    plt.xlabel('Homophily', fontsize=25 ,font='Sans serif')
    plt.ylabel('Label Informativeness', fontsize=25 ,font='Sans serif')
    #plt.title(f'Homophily vs Label Informativeness\n(color: memorization rate)\n{args.model_type.upper()}', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=25, font='Sans serif')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add padding to axes
    x_min, x_max = results_df['homophily'].min(), results_df['homophily'].max()
    y_min, y_max = results_df['informativeness'].min(), results_df['informativeness'].max()
    plt.xlim(x_min - 0.05, x_max + 0.05)
    plt.ylim(y_min - 0.05, y_max + 0.05)
    
    # Save plot 1
    plot1_path = save_path.replace('.png', '_homophily_vs_informativeness.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Label Informativeness vs Memorization Rate
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['informativeness'], results_df['percent_memorized'], 
                color='blue', s=500, alpha=0.7)
    
    # Add trend line using numpy's polyfit
    z = np.polyfit(results_df['informativeness'], results_df['percent_memorized'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results_df['informativeness'].min(), results_df['informativeness'].max(), 100)
    #plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Calculate correlation coefficient
    corr_inf = np.corrcoef(results_df['informativeness'], results_df['percent_memorized'])[0,1]
    
    plt.xlabel('Label Informativeness', fontsize=25 ,font='Sans serif')
    plt.ylabel('Memorization Rate (%)', fontsize=25 ,font='Sans serif')
    #plt.title(f'Label Informativeness vs Memorization\n(r = {corr_inf:.2f})\n{args.model_type.upper()}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add padding to axes
    x_min, x_max = results_df['informativeness'].min(), results_df['informativeness'].max()
    y_min, y_max = results_df['percent_memorized'].min(), results_df['percent_memorized'].max()
    plt.xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    plt.ylim(y_min - 5, y_max + 5)
    
    # Save plot 2
    plot2_path = save_path.replace('.png', '_informativeness_vs_memorization.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Homophily vs Memorization Rate
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['homophily'], results_df['percent_memorized'], 
                color='green', s=500, alpha=0.7)
    
    # Add trend line
    z = np.polyfit(results_df['homophily'], results_df['percent_memorized'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results_df['homophily'].min(), results_df['homophily'].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Calculate correlation coefficient
    corr_hom = np.corrcoef(results_df['homophily'], results_df['percent_memorized'])[0,1]
    
    plt.xlabel('Homophily', fontsize=24 ,font='Sans serif')
    plt.ylabel('Memorization Rate (%)', fontsize=24 ,font='Sans serif')
    #plt.title(f'Homophily vs Memorization\n(r = {corr_hom:.2f})\n{args.model_type.upper()}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add padding to axes
    x_min, x_max = results_df['homophily'].min(), results_df['homophily'].max()
    y_min, y_max = results_df['percent_memorized'].min(), results_df['percent_memorized'].max()
    plt.xlim(x_min - 0.05, x_max + 0.05)
    plt.ylim(y_min - 5, y_max + 5)
    
    # Save plot 3
    plot3_path = save_path.replace('.png', '_homophily_vs_memorization.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_node_splits(data, train_mask, swap_candidate_independent=False):
    """
    Create node splits using all available training nodes.
    
    Args:
        data: PyG data object
        train_mask: Mask for train nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train indices in their original order
    train_indices = torch.where(train_mask)[0]
    num_train_nodes = len(train_indices)
    
    # Calculate split sizes: 50% shared, 25% candidate, 25% independent
    shared_size = int(0.50 * num_train_nodes)
    remaining = num_train_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_indices[:shared_size].tolist()
    candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    independent_idx = train_indices[shared_size + split_size:].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, independent_idx, candidate_idx
    else:
        return shared_idx, candidate_idx, independent_idx



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_passes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/syncora_analysis')
    parser.add_argument('--analyze_effective_rank', action='store_true',
                       help='Analyze effective rank of node embeddings')
    parser.add_argument('--analyze_weight_identity', action='store_true',
                       help='Analyze weight identity matrices')
    parser.add_argument('--best_model_only', action='store_true', default=True,
                       help='Only generate plots for the best model of each homophily level')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'syncora_analysis_{args.model_type}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging with both file and console output
    logger = logging.getLogger('syncora_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Select homophily levels to analyze
    homophily_levels = [1.0]  # Changed to match available files
    
    # Initialize results containers
    results = []  # Initialize as list for storing main results


    # Original code path - process each dataset file individually
    dataset_files = [f'h{h:.2f}-r1' for h in homophily_levels]
    
    for dataset_name in tqdm(dataset_files, desc="Processing datasets"):
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Load and process dataset
        data = load_and_process_dataset(args, dataset_name, logger)
        
        # Move individual tensors to device instead of entire Data object
        data.x = data.x.to(device) if hasattr(data, 'x') else None
        data.edge_index = data.edge_index.to(device) if hasattr(data, 'edge_index') else None
        data.y = data.y.to(device) if hasattr(data, 'y') else None
        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask.to(device)
        if hasattr(data, 'val_mask'):
            data.val_mask = data.val_mask.to(device)
        if hasattr(data, 'test_mask'):
            data.test_mask = data.test_mask.to(device)
        
        # Get node splits
        shared_idx, candidate_idx, independent_idx = get_node_splits(
            data, data.train_mask, swap_candidate_independent=False
        )
        
        # Get extra indices from test set
        test_indices = torch.where(data.test_mask)[0]
        extra_size = len(candidate_idx)
        extra_indices = test_indices[:extra_size].tolist()
        
        # Create nodes_dict
        nodes_dict = {
            'shared': shared_idx,
            'candidate': candidate_idx,
            'independent': independent_idx,
            'extra': extra_indices,
            'val': torch.where(data.val_mask)[0].tolist(),
            'test': torch.where(data.test_mask)[0].tolist()
        }
        
        # Train models
        model_f, model_g, f_val_acc, g_val_acc = train_models(
            args=args,
            data=data,
            shared_idx=shared_idx,
            candidate_idx=candidate_idx,
            independent_idx=independent_idx,
            device=device,
            logger=logger,
            output_dir=None
        )
        
        # Calculate memorization scores
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=logger,
            num_passes=args.num_passes
        )
        
        # Plot separate memorization analysis visualizations for this homophily level
        logger.info("\nPlotting separate memorization analysis visualizations...")
        
        # Extract homophily level from dataset_name for better file naming
        homophily_level = data.homophily
        histogram_base_path = os.path.join(log_dir, f'memorization_h{homophily_level:.2f}')
        
        plot_paths = plot_separate_memorization_analysis(
            node_scores=node_scores,
            save_path=histogram_base_path,
            title_suffix=f"Homophily: {data.homophily:.2f}, Informativeness: {data.informativeness:.2f}",
            node_types_to_plot=['shared', 'candidate', 'independent', 'extra']
        )
        
        # Log paths to generated plots
        logger.info(f"Memorization visualizations created:")
        if plot_paths["confidence_comparison"]:
            logger.info(f"- Confidence comparison: {os.path.basename(plot_paths['confidence_comparison'])}")
        logger.info(f"- Combined histogram: {os.path.basename(plot_paths['combined_histogram'])}")
        logger.info(f"- Model f confidence: {os.path.basename(plot_paths['model_f_confidence'])}")
        logger.info(f"- Model g confidence: {os.path.basename(plot_paths['model_g_confidence'])}")
        logger.info("- Individual histograms:")
        for node_type, path in plot_paths["individual_histograms"].items():
            logger.info(f"  * {node_type}: {os.path.basename(path)}")
        
        # Store results
        results.append({
            'dataset': dataset_name,
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': node_scores['candidate']['percentage_above_threshold'],
            'avg_memorization': node_scores['candidate']['avg_score'],
            'num_memorized': node_scores['candidate']['nodes_above_threshold'],
            'total_nodes': len(node_scores['candidate']['mem_scores']),
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc)
        })
            


if __name__ == '__main__':
    main()