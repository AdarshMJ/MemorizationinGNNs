import torch
import numpy as np
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from matplotlib.ticker import MaxNLocator

def verify_node_access(node_idx: int, node_type: str, nodes_dict: Dict[str, List[int]]):
    """Verify that node belongs to the correct set"""
    if node_idx not in nodes_dict[node_type]:
        raise ValueError(f"Node {node_idx} is being processed as {node_type} but not found in that set!")

def calculate_node_memorization_score(
    model_f,
    model_g,
    data,
    nodes_dict,
    device: torch.device = None,
    logger=None,
    num_passes: int = 1  # Number of forward passes to average over
) -> Dict[str, Dict]:
    """Calculate memorization scores using confidence differences between models"""
    # Ensure models are in eval mode
    model_f.eval()
    model_g.eval()
    
    if device is None:
        device = next(model_f.parameters()).device
    
    # Set deterministic mode for inference
    #torch.use_deterministic_algorithms(True)
    
    results = {}
    
    # Process each node type
    for node_type, nodes in nodes_dict.items():
        if node_type in ['val', 'test']:  # Skip validation and test nodes
            continue
        
        # Lists to store scores and predictions
        mem_scores = []
        f_confidences = []
        g_confidences = []
        correct_f = 0
        correct_g = 0
        
        # Track memorized vs non-memorized accuracies
        memorized_correct_f = 0
        memorized_correct_g = 0
        non_memorized_correct_f = 0
        non_memorized_correct_g = 0
        memorized_count = 0
        non_memorized_count = 0
        
        # Track high confidence wrong predictions
        memorized_high_conf_wrong_f = 0
        memorized_high_conf_wrong_g = 0
        non_memorized_high_conf_wrong_f = 0
        non_memorized_high_conf_wrong_g = 0
        
        all_scores = []
        
        # Run multiple passes and average if requested
        for _ in range(num_passes):
            # Get model predictions with deterministic computation
            with torch.no_grad():
                with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
                    out_f = torch.softmax(model_f(data.x.to(device), data.edge_index.to(device)), dim=1)
                    out_g = torch.softmax(model_g(data.x.to(device), data.edge_index.to(device)), dim=1)
            
            # Calculate scores for each node
            for node_idx in nodes:
                true_label = data.y[node_idx].item()
                prob_f = out_f[node_idx, true_label].item()
                prob_g = out_g[node_idx, true_label].item()
                
                # Calculate memorization score as difference in confidence
                mem_score = prob_f - prob_g
                
                # Track predictions for accuracy
                pred_f = out_f[node_idx].argmax().item()
                pred_g = out_g[node_idx].argmax().item()
                max_conf_f = out_f[node_idx].max().item()
                max_conf_g = out_g[node_idx].max().item()
                
                # Update overall accuracies
                if pred_f == true_label:
                    correct_f += 1
                if pred_g == true_label:
                    correct_g += 1
                
                # Update memorized vs non-memorized statistics
                if mem_score > 0.5:  # Memorized node
                    memorized_count += 1
                    if pred_f == true_label:
                        memorized_correct_f += 1
                    if pred_g == true_label:
                        memorized_correct_g += 1
                    if pred_f != true_label and max_conf_f > 0.8:
                        memorized_high_conf_wrong_f += 1
                    if pred_g != true_label and max_conf_g > 0.8:
                        memorized_high_conf_wrong_g += 1
                else:  # Non-memorized node
                    non_memorized_count += 1
                    if pred_f == true_label:
                        non_memorized_correct_f += 1
                    if pred_g == true_label:
                        non_memorized_correct_g += 1
                    if pred_f != true_label and max_conf_f > 0.8:
                        non_memorized_high_conf_wrong_f += 1
                    if pred_g != true_label and max_conf_g > 0.8:
                        non_memorized_high_conf_wrong_g += 1
                
                all_scores.append({
                    'node_idx': node_idx,
                    'node_type': node_type,
                    'true_label': true_label,
                    'pred_f': pred_f,
                    'pred_g': pred_g,
                    'conf_f': prob_f,
                    'conf_g': prob_g,
                    'mem_score': mem_score,
                    'max_conf_f': max_conf_f,
                    'max_conf_g': max_conf_g
                })
                
                mem_scores.append(mem_score)
                f_confidences.append(prob_f)
                g_confidences.append(prob_g)
        
        # Skip node type if no valid scores were calculated
        if not mem_scores:
            if logger:
                logger.warning(f"No valid scores calculated for {node_type} nodes")
            continue
        
        # Calculate averages and statistics
        avg_score = np.mean(mem_scores)
        accuracy_f = correct_f / (len(nodes) * num_passes) if nodes else 0
        accuracy_g = correct_g / (len(nodes) * num_passes) if nodes else 0
        
        # Calculate memorized vs non-memorized statistics
        memorized_acc_f = memorized_correct_f / memorized_count if memorized_count > 0 else 0
        memorized_acc_g = memorized_correct_g / memorized_count if memorized_count > 0 else 0
        non_memorized_acc_f = non_memorized_correct_f / non_memorized_count if non_memorized_count > 0 else 0
        non_memorized_acc_g = non_memorized_correct_g / non_memorized_count if non_memorized_count > 0 else 0
        
        # Calculate high confidence wrong predictions percentages
        memorized_high_conf_wrong_f_pct = (memorized_high_conf_wrong_f / memorized_count * 100) if memorized_count > 0 else 0
        memorized_high_conf_wrong_g_pct = (memorized_high_conf_wrong_g / memorized_count * 100) if memorized_count > 0 else 0
        non_memorized_high_conf_wrong_f_pct = (non_memorized_high_conf_wrong_f / non_memorized_count * 100) if non_memorized_count > 0 else 0
        non_memorized_high_conf_wrong_g_pct = (non_memorized_high_conf_wrong_g / non_memorized_count * 100) if non_memorized_count > 0 else 0
        
        # Calculate nodes above threshold (0.5)
        nodes_above_threshold = sum(1 for score in mem_scores if score > 0.5)
        percentage_above_threshold = (nodes_above_threshold / len(mem_scores)) * 100 if mem_scores else 0
        
        # Store results
        results[node_type] = {
            'mem_scores': mem_scores,
            'f_confidences': f_confidences,
            'g_confidences': g_confidences,
            'avg_score': avg_score,
            'accuracy_f': accuracy_f,
            'accuracy_g': accuracy_g,
            'nodes_above_threshold': nodes_above_threshold,
            'percentage_above_threshold': percentage_above_threshold,
            'memorized_stats': {
                'count': memorized_count,
                'accuracy_f': memorized_acc_f,
                'accuracy_g': memorized_acc_g,
                'high_conf_wrong_f_pct': memorized_high_conf_wrong_f_pct,
                'high_conf_wrong_g_pct': memorized_high_conf_wrong_g_pct
            },
            'non_memorized_stats': {
                'count': non_memorized_count,
                'accuracy_f': non_memorized_acc_f,
                'accuracy_g': non_memorized_acc_g,
                'high_conf_wrong_f_pct': non_memorized_high_conf_wrong_f_pct,
                'high_conf_wrong_g_pct': non_memorized_high_conf_wrong_g_pct
            },
            'raw_data': pd.DataFrame(all_scores)
        }
        
        if logger:
            logger.info(f"\nNode type: {node_type}")
            logger.info(f"  Average memorization score: {avg_score:.4f}")
            logger.info(f"  Overall accuracies:")
            logger.info(f"    Model f: {accuracy_f:.4f}")
            logger.info(f"    Model g: {accuracy_g:.4f}")
            
            logger.info(f"\n  Memorized nodes (score > 0.5): {memorized_count} nodes")
            logger.info(f"    Accuracy model f: {memorized_acc_f:.4f}")
            logger.info(f"    Accuracy model g: {memorized_acc_g:.4f}")
            logger.info(f"    High confidence wrong predictions:")
            logger.info(f"      Model f: {memorized_high_conf_wrong_f_pct:.1f}%")
            logger.info(f"      Model g: {memorized_high_conf_wrong_g_pct:.1f}%")
            
            logger.info(f"\n  Non-memorized nodes: {non_memorized_count} nodes")
            logger.info(f"    Accuracy model f: {non_memorized_acc_f:.4f}")
            logger.info(f"    Accuracy model g: {non_memorized_acc_g:.4f}")
            logger.info(f"    High confidence wrong predictions:")
            logger.info(f"      Model f: {non_memorized_high_conf_wrong_f_pct:.1f}%")
            logger.info(f"      Model g: {non_memorized_high_conf_wrong_g_pct:.1f}%")
    
    # Restore default settings
    torch.use_deterministic_algorithms(False)
    
    return results

def plot_node_memorization_analysis(
    node_scores: Dict[str, Dict],
    save_path: str,
    title_suffix="",
    node_types_to_plot: List[str] = None
):
    """
    Plot node memorization analysis results based on confidence score differences
    Args:
        node_scores: Dictionary containing scores for each node type
        save_path: Path to save the plot (base path, we'll add suffixes for each plot)
        title_suffix: Additional text to add to plot titles
        node_types_to_plot: List of node types to include in histogram (e.g., ['shared', 'candidate'])
                          If None, all node types will be plotted
    """
    # Extract base path and extension
    base_path, ext = os.path.splitext(save_path)
    if not ext:
        ext = '.png'  # Default extension if none provided
        
    # Color and label definitions
    colors = {'candidate': 'blue', 'independent': 'orange', 'extra': 'green', 'shared': 'red'}
    labels = {'candidate': '$S_C$', 'independent': '$S_I$', 'extra': '$S_E$', 'shared': '$S_S$'}

    # If no specific types are provided, plot all available types
    if node_types_to_plot is None:
        node_types_to_plot = list(node_scores.keys())
    
    num_bins = 20
    threshold = 0.5
    
    # Plot 1: Model confidence comparison scatter plot (only for candidate nodes)
    if 'candidate' in node_scores:
        plt.figure(figsize=(8, 6))
        f_confidences = node_scores['candidate']['f_confidences']
        g_confidences = node_scores['candidate']['g_confidences']
        mem_scores = node_scores['candidate']['mem_scores']
        
        # Add y=x line in red
        min_val = min(min(f_confidences), min(g_confidences))
        max_val = max(max(f_confidences), max(g_confidences))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        
        # Create scatter plot with viridis colormap
        scatter = plt.scatter(f_confidences, g_confidences, 
                            c=mem_scores, cmap='viridis', 
                            alpha=0.6, s=50)
        plt.colorbar(scatter, label='Memorization Score')
        
        plt.xlabel('Model f Confidence')
        plt.ylabel('Model g Confidence')
        plt.title('Confidence Comparison (Candidate Nodes)')
        if title_suffix:
            plt.suptitle(title_suffix, fontsize=14)
            plt.subplots_adjust(top=0.85)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_path = f"{base_path}_confidence_comparison{ext}"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Histogram for specified node types
    plt.figure(figsize=(8, 6))
    
    # Create histogram with frequency counts
    for node_type in node_types_to_plot:
        if node_type in node_scores:
            scores = node_scores[node_type]['mem_scores']
            mean_score = node_scores[node_type]['avg_score']
            nodes_above = node_scores[node_type]['nodes_above_threshold']
            total_nodes = len(scores)
            percentage_above = (nodes_above / total_nodes) * 100
            
            # Plot histogram with frequency counts
            # plt.hist(scores, bins=num_bins, alpha=0.5, color=colors[node_type],
            #          label=f"{labels[node_type]} ({total_nodes} nodes, {nodes_above} > 0.5, {percentage_above:.1f}%)")
            plt.hist(scores, bins=num_bins, alpha=0.5, color=colors[node_type],
                     label=f"{labels[node_type]}")
    # Add vertical line at the threshold
    #plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
     #           label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score', fontsize=25, font='Sans serif') # Keep size 30
    plt.ylabel('Frequency Count', fontsize=25, font='Sans serif') # Keep size 30
    #title = 'Frequency Distribution of Memorization Scores'
    #plt.title(title)
    #if title_suffix:
    #    plt.suptitle(title_suffix, fontsize=14)
      #  plt.subplots_adjust(top=0.85)
    plt.xlim(-1, 1)
    plt.ylim(2, 120)
    plt.xticks(fontsize=18) # Set x-tick font size to 20
    plt.yticks(fontsize=18) # Set y-tick font size to 20
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=15) # Increase legend font size
    plt.tight_layout()
    hist_path = f"{base_path}_mem_score_distribution{ext}"
    plt.savefig(hist_path, dpi=500, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Model f confidence distribution
    plt.figure(figsize=(8, 6))
    for node_type in node_types_to_plot:
        if node_type in node_scores:
            confidences = node_scores[node_type]['f_confidences']
            mean_conf = np.mean(confidences)
            
            plt.hist(confidences, bins=num_bins, alpha=0.5, color=colors[node_type],
                     label=f"{labels[node_type]} (mean={mean_conf:.3f})")
    
    plt.xlabel('Model f Confidence')
    plt.ylabel('Frequency Count')
    #plt.title('Frequency Distribution of Model f Confidence Scores')
    #if title_suffix:
     #   plt.suptitle(title_suffix, fontsize=14)
      #  plt.subplots_adjust(top=0.85)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    f_conf_path = f"{base_path}_model_f_confidence{ext}"
    plt.savefig(f_conf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Model g confidence distribution
    plt.figure(figsize=(8, 6))
    for node_type in node_types_to_plot:
        if node_type in node_scores:
            confidences = node_scores[node_type]['g_confidences']
            mean_conf = np.mean(confidences)
            
            plt.hist(confidences, bins=num_bins, alpha=0.5, color=colors[node_type],
                     label=f"{labels[node_type]} (mean={mean_conf:.3f})")
    
    plt.xlabel('Model g Confidence')
    plt.ylabel('Frequency Count')
    plt.title('Frequency Distribution of Model g Confidence Scores')
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=14)
        plt.subplots_adjust(top=0.85)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    g_conf_path = f"{base_path}_model_g_confidence{ext}"
    plt.savefig(g_conf_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_separate_node_memorization_plots(
    node_scores: Dict[str, Dict],
    save_path: str
):
    """
    Create separate memorization score vs frequency plots for each node type.
    
    Args:
        node_scores: Dictionary containing scores for each node type
        save_path: Base path to save the plots (will append node type)
    """
    # Extract base path without extension
    base_path = os.path.splitext(save_path)[0]
    
    # Color definitions
    colors = {'candidate': 'blue', 'independent': 'orange', 'extra': 'green', 'shared': 'red'}
    labels = {'candidate': 'Memorization on $S_C$', 'independent': '$S_I$', 'extra': '$S_E$', 'shared': '$S_S$'}
    
    # Create separate plot for each node type
    for node_type in node_scores.keys():
        if node_type in ['val', 'test']:
            continue
            
        plt.figure(figsize=(8, 6))
        
        # Get scores for this node type
        mem_scores = node_scores[node_type]['mem_scores']
        
        # Create histogram
        plt.hist(mem_scores, bins=20, range=(-1, 1), color=colors[node_type], 
                alpha=0.7, density=False, label=labels[node_type])
        
        # Set axis ranges
        plt.xlim(-1, 1)
        plt.ylim(2, 120)
        
        # Set labels and appearance
        plt.xlabel('Memorization Score', fontsize=25, font='Sans Serif')
        plt.ylabel('Frequency Count', fontsize=25, font='Sans Serif')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16, bbox_to_anchor=(0.99, 0.98), loc='upper right', borderaxespad=0, handlelength=1)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{base_path}_{node_type}_distribution.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()