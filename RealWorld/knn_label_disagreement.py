import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import os
from typing import Dict, List, Union, Tuple
from torch_geometric.data import Data

def calculate_knn_label_disagreement(
    data: Data,
    nodes_dict: Dict[str, List[int]],
    node_scores: Dict[str, Dict],
    k: int = 10,
    threshold: float = 0.5,
    similarity_metric: str = 'euclidean',
    device: str = 'cpu',
    save_dir: str = None,
    timestamp: str = None,
    model_type: str = None,
    dataset_name: str = None,
    logger = None
) -> Tuple[pd.DataFrame, Dict]:

    if logger:
        logger.info("\nCalculating Label Disagreement Scores...")
        logger.info(f"Parameters: k={k}, similarity_metric={similarity_metric}, threshold={threshold}")
    
    # 1. Preprocessing
    # Define the training set indices for model f (shared + candidate)
    train_f_indices = nodes_dict['shared'] + nodes_dict['candidate']
    
    # Convert to numpy array if not already
    if isinstance(train_f_indices, torch.Tensor):
        train_f_indices = train_f_indices.cpu().numpy()
    elif isinstance(train_f_indices, list):
        train_f_indices = np.array(train_f_indices)
    
    # Get candidate set indices
    sc_indices = nodes_dict['candidate']
    if isinstance(sc_indices, torch.Tensor):
        sc_indices = sc_indices.cpu().numpy()
    elif isinstance(sc_indices, list):
        sc_indices = np.array(sc_indices)
    
    # Extract features for the training set of model f
    X_train_f = data.x[train_f_indices].cpu().numpy()
    
    # Create a mapping from position in X_train_f back to original node index
    position_to_original_idx = {i: train_f_indices[i] for i in range(len(train_f_indices))}
    
    # Create a mapping from original node index to position in X_train_f
    original_idx_to_position = {train_f_indices[i]: i for i in range(len(train_f_indices))}
    
    # Identify memorized and non-memorized nodes in the candidate set
    candidate_df = node_scores['candidate']['raw_data']
    
    # Set appropriate distance metric for NearestNeighbors
    if similarity_metric == 'cosine':
        metric = 'cosine'  # Note: sklearn uses cosine distance, which is 1 - cosine similarity
    else:
        metric = similarity_metric  # For euclidean, manhattan, etc.
    
    # 2. Build kNN Index
    # Ensure k is not larger than the number of nodes in train_f
    effective_k = min(k, len(train_f_indices) - 1)
    if effective_k < k and logger:
        logger.warning(f"Reducing k from {k} to {effective_k} due to limited training set size")
    
    # Initialize and fit NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric, n_jobs=-1)
    nn_model.fit(X_train_f)
    
    # 3. Calculate kNN Disagreement
    disagreement_results = []
    
    for node_idx in sc_indices:
        # Get the node features
        x_node = data.x[node_idx].cpu().numpy().reshape(1, -1)
        
        # Get the true label of the node
        y_node = data.y[node_idx].item()
        
        # Find k+1 nearest neighbors in the training set
        distances, indices = nn_model.kneighbors(x_node)
        
        # Convert relative indices to original node indices
        knn_indices_original = np.array([position_to_original_idx[idx] for idx in indices[0]])
        
        # Remove the node itself from its neighbors
        self_idx_pos = np.where(knn_indices_original == node_idx)[0]
        if len(self_idx_pos) > 0:
            # The node is in its own neighborhood, remove it
            mask = knn_indices_original != node_idx
            knn_indices_original = knn_indices_original[mask]
            distances = distances[0][mask]
        
        # Keep only the top k neighbors
        knn_indices_original = knn_indices_original[:effective_k]
        
        # Get labels of the k nearest neighbors
        y_neighbors = data.y[knn_indices_original].cpu().numpy()
        
        # Calculate disagreement fraction
        disagreement = np.sum(y_neighbors != y_node) / len(y_neighbors)
        
        # Check if the node is memorized
        if node_idx in candidate_df['node_idx'].values:
            mem_score = candidate_df.loc[candidate_df['node_idx'] == node_idx, 'mem_score'].values[0]
            is_memorized = mem_score > threshold
        else:
            mem_score = None
            is_memorized = None
        
        # Store results
        disagreement_results.append({
            'node_idx': int(node_idx),
            'is_memorized': is_memorized,
            'mem_score': mem_score,
            'knn_disagreement': disagreement,
            'true_label': int(y_node)
        })
    
    # Convert results to DataFrame
    df_disagreement = pd.DataFrame(disagreement_results)
    
    # 4. Analyze and Visualize
    # Split results by memorization status
    memorized_disagreements = df_disagreement[df_disagreement['is_memorized'] == True]['knn_disagreement'].values
    non_memorized_disagreements = df_disagreement[df_disagreement['is_memorized'] == False]['knn_disagreement'].values
    
    # Calculate basic statistics
    mem_mean = np.mean(memorized_disagreements) if len(memorized_disagreements) > 0 else np.nan
    non_mem_mean = np.mean(non_memorized_disagreements) if len(non_memorized_disagreements) > 0 else np.nan
    
    mem_std = np.std(memorized_disagreements) if len(memorized_disagreements) > 0 else np.nan
    non_mem_std = np.std(non_memorized_disagreements) if len(non_memorized_disagreements) > 0 else np.nan
    
    # Mann-Whitney U test for statistical significance
    if len(memorized_disagreements) > 0 and len(non_memorized_disagreements) > 0:
        u_stat, p_val = stats.mannwhitneyu(memorized_disagreements, non_memorized_disagreements, alternative='two-sided')
    else:
        u_stat, p_val = None, None
    
    # Log statistics
    if logger:
        logger.info("\nkNN Label Disagreement Analysis:")
        logger.info(f"Memorized nodes (n={len(memorized_disagreements)}): mean={mem_mean:.4f}, std={mem_std:.4f}")
        logger.info(f"Non-memorized nodes (n={len(non_memorized_disagreements)}): mean={non_mem_mean:.4f}, std={non_mem_std:.4f}")
        if p_val is not None:
            logger.info(f"Mann-Whitney U test: U={u_stat:.1f}, p-value={p_val:.6f}")
            logger.info(f"Significant difference: {'Yes' if p_val < 0.05 else 'No'} (at α=0.05)")
    
    # Create a clean box plot visualization
    plt.figure(figsize=(8, 6))
    
    # Ensure the is_memorized column contains boolean values, not strings
    if df_disagreement['is_memorized'].dtype == 'object':
        df_disagreement['is_memorized'] = df_disagreement['is_memorized'].astype(bool)
    
    # Create a clean boxplot with clear colors
    boxplot = sns.boxplot(x='is_memorized', y='knn_disagreement', data=df_disagreement, 
                        palette=['lightgreen', 'lightblue'], width=0.6,
                        showfliers=True, fliersize=3)
    
    # Calculate means for annotation
    memorized_mean = np.mean(df_disagreement[df_disagreement['is_memorized'] == True]['knn_disagreement'])
    non_memorized_mean = np.mean(df_disagreement[df_disagreement['is_memorized'] == False]['knn_disagreement'])
    
    # Draw mean markers (diamonds)
    plt.plot(0, non_memorized_mean, 'D', color='green', markersize=8)
    plt.plot(1, memorized_mean, 'D', color='blue', markersize=8)
    
    # Get number of nodes in each category
    n_memorized = len(df_disagreement[df_disagreement['is_memorized'] == True])
    n_non_memorized = len(df_disagreement[df_disagreement['is_memorized'] == False])
    
    # Update x-axis labels
    plt.xticks([0, 1], [f'Non-memorized\n(n={n_non_memorized})', f'Memorized\n(n={n_memorized})'])
    
    # Add clean horizontal grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set labels and title
    plt.ylabel('Feature-Label Disagreement', fontsize=12)
    plt.xlabel('')  # Remove x-axis label
    
    # If significant difference, add a subtle indicator bar
    if p_val is not None and p_val < 0.01:
        # Add significance bracket
        x1, x2 = 0, 1
        y = max(np.max(memorized_disagreements), np.max(non_memorized_disagreements)) + 0.05
        plt.plot([x1, x1, x2, x2], [y, y+0.02, y+0.02, y], lw=1.5, c='black')
        
        # Add asterisks based on significance level
        if p_val < 0.001:
            sig_symbol = '***'
        elif p_val < 0.01:
            sig_symbol = '**'
        elif p_val < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'n.s.'
        
        plt.text((x1+x2)*.5, y+0.03, sig_symbol, ha='center', va='bottom', color='black')
    
    # Clean up the plot
    plt.tight_layout()
    
    # Save the plot if save_dir is provided
    if save_dir and timestamp:
        # Create the save path
        save_name = f'knn_disagreement_{model_type}_{dataset_name}_{timestamp}.png'
        save_path = os.path.join(save_dir, save_name)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Plot saved to: {save_path}")
    
    plt.close()
    
    # Create a scatter plot showing the relationship between memorization score and disagreement
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with different colors for memorized vs. non-memorized
    sns.scatterplot(x='mem_score', y='knn_disagreement', hue='is_memorized',
                   data=df_disagreement, palette={True: 'blue', False: 'green'}, alpha=0.7)
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Threshold = {threshold}')
    
    # Calculate and plot regression line
    if len(df_disagreement) > 2:  # Need at least 3 points for regression
        x = df_disagreement['mem_score'].values
        y = df_disagreement['knn_disagreement'].values
        
        # Calculate correlation
        corr, p_corr = stats.pearsonr(x, y)
        
        # Plot regression line if correlation is significant
        if p_corr < 0.05:
            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m*x + b, color='black', linestyle='-', alpha=0.5)
            plt.annotate(f'r = {corr:.3f} (p = {p_corr:.4f})',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='center',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    # Set plot labels and title
    plt.xlabel('Memorization Score')
    plt.ylabel('Feature Label Disagreement')
    scatter_title = f"Memorization Score vs. kNN Label Disagreement (k={effective_k})"
    if dataset_name and model_type:
        scatter_title += f"\nDataset: {dataset_name}, Model: {model_type}"
    plt.title(scatter_title)
    
    plt.legend(title='Is Memorized')
    
    # Save the scatter plot
    if save_dir and timestamp:
        # Create the save path
        scatter_name = f'knn_disagreement_scatter_{model_type}_{dataset_name}_{timestamp}.png'
        scatter_path = os.path.join(save_dir, scatter_name)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Scatter plot saved to: {scatter_path}")
    
    plt.close()
    
    # Save the raw disagreement data
    if save_dir and timestamp:
        csv_path = os.path.join(save_dir, f'knn_disagreement_data_{model_type}_{dataset_name}_{timestamp}.csv')
        df_disagreement.to_csv(csv_path, index=False)
        if logger:
            logger.info(f"Raw data saved to: {csv_path}")
    
    # Compile analysis results
    analysis_results = {
        'memorized': {
            'mean': mem_mean,
            'std': mem_std,
            'count': len(memorized_disagreements)
        },
        'non_memorized': {
            'mean': non_mem_mean,
            'std': non_mem_std,
            'count': len(non_memorized_disagreements)
        },
        'statistical_test': {
            'u_statistic': u_stat,
            'p_value': p_val,
            'significant': p_val < 0.01 if p_val is not None else None
        },
        'parameters': {
            'k': effective_k,
            'similarity_metric': similarity_metric,
            'threshold': threshold
        }
    }
    
    return df_disagreement, analysis_results

def run_knn_label_disagreement_analysis(
    data: Data,
    nodes_dict: Dict[str, List[int]],
    node_scores: Dict[str, Dict],
    save_dir: str,
    model_type: str,
    dataset_name: str,
    timestamp: str,
    k: int = 10,
    threshold: float = 0.5,
    similarity_metric: str = 'euclidean',
    device: str = 'cpu',
    logger = None
) -> Dict:
    """
    Run the kNN Label Disagreement analysis with various parameter configurations.
    """
    results = {}
    
    if logger:
        logger.info("\n===== Running kNN Label Disagreement Analysis =====")
    
    # Run the analysis with different k values
    k_values = [3,5,10]
    metrics = ['euclidean']
    
    # First run with the default parameters
    df, analysis = calculate_knn_label_disagreement(
        data=data,
        nodes_dict=nodes_dict,
        node_scores=node_scores,
        k=k,
        threshold=threshold,
        similarity_metric=similarity_metric,
        device=device,
        save_dir=save_dir,
        timestamp=timestamp,
        model_type=model_type,
        dataset_name=dataset_name,
        logger=logger
    )
    
    results['default'] = {
        'dataframe': df,
        'analysis': analysis
    }
    
    # Run with different parameter combinations for sensitivity analysis
    # This is optional and can be commented out if not needed
    sensitivity_results = {}
    
    for curr_k in k_values:
        if curr_k == k:  # Skip if already done in default analysis
            continue
            
        for curr_metric in metrics:
            if curr_metric == similarity_metric and curr_k == k:  # Skip if already done in default analysis
                continue
                
            config_name = f"k{curr_k}_{curr_metric}"
            
            if logger:
                logger.info(f"\nRunning configuration: {config_name}")
                
            df, analysis = calculate_knn_label_disagreement(
                data=data,
                nodes_dict=nodes_dict,
                node_scores=node_scores,
                k=curr_k,
                threshold=threshold,
                similarity_metric=curr_metric,
                device=device,
                save_dir=save_dir,
                timestamp=timestamp,
                model_type=model_type,
                dataset_name=dataset_name,
                logger=logger
            )
            
            sensitivity_results[config_name] = {
                'dataframe': df,
                'analysis': analysis
            }
    
    results['sensitivity'] = sensitivity_results
    
    # Find statistically significant configurations
    significant_configs = []
    
    # Check if default configuration is significant
    default_p_value = results['default']['analysis']['statistical_test']['p_value']
    if default_p_value is not None and default_p_value < 0.01:
        significant_configs.append({
            'name': 'default',
            'p_value': default_p_value,
            'k': results['default']['analysis']['parameters']['k'],
            'metric': results['default']['analysis']['parameters']['similarity_metric']
        })
    
    # Check sensitivity results for significant configurations
    for config_name, config_results in sensitivity_results.items():
        config_p_value = config_results['analysis']['statistical_test']['p_value']
        
        # Skip configurations with None p-values
        if config_p_value is None:
            continue
            
        # Add to significant configs if p-value is below threshold
        if config_p_value < 0.01:
            significant_configs.append({
                'name': config_name,
                'p_value': config_p_value,
                'k': config_results['analysis']['parameters']['k'],
                'metric': config_results['analysis']['parameters']['similarity_metric']
            })
    
    # Sort significant configs by k (ascending) and then by p-value (ascending)
    significant_configs.sort(key=lambda x: (x['k'], x['p_value']))
    
    # If we have significant configurations, use the one with smallest k
    # If multiple with same k, use the one with lowest p-value (it's sorted that way)
    if significant_configs:
        best_config = significant_configs[0]
    else:
        # If no significant configurations, revert to the one with the lowest p-value
        default_p = float('inf') if default_p_value is None else default_p_value
        best_p_value = default_p
        best_config = {'name': 'default', 'p_value': default_p_value, 
                       'k': results['default']['analysis']['parameters']['k'],
                       'metric': results['default']['analysis']['parameters']['similarity_metric']}
        
        for config_name, config_results in sensitivity_results.items():
            config_p_value = config_results['analysis']['statistical_test']['p_value']
            if config_p_value is not None and config_p_value < best_p_value:
                best_p_value = config_p_value
                best_config = {
                    'name': config_name,
                    'p_value': config_p_value,
                    'k': config_results['analysis']['parameters']['k'],
                    'metric': config_results['analysis']['parameters']['similarity_metric']
                }
    
    # Store the best configuration
    results['best_config'] = best_config
    results['significant_configs'] = significant_configs
    
    # Visualize the best configuration
    if best_config['name'] != 'default':
        # Get data for best configuration from sensitivity results
        best_df = sensitivity_results[best_config['name']]['dataframe']
        best_analysis = sensitivity_results[best_config['name']]['analysis']
        best_k = best_analysis['parameters']['k']
        best_metric = best_analysis['parameters']['similarity_metric']
        
        # Create visualization for the best configuration
        if logger:
            logger.info(f"\nCreating visualization for best configuration: {best_config['name']} (k={best_k}, metric={best_metric})")
        
        # Create a clean box plot for best configuration
        plt.figure(figsize=(8, 6))
        
        # Ensure the is_memorized column contains boolean values, not strings
        if best_df['is_memorized'].dtype == 'object':
            best_df['is_memorized'] = best_df['is_memorized'].astype(bool)
        
        # Split results by memorization status for best config
        memorized_disagreements = best_df[best_df['is_memorized'] == True]['knn_disagreement'].values
        non_memorized_disagreements = best_df[best_df['is_memorized'] == False]['knn_disagreement'].values
        
        # Create a clean boxplot with clear colors and force showing all boxplot elements
        boxplot = sns.boxplot(x='is_memorized', y='knn_disagreement', data=best_df, 
                            palette=['lightgreen', 'lightblue'], width=0.6,
                            showfliers=True, fliersize=3)
        
        # Add strip plot (individual points) to show raw data distribution
        sns.stripplot(x='is_memorized', y='knn_disagreement', data=best_df,
                     color='gray', alpha=0.3, jitter=True, size=3)
        
        # Calculate means for annotation
        memorized_mean = np.mean(best_df[best_df['is_memorized'] == True]['knn_disagreement'])
        non_memorized_mean = np.mean(best_df[best_df['is_memorized'] == False]['knn_disagreement'])
        
        # Draw mean markers (diamonds)
        plt.plot(0, non_memorized_mean, 'D', color='green', markersize=8)
        plt.plot(1, memorized_mean, 'D', color='blue', markersize=8)
        
        # Get number of nodes in each category
        n_memorized = len(best_df[best_df['is_memorized'] == True])
        n_non_memorized = len(best_df[best_df['is_memorized'] == False])
        
        # Update x-axis labels
        plt.xticks([0, 1], [f'Non-memorized\n(n={n_non_memorized})', f'Memorized\n(n={n_memorized})'])
        
        # Add clean horizontal grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Set labels and title
        plt.ylabel('Feature-Label Disagreement', fontsize=12)
        plt.xlabel('')  # Remove x-axis label
        # Remove the title
        
        # Get p-value from best analysis
        p_val = best_analysis['statistical_test']['p_value']
        
        # If significant difference, add a subtle indicator bar
        if p_val is not None and p_val < 0.05:
            # Add significance bracket
            x1, x2 = 0, 1
            y = max(np.max(memorized_disagreements), np.max(non_memorized_disagreements)) + 0.05
            plt.plot([x1, x1, x2, x2], [y, y+0.02, y+0.02, y], lw=1.5, c='black')
            
            # Add asterisks based on significance level
            if p_val < 0.001:
                sig_symbol = '***'
            elif p_val < 0.01:
                sig_symbol = '**'
            elif p_val < 0.05:
                sig_symbol = '*'
            else:
                sig_symbol = 'n.s.'
            
            plt.text((x1+x2)*.5, y+0.03, sig_symbol, ha='center', va='bottom', color='black')
        
        # Clean up the plot
        plt.tight_layout()
        
        # Save the plot for best configuration
        if save_dir and timestamp:
            # Create the save path with best config info
            save_name = f'knn_disagreement_best_{model_type}_{dataset_name}_{timestamp}.png'
            save_path = os.path.join(save_dir, save_name)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if logger:
                logger.info(f"Best configuration plot saved to: {save_path}")
        
        plt.close()
        
        # Create a scatter plot for best configuration
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot with different colors for memorized vs. non-memorized
        sns.scatterplot(x='mem_score', y='knn_disagreement', hue='is_memorized',
                       data=best_df, palette={True: 'blue', False: 'green'}, alpha=0.7)
        
        # Add vertical line at the threshold
        plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold = {threshold}')
        
        # Calculate and plot regression line
        if len(best_df) > 2:  # Need at least 3 points for regression
            x = best_df['mem_score'].values
            y = best_df['knn_disagreement'].values
            
            # Calculate correlation
            corr, p_corr = stats.pearsonr(x, y)
            
            # Plot regression line if correlation is significant
            if p_corr < 0.05:
                m, b = np.polyfit(x, y, 1)
                plt.plot(x, m*x + b, color='black', linestyle='-', alpha=0.5)
                plt.annotate(f'r = {corr:.3f} (p = {p_corr:.4f})',
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            ha='left', va='center',
                            bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Set plot labels and title
        plt.xlabel('Memorization Score')
        plt.ylabel('kNN Label Disagreement')
        scatter_title = f"Memorization Score vs. kNN Label Disagreement (k={best_k}, metric={best_metric})"
        if dataset_name and model_type:
            scatter_title += f"\nDataset: {dataset_name}, Model: {model_type}"
        plt.title(scatter_title)
        
        plt.legend(title='Is Memorized')
        
        # Save the scatter plot for best configuration
        if save_dir and timestamp:
            # Create the save path
            scatter_name = f'knn_disagreement_scatter_best_{model_type}_{dataset_name}_{timestamp}.png'
            scatter_path = os.path.join(save_dir, scatter_name)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            if logger:
                logger.info(f"Best configuration scatter plot saved to: {scatter_path}")
        
        plt.close()
    
    # Log summary
    if logger:
        logger.info("\n===== kNN Label Disagreement Analysis Summary =====")
        logger.info(f"Default configuration (k={k}, metric={similarity_metric}):")
        default_analysis = results['default']['analysis']
        
        mem_mean = default_analysis['memorized']['mean']
        non_mem_mean = default_analysis['non_memorized']['mean']
        p_val = default_analysis['statistical_test']['p_value']
        
        logger.info(f"- Memorized nodes (n={default_analysis['memorized']['count']}): mean={mem_mean:.4f}")
        logger.info(f"- Non-memorized nodes (n={default_analysis['non_memorized']['count']}): mean={non_mem_mean:.4f}")
        logger.info(f"- Difference: {mem_mean - non_mem_mean:.4f}")
        if p_val is not None:
            logger.info(f"- Statistical significance: p={p_val:.6f} ({'significant' if p_val < 0.05 else 'not significant'})")
        
        # Log best configuration
        if best_config['name'] != 'default':
            best_config_data = sensitivity_results[best_config['name']]['analysis']
            best_k = best_config_data['parameters']['k']
            best_metric = best_config_data['parameters']['similarity_metric']
            
            logger.info(f"\nBest configuration ({best_config['name']}, k={best_k}, metric={best_metric}):")
            
            best_mem_mean = best_config_data['memorized']['mean']
            best_non_mem_mean = best_config_data['non_memorized']['mean']
            best_p_val = best_config_data['statistical_test']['p_value']
            
            logger.info(f"- Memorized nodes (n={best_config_data['memorized']['count']}): mean={best_mem_mean:.4f}")
            logger.info(f"- Non-memorized nodes (n={best_config_data['non_memorized']['count']}): mean={best_non_mem_mean:.4f}")
            logger.info(f"- Difference: {best_mem_mean - best_non_mem_mean:.4f}")
            logger.info(f"- Statistical significance: p={best_p_val:.6f} ({'significant' if best_p_val < 0.05 else 'not significant'})")
    
    return results

#if __name__ == "__main__":
 #   pass
