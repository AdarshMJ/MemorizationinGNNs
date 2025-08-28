# Memorization in Graph Neural Networks

This repository contains the **first framework to systematically study memorization in Graph Neural Networks (GNNs) for node classification tasks**. 


## 🎯 Overview
Deep neural networks (DNNs) have been shown to memorize their training data, yet similar analyses for graph neural networks (GNNs) remain largely under-explored. We introduce \ours (\oursLong), the first framework to quantify label memorization in semi-supervised node classification.
We first establish an inverse relationship between memorization and graph homophily, \ie the property that connected nodes share similar labels/features. We find that lower homophily significantly increases memorization, indicating that GNNs rely on memorization to learn less homophilic graphs. %
Secondly, we analyze GNN training dynamics. We find that the increased memorization in low homophily graphs is tightly coupled to the GNNs' implicit bias on using graph structure during learning. In low homophily regimes, this structure is less informative, hence inducing memorization of the node labels to minimize training loss.Finally, we show that nodes with higher label inconsistency in their feature-space neighborhood are significantly more prone to memorization. Building on our insights into the link between graph homophily and memorization, we investigate graph rewiring as a means to mitigate memorization. Our results demonstrate that this approach effectively reduces memorization without compromising model performance. Moreover, we show that it lowers the privacy risk for previously memorized data points in practice. Thus, our work not only advances understanding of GNN learning but also supports more privacy-preserving GNN deployment.

## 🏗️ Framework Architecture

### Core Components

```
├── RealWorld/          # Analysis on real-world graph datasets
├── SBM/               # Analysis on synthetic Stochastic Block Model graphs
├── nodeli.py          # Node Label Informativeness and Homophily calculations
└── README.md          # This file
```
## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torch-geometric
pip install deeprobust  # For synthetic dataset handling
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### Real-World Graphs

```Python
cd RealWorld/
python main_fixed.py --dataset Cora --model_type gcn --num_layers 3
```

### Synthetic Graph Analysis (SBM)

```Python
cd SBM/
python main_syncora.py --model_type gcn
```

### Supported Datasets

#### Real-World Datasets
- **Citation Networks**: Cora, Citeseer, Pubmed
- **Co-purchase Networks**: Computers, Photo
- **Social Networks**: Actor
- **Web Networks**: Cornell, Wisconsin, Texas
- **Wikipedia Networks**: Chameleon, Squirrel
- **Heterophilous Graphs**: Roman-empire, Amazon-ratings

#### Synthetic Datasets
- **Stochastic Block Models**: Various homophily levels (h0.00 to h1.00)


## 📝 Citation

If you use this framework in your research, please cite our work:

```bibtex
@misc{jamadandi2025memorizationgraphneuralnetworks,
      title={Memorization in Graph Neural Networks}, 
      author={Adarsh Jamadandi and Jing Xu and Adam Dziedzic and Franziska Boenisch},
      year={2025},
      eprint={2508.19352},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.19352}, 
}
```
