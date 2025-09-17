# GFairHint
Implementation for `Paiheng Xu, Yuhang Zhou, Bang An, Wei Ai, and Furong Huang. 2025. GFairHint: Improving Individual Fairness for Graph Neural Networks via Fairness Hint. ACM Trans. Knowl. Discov. Data 19, 3, Article 67 (April 2025), 22 pages. https://doi.org/10.1145/3714472`
## Install
```
conda create -n fair_gnn python=3.7

conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
pip install ogb
pip install -r requirements.txt
```

We follow the instructions to install [Open Graph Benchmark (OGB)](https://github.com/snap-stanford/ogb) package.
Specifically, [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [Deep Graph Library (DGL)](https://www.dgl.ai/).


## Run

### Step 1: Data

#### Academic networks 

1. For ACM, coauthor-phy, and coauthor-cs, we could obtain from the REDRESS GitHub repo: (https://github.com/yushundong/REDRESS/tree/main/node%20classification/data) and copy to the `data` folder
2. For ogbn-arxiv, we could use the OGB command to download the dataset:
   ```
   dataset = PygNodePropPredDataset(name='ogbn-arxiv')
   ```

#### Crime network
1. Download [Crime dataset](https://archive.ics.uci.edu/dataset/183/communities+and+crime) to the `data` folder.
2. Follow instructions from [Lahoti et al.](https://dl.acm.org/doi/10.14778/3372716.3372723) to get ratings from [Niche.com](https://www.niche.com/). We cannot share the code and data for this part due to legal issues.
3. Run
```
cd data
python crime.py
```

### Step 2: Train Fairness Hint

```
python fairness_graph.py
```


### Step 3: Train and Evaluate on Original Graph and Task

For example,
```
python obg-dataset.py --model GCN --hidden 16 --num_layers 2 --graph_name acm
```
