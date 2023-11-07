import argparse
import torch


def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--ranking_loss', type=bool, default=False,
                        help='Disables REDRESS loss.')
    parser.add_argument('--inform_loss', type=bool, default=False,
                        help='Disables Inform loss.')
    parser.add_argument('--add_edge', type=int, default=0,
                        help='add the edges of the fairness graph into the original graph.')
    parser.add_argument('--PFR', type=bool, default=False,
                        help='Disables PFR pre-processing.')
    parser.add_argument('--PFR_k', type=int, default=10, help='Dimension of PFR embeddings')
    parser.add_argument('--top_k', type=int, default=10, help='Top K in NDCG and ERR')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,  # 500
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,  # 0.01
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,  # 5e-6
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--num_layers', type=int, default=10,
                        help='Number of GNN layers.')
    parser.add_argument('--dropout', type=float, default=0.3,  # 0.3
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lambdas_para', type=float, default=1.0,
                        help='the parameter of lambdas')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["SAGE", "GCN", "GAT", "FairGCN", "FairSAGE", "FairGAT", "DummyFairGCN",
                                 "DummyFairSAGE", "DummyFairGAT"],
                        help='model to use.')
    parser.add_argument('--graph_name', type=str, default="arxiv",
                        choices=["arxiv", "acm", "coauthor-cs", "coauthor-phy", "crime"],
                        help='graph to run.')
    parser.add_argument('--simi_type', type=str, default="cosine",
                        choices=["cosine", "euclidean"],
                        help='similarity type.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        choices=['AugNormAdj'],
                        help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
