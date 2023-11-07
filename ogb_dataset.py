from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch

torch.cuda.empty_cache()
import numpy as np
import torch.nn.functional as F
from args import get_citation_args
from utils import Logger
from torch_sparse import SparseTensor
from torch_geometric.utils import from_scipy_sparse_matrix
from data_loader.feature_utils import load_data2, load_npz
from torch_geometric.data import Data
import os
from ogb_models import GCN, SAGE, GAT, FairGCN, FairSAGE, FairGAT, DummyFairGCN, DummyFairSAGE, DummyFairGAT, \
    LinkPredictor
from data.crime import CrimeDataset
import networkx as nx
import time
from scipy.sparse.csgraph import laplacian

startTime = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Arguments
args = get_citation_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)
print(device)


def add_edge(data, pos_edge_index_file):
    pos_edge_index = torch.load('data/arxiv_fairness_graph/' + pos_edge_index_file)
    pos_edge_index = pos_edge_index[pos_edge_index[:, 0] != pos_edge_index[:, 1]]
    original_edge = data.edge_index.transpose(0, 1)
    fair_edge = torch.cat((original_edge, pos_edge_index), 0)
    fair_edge = torch.unique(fair_edge, dim=0)
    print("Original Edge shape: ")
    print(original_edge.shape)
    print("Fair Edge shape: ")
    print(fair_edge.shape)
    data = Data(x=data.x,
                edge_index=torch.transpose(fair_edge, 0, 1),
                num_nodes=data.num_nodes, y=data.y)
    return data


def simi(output):
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a == 0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res


def euclidean_simi(output):
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a == 0, the_ones, a)
    a_norm = output / a
    dis = torch.cdist(a_norm, a_norm, p=2)
    res = 1 - dis / dis.max(dim=1)[0]

    return res


def avg_err(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    the_maxs, _ = torch.max(x_corresponding, 1)
    the_maxs = the_maxs.reshape(the_maxs.shape[0], 1).repeat(1, x_corresponding.shape[1])
    c = 2 * torch.ones_like(x_corresponding)
    x_corresponding = (c.pow(x_corresponding) - 1) / c.pow(the_maxs)
    the_ones = torch.ones_like(x_corresponding)
    new_x_corresponding = torch.cat((the_ones, 1 - x_corresponding), 1)

    for i in range(x_corresponding.shape[1] - 1):
        x_corresponding = torch.mul(x_corresponding, new_x_corresponding[:, -x_corresponding.shape[1] - 1 - i: -1 - i])
    the_range = torch.arange(0., x_corresponding.shape[1]).repeat(x_corresponding.shape[0], 1) + 1
    score_rank = (1 / the_range[:, 0:]) * x_corresponding[:, 0:]
    final = torch.mean(torch.sum(score_rank, axis=1))
    # print("Now Average ERR@k = ", final.item())

    return final.item()


def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(
        x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator.cuda() / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(
        x_sorted_scores.shape[0], 1)
    ndcg_list = torch.sum((numerator.cuda() / denominator.cuda()), 1) / idcg
    ndcg = torch.mean(ndcg_list)
    # print("Now Average NDCG@k = ", avg_ndcg.item())

    return ndcg.item()


def err_computation(score_rank):
    the_maxs = torch.max(score_rank).repeat(1, score_rank.shape[0])
    c = 2 * torch.ones_like(score_rank)
    score_rank = ((c.pow(score_rank) - 1) / c.pow(the_maxs))[0]
    the_ones = torch.ones_like(score_rank)
    new_score_rank = torch.cat((the_ones, 1 - score_rank))

    for i in range(score_rank.shape[0] - 1):
        score_rank = torch.mul(score_rank, new_score_rank[-score_rank.shape[0] - 1 - i: -1 - i])
    the_range = torch.arange(0., score_rank.shape[0]) + 1

    final = (1 / the_range[0:]) * score_rank[0:]

    return torch.sum(final)


def err_exchange_abs(x_corresponding, j, k, top_k):
    new_score_rank = x_corresponding
    err1 = err_computation(new_score_rank)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    err2 = err_computation(new_score_rank)

    return torch.abs(err1 - err2)


def lambdas_computation_only_review(x_similarity, y_similarity, top_k):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(
        x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(
        y_similarity[0, :])
    (x_sorted_scores, x_sorted_idxs) = x_similarity.cpu().sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.cpu().sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    k_para = 1
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1:(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1:(length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    return x_sorted_scores, y_sorted_idxs, x_corresponding


def lambdas_computation(x_similarity, y_similarity, top_k, sigma_1):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(
        x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(
        y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    # ***************************** pairwise delta ******************************
    sigma_tuned = sigma_1
    k_para = 1
    length_of_k = k_para * top_k
    y_sorted_scores = y_sorted_scores[:, 1:(length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1:(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1:(length_of_k + 1)]
    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        # print(i/y_sorted_scores.shape[0])
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        # print(i / x_corresponding.shape[0])
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************
    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_similarity.shape[0]):
        # if i >= 0.4 * y_similarity.shape[0]:
        #     break
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = err_exchange_abs(x_corresponding[i, :], j, k, top_k)
                    # print(the_delta)
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])

    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(without_zero[:, j, i])  # 本来是 -

    mid = torch.zeros_like(x_similarity)
    the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(
        length_of_k * x_similarity.shape[0], 1).squeeze()
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding


def train(model, data, train_idx, optimizer, model_name, adj_t, fair_node_embedding=None, flag=1):
    # flag is used when using redress method
    model.train()
    optimizer.zero_grad()

    if model_name == "GCN" or model_name == "SAGE" or model_name == "GAT" or model_name == "DummyFairGCN" or model_name == "DummyFairSAGE" or model_name == "DummyFairGAT":
        out1 = model(data.x, adj_t)
    elif model_name == "FairGCN" or model_name == "FairSAGE" or model_name == "FairGAT":
        out1 = model(data.x, adj_t, fair_node_embedding)
    else:
        return
    # out = F.log_softmax(out1[train_idx], dim=1)
    # loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])

    loss = F.cross_entropy(out1[train_idx], data.y.squeeze(1)[train_idx])
    # print('utility training loss: ')
    # print(loss)
    if flag == 0:
        loss.backward(retain_graph=True)
        return out1
    else:
        loss.backward()
        optimizer.step()
        return


def train_fair(model, data, train_idx, optimizer, out, sigma_1, lambdas_para, simi_type="cosine", top_k=10):
    model.train()

    if simi_type == "cosine":
        x_similarity = simi(data.x[train_idx])
        y_similarity1 = simi(out[train_idx])
    elif simi_type == "euclidean":
        x_similarity = euclidean_simi(data.x[train_idx])
        y_similarity1 = euclidean_simi(out[train_idx])
    else:
        return

    lambdas1, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(x_similarity, y_similarity1, top_k,
                                                                                    sigma_1)
    # print(y_similarity1.shape)
    assert lambdas1.shape == y_similarity1.shape

    # print("Ranking optimizing... ")
    y_similarity1.backward(lambdas_para * lambdas1)
    optimizer.step()


def train_inform(model, data, train_idx, optimizer, model_name, adj_t, dataset=None, fair_edge_index=None,
                 simi_type="cosine"):
    model.train()
    optimizer.zero_grad()

    if model_name == "GCN" or model_name == "SAGE" or model_name == "GAT":
        out1 = model(data.x, adj_t)
    else:
        return

    utility_loss = F.cross_entropy(out1[train_idx], data.y.squeeze(1)[train_idx])
    if dataset == 'crime':
        y_simi = simi(out1[sorted(train_idx)])
        G = nx.Graph()
        G.add_nodes_from(list(range(len(data.y))))
        # G.add_edges_from(data.edge_index.T.cpu().detach().numpy())
        G.add_edges_from(np.array(fair_edge_index).T)

        # H = G.subgraph(sorted(split_idx['test']))
        H = G.subgraph(sorted(train_idx.numpy()))
        # print("# nodes in test graph", H.number_of_nodes())
        # print("# edges in test graph", H.number_of_edges())
        # x_similarity = nx.to_numpy_array(G)
        x_simi = nx.to_numpy_array(H)
        x_simi = torch.from_numpy(x_simi)
    else:
        if simi_type == "cosine":
            x_simi = simi(data.x[train_idx])
            y_simi = simi(out1[train_idx])
        elif simi_type == "euclidean":
            x_simi = simi(data.x[train_idx])
            y_simi = simi(out1[train_idx])
        else:
            return
    y_simi_trans = torch.transpose(y_simi, 0, 1)
    L_s = torch.tensor(laplacian(x_simi.cpu().detach().numpy())).cuda()

    alpha = 1e-7
    inform_loss = alpha * torch.trace(torch.matmul(torch.matmul(y_simi_trans, L_s), y_simi)) / y_simi.size()[0]
    # print(utility_loss)
    # print(inform_loss)
    total_loss = utility_loss + inform_loss
    # print("Inform optimizing... ")
    total_loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model, data, split_idx, evaluator, model_name, adj_t, dataset, fair_node_embedding=None, fair_edge_index=None,
         simi_type="cosine", top_k=10):
    model.eval()
    if model_name == "GCN" or model_name == "SAGE" or model_name == "GAT" or model_name == "DummyFairGCN" or model_name == "DummyFairSAGE" or model_name == "DummyFairGAT":
        out = model(data.x, adj_t)
    elif model_name == "FairGCN" or model_name == "FairSAGE" or model_name == "FairGAT":
        out = model(data.x, adj_t, fair_node_embedding)
    else:
        return

    loss = F.cross_entropy(out[split_idx['train']], data.y.squeeze(1)[split_idx['train']])
    # print("utility test loss: ")
    # print(loss)

    sample_number = 10000
    if sample_number < len(split_idx['test']):
        perm = torch.randperm(len(split_idx['test']))
        idx = perm[: sample_number]
        fairness_test_idx = split_idx['test'][idx]
    else:
        fairness_test_idx = split_idx['test']

    if dataset == 'crime':
        y_similarity = simi(out[sorted(fairness_test_idx)])
        G = nx.Graph()
        G.add_nodes_from(list(range(len(data.y))))
        # G.add_edges_from(data.edge_index.T.cpu().detach().numpy())
        G.add_edges_from(np.array(fair_edge_index).T)

        # H = G.subgraph(sorted(split_idx['test']))
        H = G.subgraph(sorted(fairness_test_idx.numpy()))
        # print("# nodes in test graph", H.number_of_nodes())
        # print("# edges in test graph", H.number_of_edges())
        # x_similarity = nx.to_numpy_array(G)
        x_similarity = nx.to_numpy_array(H)
        x_similarity = torch.from_numpy(x_similarity).to(device)

    else:
        if simi_type == "cosine":
            y_similarity = simi(out[fairness_test_idx])
            x_similarity = simi(data.x[fairness_test_idx])
        elif simi_type == "euclidean":
            y_similarity = euclidean_simi(out[fairness_test_idx])
            x_similarity = euclidean_simi(data.x[fairness_test_idx])
        else:
            return

    x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity, top_k)

    err = avg_err(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)
    ndcg = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    y_pred_test = y_pred[sorted(fairness_test_idx)].cpu().detach().numpy()
    # y_pred = y_pred.cpu().detach().numpy()
    x_similarity = x_similarity.cpu().detach().numpy()

    assert len(fairness_test_idx) == x_similarity.shape[0]

    numerator = 0
    denominator = 0
    for i in range(len(fairness_test_idx)):
        for j in range(i + 1, len(fairness_test_idx)):
            denominator += x_similarity[i][j]
            indicator = 0 if y_pred_test[i] == y_pred_test[j] else 1
            numerator += indicator * x_similarity[i][j]
    consistency = 1 - numerator / denominator

    # code for testing consistency with true labels
    # numerator = 0
    # denominator = 0
    # label_test = data.y[sorted(fairness_test_idx)]
    # for i in range(len(fairness_test_idx)):
    #     for j in range(i + 1, len(fairness_test_idx)):
    #         denominator += x_similarity[i][j]
    #         indicator = 0 if label_test[i] == label_test[j] else 1
    #         numerator += indicator * x_similarity[i][j]
    # consistency_label = 1 - numerator / denominator
    # print(consistency_label)

    return train_acc, valid_acc, test_acc, err, ndcg, consistency, loss.item()


def model_init(model_name, data, num_classes, use_batch_only_training, fair_node_embedding):
    if model_name == "GCN":
        model = GCN(data.num_features, args.hidden,
                    num_classes, args.num_layers,
                    args.dropout, use_batch_only_training)
    elif model_name == "SAGE":
        model = SAGE(data.num_features, args.hidden,
                     num_classes, args.num_layers,
                     args.dropout, use_batch_only_training)
    elif model_name == "GAT":
        model = GAT(data.num_features, args.hidden,
                    num_classes, args.num_layers,
                    args.dropout, heads=1, use_batch_only_training=use_batch_only_training)
    elif model_name == "FairGCN":
        model = FairGCN(data.num_features, args.hidden, num_classes, fair_node_embedding.size()[1],
                        args.num_layers, args.dropout)
    elif model_name == "FairSAGE":
        model = FairSAGE(data.num_features, args.hidden, num_classes, fair_node_embedding.size()[1],
                         args.num_layers, args.dropout)
    elif model_name == "FairGAT":
        model = FairGAT(data.num_features, args.hidden, num_classes, fair_node_embedding.size()[1],
                        args.num_layers, args.dropout)
    elif model_name == "DummyFairGCN":
        model = DummyFairGCN(data.num_features, args.hidden, num_classes, fair_node_embedding.size()[1],
                             args.num_layers, args.dropout)
    elif model_name == "DummyFairSAGE":
        model = DummyFairSAGE(data.num_features, args.hidden, num_classes, fair_node_embedding.size()[1],
                              args.num_layers, args.dropout)
    elif model_name == "DummyFairGAT":
        model = DummyFairGAT(data.num_features, args.hidden, num_classes, fair_node_embedding.size()[1],
                             args.num_layers, args.dropout)
    else:
        print("Not valid model name")
        return
    model = model.to(device)
    return model


def main():
    if args.graph_name == 'crime':
        args.runs = 5
    else:
        args.runs = 2
    args.log_steps = 50

    # args.lr = 0.01
    # args.model = "FairSAGE"
    # args.hidden = 128
    # args.num_layers = 10
    # args.graph_name = "acm"
    # args.ranking_loss = 1

    ranking_loss = args.ranking_loss
    model_name = args.model
    lambdas_para = args.lambdas_para
    top_k = args.top_k
    use_batch_only_training = True

    print("model", args.model)
    print("hidden dim", args.hidden)
    print("num of layers", args.num_layers)
    print("graph name", args.graph_name)
    print("learning rate", args.lr)
    print("top_k", top_k)
    print("ranking loss")
    print(args.ranking_loss)

    # read dataset and corresponding fair node embedding
    if args.graph_name == "arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]

        if args.add_edge > 0:
            pos_edge_index_file = 'edge_index_' + str(top_k) + '.pt'
            data = add_edge(data, pos_edge_index_file)

        data = data.to(device)
        adj_t = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            sparse_sizes=(data.x.size()[0], data.x.size()[0]),
            is_sorted=True).to_symmetric()
        fair_node_embedding = torch.load('data/arxiv_fairness_graph/fair_node_embedding_arxiv_' + str(top_k) + '.pt',
                                         map_location=device)
        print(fair_node_embedding.size())
        fair_edge_index = None
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train'].to(device)
        print("Training dataset size: ")
        print(len(train_idx))
        num_classes = dataset.num_classes
        sigma_1 = 4e-3
        if ranking_loss:
            ranking_epoch = 150
            args.epochs = 250
        else:
            args.epochs = 300

    elif args.graph_name == 'crime':
        dataset = CrimeDataset(data_dir='data')
        graph, labels = dataset[0]
        if args.PFR:
            use_batch_only_training = False
            pfr_embed_fn = f'data/arxiv_fairness_graph/PFR_k{args.PFR_k}_node_embedding_crime.pt'
            print(f"Loading PFR embedding from {pfr_embed_fn}")
            assert model_name in ["GCN", "GAT", "SAGE"]
            assert not ranking_loss
            node_feat = graph['node_feat']
            data = Data(x=torch.tensor(node_feat, dtype=torch.float32),
                        edge_index=torch.tensor(graph['edge_index']),
                        num_nodes=graph['num_nodes'], y=torch.tensor(labels).reshape(-1, 1))
        else:
            data = Data(x=torch.tensor(graph['node_feat'], dtype=torch.float32),
                        edge_index=torch.tensor(graph['edge_index']),
                        num_nodes=graph['num_nodes'], y=torch.tensor(labels).reshape(-1, 1))

        if args.add_edge > 0:
            pos_edge_index_file = args.graph_name + '_edge_index.pt'
            data = add_edge(data, pos_edge_index_file)

        data = data.to(device)
        adj_t = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            sparse_sizes=(data.x.size()[0], data.x.size()[0]),
            is_sorted=True).to_symmetric()
        fair_node_embedding = torch.load('data/arxiv_fairness_graph/fair_node_embedding_crime.pt', map_location=device)
        print(fair_node_embedding.size())
        fair_edge_index = graph['fair_edge_index']
        split_idx = dataset.get_idx_split(rand=True)
        for k, v in split_idx.items():
            split_idx[k] = torch.tensor(v)
        train_idx = split_idx['train'].to(device)
        print("Training dataset size: ")
        print(len(train_idx))
        num_classes = len(set(labels))
        sigma_1 = 4e-3
        if ranking_loss:
            ranking_epoch = 300
            args.epochs = 250
        else:
            args.epochs = 500

    elif args.graph_name == "acm" or args.graph_name == "coauthor-cs" or args.graph_name == "coauthor-phy":
        if args.graph_name == "acm":
            _, node_features, labels, idx_train, idx_val, idx_test, adj_original = load_data2("ACM")
            if args.PFR:
                use_batch_only_training = False
                pfr_embed_fn = f'data/arxiv_fairness_graph/PFR_k{args.PFR_k}_node_embedding_acm.pt'
                print(f"Loading PFR embedding from {pfr_embed_fn}")
                node_features = torch.load(pfr_embed_fn)
                assert model_name in ["GCN", "GAT", "SAGE"]
                assert not ranking_loss
            sigma_1 = 1e-3
            if ranking_loss:
                ranking_epoch = 150
                args.epochs = 50
            else:
                args.epochs = 150
        elif args.graph_name == "coauthor-cs":
            _, node_features, labels, idx_train, idx_val, idx_test, adj_original = load_npz(args.graph_name)
            sigma_1 = 3e-3
            if ranking_loss:
                ranking_epoch = 500
                args.epochs = 50
            else:
                args.epochs = 300
        else:
            _, node_features, labels, idx_train, idx_val, idx_test, adj_original = load_npz(args.graph_name)
            sigma_1 = 4e-3
            if ranking_loss:
                ranking_epoch = 600
                args.epochs = 50
            else:
                args.epochs = 400
        edge_index, _ = from_scipy_sparse_matrix(adj_original)

        data = Data(x=node_features, edge_index=edge_index, num_nodes=node_features.size()[0],
                    y=labels.reshape(-1, 1))

        if args.add_edge > 0:
            pos_edge_index_file = args.graph_name + '_edge_index.pt'
            data = add_edge(data, pos_edge_index_file)

        data = data.to(device)
        split_idx = {'train': idx_train, 'valid': idx_val, 'test': idx_test}
        adj_t = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            sparse_sizes=(data.x.size()[0], data.x.size()[0]),
            is_sorted=True).to_symmetric()
        train_idx = split_idx['train'].to(device)
        num_classes = len(torch.unique(labels))
        print("Training dataset size: ")
        print(len(train_idx))
        fair_node_embedding = torch.load('data/arxiv_fairness_graph/fair_node_embedding_' + args.graph_name + '.pt',
                                         map_location=device)
        fair_edge_index = None
        # print(adj_original)
        # print(edge_index)
    else:
        return

    # if model_name == "GCN" or model_name == "SAGE":
    # print(data.edge_index)
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    # data = dataset[0]
    # data.adj_t = data.adj_t.to_symmetric()
    # data = data.to(device)
    # print(data.adj_t)

    # elif model_name == "SGC":
    #     args.dropedge_rate = 0.4
    #     data.edge_index, _ = dropout_adj(data.edge_index, p=args.dropedge_rate, num_nodes=data.num_nodes)
    #     data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    model = model_init(model_name, data, num_classes, use_batch_only_training, fair_node_embedding)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        if args.graph_name == 'crime':
            split_idx = dataset.get_idx_split(rand=True)
            for k, v in split_idx.items():
                split_idx[k] = torch.tensor(v)
            train_idx = split_idx['train'].to(device)
            print("Training dataset size: ")
            print(len(train_idx))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            train(model, data, train_idx, optimizer, model_name, adj_t, fair_node_embedding)
            if epoch % args.log_steps == 0:
                result = test(model, data, split_idx, evaluator, model_name, adj_t, args.graph_name,
                              fair_node_embedding, fair_edge_index, simi_type=args.simi_type, top_k=top_k)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc, err, ndcg, consistency, loss = result
                executionTime = (time.time() - startTime)

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%, '
                      f'Average ERR@k = {100 * err:.2f}%, '
                      f'Average NDCG@k = {100 * ndcg:.2f}%, '
                      f'Consistency: {consistency:.4f}, '
                      f'Execution Time: {executionTime:.4f}, ')

        if ranking_loss:
            for epoch in range(1, 1 + ranking_epoch):
                sample_number = 2000
                train_idx = split_idx['train'].to(device)
                if sample_number < len(train_idx):
                    # print(len(train_idx))
                    perm = torch.randperm(len(train_idx))
                    idx = perm[: sample_number]
                    train_idx = train_idx[idx]
                if args.inform_loss:
                    train_inform(model, data, train_idx, optimizer, model_name, adj_t, simi_type=args.simi_type)
                else:
                    output = train(model, data, train_idx, optimizer, model_name, adj_t, fair_node_embedding, flag=0)
                    train_fair(model, data, train_idx, optimizer, output, sigma_1, lambdas_para,
                               simi_type=args.simi_type, top_k=top_k)
                # print(epoch)
                if epoch % args.log_steps == 0:
                    result = test(model, data, split_idx, evaluator, model_name, adj_t, args.graph_name,
                                  fair_node_embedding, fair_edge_index, simi_type=args.simi_type, top_k=top_k)
                    logger.add_result(run, result)
                    train_acc, valid_acc, test_acc, err, ndcg, consistency, loss = result
                    executionTime = (time.time() - startTime)
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%, '
                          f'Average ERR@k = {100 * err:.2f}%, '
                          f'Average NDCG@k = {100 * ndcg:.2f}%, '
                          f'Consistency: {consistency:.4f}, '
                          f'Execution Time: {executionTime:.4f}, ')

        logger.print_statistics(run)
        if args.inform_loss:
            inform_loss = 1
            print("Inform Loss")
            torch.save(model, "saved_model/" + model_name + "_" + str(args.hidden) + "_" + str(
                args.num_layers) + "_" + args.graph_name + "_inform_loss_" + str(int(inform_loss)))
        else:
            torch.save(model, "saved_model/" + model_name + "_" + str(args.hidden) + "_" + str(
                args.num_layers) + "_" + args.graph_name + "_" + str(top_k) + "_" + str(int(ranking_loss)))
    logger.print_statistics()


if __name__ == "__main__":
    main()
