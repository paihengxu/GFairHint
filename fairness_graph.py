from ogb.nodeproppred import PygNodePropPredDataset
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from ogb_models import GCN, LinkPredictor
from args import get_citation_args
import os
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from data_loader.feature_utils import load_data2, load_npz
from data.crime import CrimeDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Arguments
args = get_citation_args()
torch.manual_seed(12345)


def train(model, predictor, data, split_edge, optimizer, batch_size, adj_t):
    model.train()
    predictor.train()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        h = model(data.x, adj_t)
        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # Just do some trivial random sampling.
        neg_edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                                 device=h.device)
        neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    # print("neg loss: ")
    # print(torch.sigmoid(torch.sum(h[neg_edge[0]] * h[neg_edge[1]], dim=-1)).mean())
    # print(neg_loss.item())
    # print("pos loss: ")
    # print(torch.sigmoid(torch.sum(h[edge[0]] * h[edge[1]], dim=-1)).mean())
    # print(pos_loss.item())
    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, batch_size, adj_t):
    model.eval()
    predictor.eval()
    h = model(data.x, adj_t)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    results = {}
    for K in [10, 50, 100]:
        test_hits = eval_hits(K, pos_test_pred, neg_test_pred)
        results[f'Hits@{K}'] = test_hits
    return results


def eval_hits(K, y_pred_pos, y_pred_neg):
    """
        compute Hits@K
        For each positive target node, the negative target nodes are the same.
        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    """
    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}
    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return hitsK


def simi_distribution(graph_name):
    if graph_name == "arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]
        node_features = data.x
        print(data.x.size())
    elif graph_name == "acm":
        _, node_features, labels, idx_train, idx_val, idx_test, _ = load_data2("ACM")
        print(node_features.size())
    else:
        return
    # show the similarity distribution
    sample_number = 3000
    if sample_number < node_features.size(0):
        perm = torch.randperm(node_features.size(0))
        idx = perm[: sample_number]
        node_features = node_features[idx]
    similarity = []
    # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for i in range(node_features.size()[0]):
        for j in range(node_features.size()[0]):
            if j == i:
                continue
            output = torch.cosine_similarity(node_features[i].view(1, -1),
                                             node_features[j].view(1, -1)).item()
            similarity.append(output)
        if i % 100 == 0:
            print(i)
    # for i in range(node_features.size()[0]):
    #     if i % 100 == 0:
    #         print(i)
    #     current_node_feature = node_features[i]
    #     current_simi = pw_cosine_distance(current_node_feature.view(1, -1), node_features)
    #     current_simi = list(current_simi.view(-1))
    #     del current_simi[i]
    #     similarity += current_simi
    similarity = np.array(similarity)
    print(len(similarity))
    print(similarity)
    # large_number, small_number = 0, 0
    # for s in similarity:
    #     if s > 0.93:
    #         large_number += 1
    #     else:
    #         small_number += 1
    # print(large_number)
    # print(small_number)
    similarity_sorted = sorted(similarity)
    print(similarity_sorted[-int(50 * node_features.size()[0])])
    bins = np.linspace(np.min(similarity),
                       np.max(similarity),
                       20)
    print(bins)
    plt.xlim([np.min(similarity), np.max(similarity)])
    plt.hist(similarity, bins=bins, alpha=0.5)
    plt.title('Similarity distribution of ' + graph_name + ' data')
    plt.xlabel('cosine similarity')
    plt.ylabel('count')
    plt.savefig(graph_name + "_similarity_distribution.png")
    plt.show()


def pw_cosine_distance(input_a, input_b):
    normalized_input_a = torch.nn.functional.normalize(input_a)
    normalized_input_b = torch.nn.functional.normalize(input_b)
    res = torch.mm(normalized_input_a, normalized_input_b.T)
    return res


def split_edge_data(pos_edge_index, neg_edge_index):
    split_edge = {"train": {}, "test": {}, "valid": {}}
    train_edge_number = pos_edge_index.size()[0] - pos_edge_index.size()[0] // 40 - pos_edge_index.size()[0] // 20
    test_edge_number = pos_edge_index.size()[0] // 40
    valid_edge_number = pos_edge_index.size()[0] // 20
    valid_neg_number, test_neg_number = valid_edge_number, test_edge_number
    perm = torch.randperm(pos_edge_index.size()[0])
    idx = perm[: train_edge_number]
    split_edge['train']['edge'] = pos_edge_index[idx]
    idx = perm[train_edge_number: train_edge_number + valid_edge_number]
    split_edge['valid']['edge'] = pos_edge_index[idx]
    idx = perm[train_edge_number + valid_edge_number:]
    split_edge['test']['edge'] = pos_edge_index[idx]
    perm = torch.randperm(neg_edge_index.size()[0])
    idx = perm[: valid_neg_number]
    split_edge['valid']['edge_neg'] = neg_edge_index[idx]
    idx = perm[valid_neg_number: valid_neg_number + test_neg_number]
    split_edge['test']['edge_neg'] = neg_edge_index[idx]
    return split_edge


def fairness_graph_construction(graph_name, top_k=10):
    positive_sample_number = top_k
    print(positive_sample_number)
    if graph_name == "arxiv":
        node_feature_file = 'node_features.pt'
        pos_edge_index_file = 'edge_index_' + str(positive_sample_number) + '.pt'
        neg_edge_index_file = 'edge_neg_' + str(positive_sample_number) + '.pt'
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]
        node_features = data.x
        # simi_threshold = 0.93
        print(node_features.size())
    elif graph_name == 'crime':
        node_feature_file = graph_name + '_node_features.pt'
        pos_edge_index_file = graph_name + '_edge_index.pt'
        dataset = CrimeDataset(data_dir='data')
        graph, label = dataset[0]
        node_features = torch.tensor(graph['node_feat'], dtype=torch.float32)
        torch.save(node_features, 'data/arxiv_fairness_graph/' + node_feature_file)
        torch.save(torch.tensor(graph['fair_edge_index'].T), 'data/arxiv_fairness_graph/' + pos_edge_index_file)
        return
    elif graph_name == "acm" or graph_name == "coauthor-cs" or graph_name == "coauthor-phy":
        node_feature_file = graph_name + '_node_features.pt'
        pos_edge_index_file = graph_name + '_edge_index_' + str(positive_sample_number) + '.pt'
        neg_edge_index_file = graph_name + '_edge_neg_' + str(positive_sample_number) + '.pt'
        if graph_name == "acm":
            _, node_features, labels, idx_train, idx_val, idx_test, _ = load_data2("ACM")
            # simi_threshold = 0.2193
        else:
            _, node_features, labels, idx_train, idx_val, idx_test, _ = load_npz(graph_name)
        print(node_features.size())
    else:
        print("No Valid Dataset")
        return
    # show the similarity distribution
    # sample_number = 10000
    # perm = torch.randperm(data.x.size(0))
    # idx = perm[: sample_number]
    # node_features = data.x[idx]
    pos_edges = torch.tensor([[0, 0]])
    neg_edges = torch.tensor([[0, 0]])
    torch.save(node_features, 'data/arxiv_fairness_graph/' + node_feature_file)
    for i in range(node_features.size()[0]):
        current_node_feature = node_features[i]
        current_simi = pw_cosine_distance(current_node_feature.view(1, -1), node_features)
        current_simi = current_simi.view(-1)
        _, pos_edge = torch.topk(current_simi, positive_sample_number)
        pos_edge = pos_edge.view(-1, 1)
        current_index = torch.tensor([i]).repeat([pos_edge.size()[0], 1])
        pos_edge = torch.cat((current_index, pos_edge), 1)
        sample_number = 5
        _, neg_edge = torch.topk(input=current_simi, k=sample_number, largest=False)
        neg_edge = neg_edge.view(-1, 1)
        current_index = torch.tensor([i]).repeat([neg_edge.size()[0], 1])
        neg_edge = torch.cat((current_index, neg_edge), 1)
        if i == 0:
            pos_edges = pos_edge
            neg_edges = neg_edge
        else:
            pos_edges = torch.cat((pos_edges, pos_edge), 0)
            neg_edges = torch.cat((neg_edges, neg_edge), 0)
        if i % 1000 == 0:
            print("Current node features: " + str(i))
    print(pos_edges.size())
    print(neg_edges.size())
    # print(pos_edges)
    # print(neg_edges)
    torch.save(pos_edges, 'data/arxiv_fairness_graph/' + pos_edge_index_file)
    torch.save(neg_edges, 'data/arxiv_fairness_graph/' + neg_edge_index_file)


def collab_link_prediction():
    # simi_distribution()
    args.num_layers = 2
    args.epochs = 1
    args.runs = 1
    args.log_steps = 50
    args.batch_size = 64 * 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    device = torch.device(device)
    dataset = PygLinkPropPredDataset(name='ogbl-collab')
    data = dataset[0]
    data = data.to(device)
    print(data.edge_index)
    print(data.edge_index.size())
    split_edge = dataset.get_edge_split()
    adj_t = SparseTensor.from_edge_index(data.edge_index).t()
    # print(split_edge)
    # print(split_edge['train']['edge'].size())
    # print(split_edge['valid']['edge'].size())
    # print(split_edge['valid']['edge_neg'].size())
    # print(split_edge['test']['edge'].size())
    # print(split_edge['test']['edge_neg'].size())
    model = GCN(data.num_features, args.hidden,
                args.hidden, args.num_layers,
                args.dropout).to(device)
    predictor = LinkPredictor(args.hidden, args.hidden, 1,
                              args.num_layers, args.dropout).to(device)
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, adj_t)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            # if epoch % args.log_steps == 0:
            results = test(model, predictor, data, split_edge, args.batch_size, adj_t)
            for key, result in results.items():
                test_hits = result
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Test: {100 * test_hits:.2f}%')


def inference(graph_name, top_k=10):
    args.num_layers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if graph_name == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        model_path = 'saved_model/gcn_link_prediction_epochs_200_arxiv_' + str(top_k) + '.pth'
        data = dataset[0]
        data = data.to(device)
        pos_edge_index_file = 'edge_index_' + str(top_k) + '.pt'
        neg_edge_index_file = 'edge_neg_' + str(top_k) + '.pt'
        fair_node_embedding_file = 'fair_node_embedding_arxiv_' + str(top_k) + '.pt'
    elif graph_name == 'crime':
        dataset = CrimeDataset(data_dir='data')
        graph, labels = dataset[0]
        model_path = 'saved_model/gcn_link_prediction_epochs_200_' + graph_name + '.pth'
        pos_edge_index_file = graph_name + '_edge_index.pt'
        neg_edge_index_file = graph_name + '_edge_neg.pt'
        fair_node_embedding_file = 'fair_node_embedding_' + graph_name + '.pt'
        data = Data(x=torch.tensor(graph['node_feat'], dtype=torch.float32),
                    edge_index=torch.tensor(graph['edge_index']),
                    num_nodes=graph['num_nodes'], y=torch.tensor(labels).reshape(-1, 1))
        data = data.to(device)
    elif graph_name == "acm" or graph_name == "coauthor-cs" or graph_name == "coauthor-phy":
        if graph_name == 'acm':
            _, node_features, labels, idx_train, idx_val, idx_test, adj_original = load_data2("ACM")
        else:
            _, node_features, labels, idx_train, idx_val, idx_test, adj_original = load_npz(graph_name)
        edge_index, _ = from_scipy_sparse_matrix(adj_original)
        data = Data(x=node_features, edge_index=edge_index, num_nodes=node_features.size()[0],
                    y=labels.reshape(-1, 1))
        data = data.to(device)
        model_path = "saved_model/gcn_link_prediction_epochs_200_" + graph_name + ".pth"
        fair_node_embedding_file = 'fair_node_embedding_' + graph_name + '.pt'
        pos_edge_index_file = graph_name + '_edge_index.pt'
        neg_edge_index_file = graph_name + '_edge_neg.pt'
    else:
        return
    model = GCN(data.num_features, args.hidden,
                args.hidden, args.num_layers,
                args.dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    pos_edge_index = torch.load('data/arxiv_fairness_graph/' + pos_edge_index_file, map_location=device)
    neg_edge_index = torch.load('data/arxiv_fairness_graph/' + neg_edge_index_file, map_location=device)
    pos_edge_index = pos_edge_index[pos_edge_index[:, 0] != pos_edge_index[:, 1]]
    edge_index = torch.transpose(pos_edge_index, 0, 1)
    adj_t = SparseTensor(
        row=edge_index[0], col=edge_index[1],
        sparse_sizes=(data.x.size()[0], data.x.size()[0]),
        is_sorted=True).to_symmetric()
    h = model(data.x, adj_t)
    print(data.x.size())
    print(h)
    print(h.size())

    print(pos_edge_index)

    torch.save(h, 'data/arxiv_fairness_graph/' + fair_node_embedding_file)


def main(graph_name, top_k=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(graph_name)
    device = torch.device(device)
    if graph_name == "arxiv":
        node_feature_file = 'node_features.pt'
        pos_edge_index_file = 'edge_index_' + str(top_k) + '.pt'
        neg_edge_index_file = 'edge_neg_' + str(top_k) + '.pt'
        saved_model_name = 'gcn_link_prediction_epochs_200_arxiv_' + str(top_k) + '.pth'
    elif graph_name in ["acm", "coauthor-cs", "coauthor-phy", "crime"]:
        node_feature_file = graph_name + '_node_features.pt'
        pos_edge_index_file = graph_name + '_edge_index.pt'
        neg_edge_index_file = graph_name + '_edge_neg.pt'
        saved_model_name = 'gcn_link_prediction_epochs_200_' + graph_name + '.pth'
    else:
        return
    data_x = torch.load('data/arxiv_fairness_graph/' + node_feature_file, map_location=device)
    pos_edge_index = torch.load('data/arxiv_fairness_graph/' + pos_edge_index_file, map_location=device)
    neg_edge_index = torch.load('data/arxiv_fairness_graph/' + neg_edge_index_file, map_location=device)
    # print(data_x)
    print(data_x.size())
    pos_edge_index = pos_edge_index[pos_edge_index[:, 0] != pos_edge_index[:, 1]]
    data = Data(x=data_x, edge_index=torch.transpose(pos_edge_index, 0, 1), num_nodes=data_x.size()[0])
    data = data.to(device)
    print(data.edge_index)
    print(data.edge_index.size())

    split_edge = split_edge_data(pos_edge_index, neg_edge_index)
    # print(split_edge)
    print("train edge size", split_edge["train"]["edge"].size())
    print("test edge size", split_edge["test"]["edge"].size())
    print("test edge size", split_edge["test"]["edge_neg"].size())

    print(data.edge_index)
    print(split_edge["train"]["edge"])
    adj_t = SparseTensor(
        row=split_edge["train"]["edge"].t()[0], col=split_edge["train"]["edge"].t()[1],
        sparse_sizes=(data.x.size()[0], data.x.size()[0]),
        is_sorted=True).to_symmetric()
    print(adj_t)
    model = GCN(data.num_features, args.hidden,
                args.hidden, args.num_layers,
                args.dropout).to(device)
    predictor = LinkPredictor(args.hidden, args.hidden, 1,
                              args.num_layers, args.dropout).to(device)
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, adj_t)
            if epoch % args.log_steps == 0:
                results = test(model, predictor, data, split_edge, args.batch_size, adj_t)
                for key, result in results.items():
                    test_hits = result
                    print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Test: {100 * test_hits:.2f}%')
    torch.save(model.state_dict(), "saved_model/" + saved_model_name)
    inference(graph_name, top_k=top_k)


def dis_cal(graph_name, top_k):
    if graph_name == 'arxiv':
        fair_node_embedding_file = 'fair_node_embedding_arxiv_' + str(top_k) + '.pt'
        pos_edge_index_file = 'edge_index_' + str(top_k) + '.pt'
        neg_edge_index_file = 'edge_neg_' + str(top_k) + '.pt'
    elif graph_name == 'crime':
        fair_node_embedding_file = 'fair_node_embedding_' + graph_name + '.pt'
        pos_edge_index_file = graph_name + '_edge_index.pt'
        neg_edge_index_file = graph_name + '_edge_neg.pt'
    elif graph_name == "acm" or graph_name == "coauthor-cs" or graph_name == "coauthor-phy":
        fair_node_embedding_file = 'fair_node_embedding_' + graph_name + '.pt'
        pos_edge_index_file = graph_name + '_edge_index.pt'
        neg_edge_index_file = graph_name + '_edge_neg.pt'
    else:
        return
    fair_node_embedding = torch.load(f'data/arxiv_fairness_graph/{fair_node_embedding_file}')

    link_dis = []
    nonlink_dis = []

    pos_edge_index = torch.load('data/arxiv_fairness_graph/' + pos_edge_index_file)
    neg_edge_index = torch.load('data/arxiv_fairness_graph/' + neg_edge_index_file)

    for i, j in pos_edge_index[: 10000]:
        dis_value = 1 - torch.cosine_similarity(fair_node_embedding[i].view(1, -1),
                                                fair_node_embedding[j].view(1, -1)).item()
        link_dis.append(dis_value)
    for i, j in neg_edge_index[: 10000]:
        dis_value = 1 - torch.cosine_similarity(fair_node_embedding[i].view(1, -1),
                                                fair_node_embedding[j].view(1, -1)).item()
        nonlink_dis.append(dis_value)

    print("Link distance: ")
    print(sum(link_dis) / len(link_dis))

    print("Non link distance: ")
    print(sum(nonlink_dis) / len(nonlink_dis))


if __name__ == "__main__":
    args.num_layers = 2
    args.epochs = 200
    args.runs = 1
    args.log_steps = 10
    args.batch_size = 64 * 1024
    args.hidden = 256

    # _, node_features, labels, idx_train, idx_val, idx_test, adj_original = load_data2("ACM")
    # edge_index, _ = from_scipy_sparse_matrix(adj_original)
    # data = Data(x=node_features, edge_index=edge_index, num_nodes=node_features.size()[0], y=labels)
    # split_idx = {'train': idx_train, 'valid': idx_val, 'test': idx_test}
    # print(labels.reshape(-1, 1))
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    # data = dataset[0]
    # split_idx = dataset.get_idx_split()
    # train_idx = split_idx['train']
    # print(data.y)
    # simi_distribution('acm')
    # simi_distribution('arxiv')
    # fairness_graph_construction('coauthor-cs')
    # main('coauthor-cs')
    # inference('coauthor-cs')
    # fairness_graph_construction('arxiv', top_k=top_k)

    top_k = args.top_k
    # main('arxiv', top_k=top_k)
    # dis_cal('arxiv', top_k=top_k)

    # fairness_graph_construction('acm')
    # fairness_graph_construction('arxiv')
    # fairness_graph_construction('acm')
    # fairness_graph_construction('arxiv')

    fairness_graph_construction('crime')
    # main('crime')
    # inference('crime')
