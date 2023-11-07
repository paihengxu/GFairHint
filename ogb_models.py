import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        # x = torch.sum(x_i * x_j, dim=-1)
        x = torch.sigmoid(x)
        return x


class MultiLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MultiLinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = self.lins[-1](x)
        x = torch.sigmoid(x)
        return x


class NodePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(NodePredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        x = self.lins[-1](x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_batch_only_training=True):
        super(SAGE, self).__init__()
        self.use_batch_only_training = use_batch_only_training
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels, track_running_stats=use_batch_only_training))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):

        if len(self.convs) == 2:
            x = self.convs[0](x, adj_t)
            if not self.use_batch_only_training:
                x = self.bns[0](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.convs[1](x, adj_t)
            return x

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
        # return x.log_softmax(dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_batch_only_training=True):
        super(GCN, self).__init__()
        self.use_batch_only_training = use_batch_only_training
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels, track_running_stats=use_batch_only_training))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels, track_running_stats=use_batch_only_training))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        if len(self.convs) == 2:
            x = self.convs[0](x, adj_t)
            if not self.use_batch_only_training:
                x = self.bns[0](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.convs[1](x, adj_t)
            return x

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
        # return x.log_softmax(dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, heads=1, use_batch_only_training=True):
        super(GAT, self).__init__()
        self.use_batch_only_training = use_batch_only_training
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(heads * hidden_channels, track_running_stats=use_batch_only_training))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
            self.bns.append(torch.nn.BatchNorm1d(heads * hidden_channels, track_running_stats=use_batch_only_training))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        if len(self.convs) == 2:
            x = self.convs[0](x, adj_t)
            if not self.use_batch_only_training:
                x = self.bns[0](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.convs[1](x, adj_t)
            return x

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class FairGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, embedding_channels, num_layers,
                 dropout):
        super(FairGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels + embedding_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t, fair_node_embedding):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-2](torch.cat((x, fair_node_embedding), dim=1))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
        # return x.log_softmax(dim=-1)


class FairSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, embedding_channels, num_layers,
                 dropout):
        super(FairSAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels + embedding_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t, fair_node_embedding):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-2](torch.cat((x, fair_node_embedding), dim=1))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
        # return x.log_softmax(dim=-1)


class FairGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, embedding_channels, num_layers,
                 dropout, heads=1):
        super(FairGAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(heads * hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
            self.bns.append(torch.nn.BatchNorm1d(heads * hidden_channels))
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(heads * hidden_channels + embedding_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t, fair_node_embedding):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-2](torch.cat((x, fair_node_embedding), dim=1))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
        # return x.log_softmax(dim=-1)


class DummyFairGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, embedding_channels, num_layers,
                 dropout):
        super(DummyFairGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels + embedding_channels, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels + embedding_channels))

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels + embedding_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-2](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class DummyFairSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, embedding_channels, num_layers,
                 dropout):
        super(DummyFairSAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            SAGEConv(hidden_channels, hidden_channels + embedding_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels + embedding_channels))

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels + embedding_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-2](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class DummyFairGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, embedding_channels, num_layers,
                 dropout, heads=1):
        super(DummyFairGAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(heads * hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
            self.bns.append(torch.nn.BatchNorm1d(heads * hidden_channels))

        self.convs.append(
            GATConv(heads * hidden_channels, heads * hidden_channels, heads))
        self.tran_lin = torch.nn.Linear(heads * hidden_channels, heads * hidden_channels + embedding_channels)
        self.bns.append(torch.nn.BatchNorm1d(heads * hidden_channels + embedding_channels))

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(heads * hidden_channels + embedding_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            if i == len(self.convs) - 1:
                x = self.tran_lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-2](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
