from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.typing import Adj, OptTensor, PairTensor

class CGConv(MessagePassing):
    """The crystal graph convolutional operator"""
    def __init__(self, atom_dim, edge_dim,
                 aggr: str = 'add', batch_norm: bool = True,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm

        self.lin_f = Linear(2 * atom_dim + edge_dim, atom_dim, bias=bias)
        self.lin_s = Linear(2 * atom_dim + edge_dim, atom_dim, bias=bias)
        if batch_norm:
            self.bn = BatchNorm1d(atom_dim)
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.bn is None else self.bn(out)
        out = out + x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'


class HiddenLayer(Module):
    def __init__(self, in_features, out_features, activation):
        super(HiddenLayer, self).__init__()
        self.linear = Linear(in_features, out_features)
        self.activation = activation
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    

class CGCNN(Module):
    def __init__(self, args):
        super(CGCNN, self).__init__()
        self.embedding_dim = args.dim
        self.num_layers = args.layers
        self.atom_embedding = torch.nn.Linear(args.init_atom_dim, self.embedding_dim)
        self.edge_embedding = torch.nn.Linear(args.init_edge_dim, self.embedding_dim)
        self.convs = torch.nn.ModuleList([CGConv(self.embedding_dim, self.embedding_dim) for _ in range(self.num_layers)])
        self.fc_hidden = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.hidden_layers = torch.nn.ModuleList([HiddenLayer(self.embedding_dim, self.embedding_dim, torch.nn.ReLU()) for _ in range(self.num_layers)])
        self.fc_out = torch.nn.Linear(self.embedding_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.atom_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = global_mean_pool(x, data.batch)
        x = self.fc_hidden(x)
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.fc_out(x)

        return out.squeeze()
