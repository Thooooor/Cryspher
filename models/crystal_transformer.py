import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool


class CrystalTransformer(nn.Module):
    def __init__(
        self,
        args,
        init_atom_dim,
        atom_dim,
        init_edge_dim,
        edge_dim,
        hidden_dim, 
        num_convs,
        out_channels=1, 
        num_heads=4, 
        ):
        super(CrystalTransformer, self).__init__()

        self.atom_embedding = torch.nn.Linear(init_atom_dim, atom_dim)
        self.edge_embedding = torch.nn.Linear(init_edge_dim, edge_dim)
        
        # Define the transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(atom_dim, hidden_dim, heads=num_heads, edge_dim=edge_dim, concat=False) for i in range(num_convs)
        ])

        # Define the final output layer
        self.output_layer = nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.atom_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # Apply the transformer layers
        for transformer_layer in self.transformer_layers:
            x = F.relu(transformer_layer(x, edge_index, edge_attr))

        # Compute the output
        x = global_mean_pool(x, data.batch)
        x = self.output_layer(x)
        
        return x.squeeze()
