import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class StudentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, heads_1=8, heads_2=10, dropout=0.1):
        super().__init__()

        self.gatconv1 = GATConv(input_size, hidden_size, heads=heads_1, dropout=dropout)
        self.gatconv2 = GATConv(hidden_size * heads_1, hidden_size, heads=heads_1, dropout=dropout)
        self.gatconv3 = GATConv(hidden_size * heads_1, output_size, heads=heads_2, concat=False, dropout=dropout)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Projection layers for skip connections (to match dimensions)
        self.proj1 = nn.Linear(input_size, hidden_size * heads_1)
        self.proj2 = nn.Identity()
        self.proj3 = nn.Linear(hidden_size * heads_1, output_size)

    def forward(self, x, edge_index):
        # First GAT layer with skip connection
        identity1 = self.proj1(x)
        x = self.gatconv1(x, edge_index)
        x = self.elu(x) + identity1
        x = self.dropout(x)

        # Second GAT layer with skip connection
        identity2 = self.proj2(x)
        x = self.gatconv2(x, edge_index)
        x = self.elu(x) + identity2
        x = self.dropout(x)

        # Third GAT layer with skip connection, plus skip from input
        identity3 = self.proj3(x)
        x = self.gatconv3(x, edge_index)
        x = x + identity3

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, GATConv):
                for param in m.parameters():
                    if param.dim() > 1:  # Only apply Xavier to weight matrices
                        nn.init.xavier_normal_(param)