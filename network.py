from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_states, out_actions, hidden_layers=2, hidden_nodes=64):
        super(Network, self).__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_states, hidden_nodes)])
        self.hidden_layers.extend([nn.Linear(hidden_nodes, hidden_nodes) for _ in range(hidden_layers - 1)])
        self.out = nn.Linear(hidden_nodes, out_actions)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))  # use ReLU activation function
        x = self.out(x)
        return x
