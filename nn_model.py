import torch
import torch.nn as nn

class ParamNet(nn.Module):
    def __init__(self, input_size_par, hidden_size, output_size):
        super(ParamNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size_par, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # output_size can be > 1
        )

    def forward(self, x):
        return self.network(x)

class BasisNet(nn.Module):
    def __init__(self, input_size_basis, hidden_size, output_size):
        super(BasisNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size_basis, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)  # Match output_size with ParamNet
        )

    def forward(self, x):
        return self.network(x)

class NNPodNet(nn.Module):
    def __init__(self, input_size_par, input_size_basis, hidden_size, output_size):
        super(NNPodNet, self).__init__()
        self.param_net = ParamNet(input_size_par, hidden_size, output_size)
        self.basis_net = BasisNet(input_size_basis, hidden_size, output_size)

    def forward(self, param_input, basis_input):
        # Get outputs from both networks
        param_output = self.param_net(param_input)
        basis_output = self.basis_net(basis_input)

        # Element-wise multiplication of outputs and sum across all dimensions
        # to get a single output per sample
        combined_output = (param_output * basis_output).sum(dim=1, keepdim=True)
        return combined_output
