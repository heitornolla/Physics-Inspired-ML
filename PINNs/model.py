import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, n_hidden=20):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, t):
        return self.net(t)
