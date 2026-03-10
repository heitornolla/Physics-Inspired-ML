import torch
import torch.nn as nn
from torchdiffeq import odeint

class DampeningNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super(DampeningNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, u):
        return self.net(u)

class SpringMassUDE(nn.Module):
    def __init__(self, m, k, nn_model):
        super(SpringMassUDE, self).__init__()
        self.m = m
        self.k = k
        self.nn_model = nn_model

    def forward(self, t, y):
        u1, u2 = y[..., 0], y[..., 1]
        u_cat = torch.stack([u1, u2], dim=-1)
        
        # NN learns the damping force
        damp_force = self.nn_model(t, u_cat).squeeze(dim=-1)

        du1dt = u2
        du2dt = - (self.k / self.m) * u1 - (1.0 / self.m) * damp_force
        return torch.stack([du1dt, du2dt], dim=-1)

def forward_sim(ude_model, y0, t_points):
    y0_torch = y0.unsqueeze(0)
    sol = odeint(ude_model, y0_torch, t_points, method='rk4')
    return sol[:, 0, :]
