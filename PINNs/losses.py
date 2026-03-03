import torch
from utils.physics_utils import G, V0, H0, derivative

def physics_loss(model, t):
    """Enforces the ODE: dh/dt = v0 - g*t"""
    t.requires_grad_(True)
    h_pred = model(t)
    dh_dt_pred = derivative(h_pred, t)
    dh_dt_true = V0 - G * t
    return torch.mean((dh_dt_pred - dh_dt_true)**2)

def initial_condition_loss(model):
    """Enforces h(0) = h0"""
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    h0_pred = model(t0)
    return torch.mean((h0_pred - H0)**2)

def data_loss(model, t_data, h_data):
    """Standard MSE loss against measurements"""
    h_pred = model(t_data)
    return torch.mean((h_pred - h_data)**2)
