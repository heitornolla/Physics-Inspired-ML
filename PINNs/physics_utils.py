import numpy as np
import torch

# Physics parameters
G = 9.8        # acceleration due to gravity
H0 = 1.0       # initial height
V0 = 10.0      # initial velocity

def true_solution(t):
    """Analytical solution h(t) = h0 + v0*t - 0.5*g*t^2"""
    return H0 + V0 * t - 0.5 * G * (t**2)

def generate_data(t_min=0.0, t_max=2.0, n_data=10, noise_level=0.7, seed=0):
    """Generates noisy synthetic data for training."""
    np.random.seed(seed)
    t_data = np.linspace(t_min, t_max, n_data)
    h_exact = true_solution(t_data)
    h_noisy = h_exact + noise_level * np.random.randn(n_data)
    
    t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
    h_tensor = torch.tensor(h_noisy, dtype=torch.float32).view(-1, 1)
    return t_tensor, h_tensor

def derivative(y, x):
    """Computes dy/dx using PyTorch's autograd."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]
