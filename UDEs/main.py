import torch
import torch.optim as optim
from utils.generate_synthetic_data import generate_synthetic_data
from utils.model import DampeningNN, SpringMassUDE, forward_sim
from utils.vis import plot_results

def train():
    m, k = 1.0, 5.0
    t_eval, x_true, x_noisy, v_noisy = generate_synthetic_data(m, k)
    
    t_train = torch.tensor(t_eval, dtype=torch.float32)
    x_train = torch.tensor(x_noisy, dtype=torch.float32)
    v_train = torch.tensor(v_noisy, dtype=torch.float32)
    y0_train = torch.tensor([x_noisy[0], v_noisy[0]], dtype=torch.float32)

    nn_model = DampeningNN(input_dim=2, hidden_dim=16)
    ude = SpringMassUDE(m=m, k=k, nn_model=nn_model)
    optimizer = optim.Adam(ude.parameters(), lr=1e-2)

    for epoch in range(1, 101):
        optimizer.zero_grad()
        sol_pred = forward_sim(ude, y0_train, t_train)
        
        loss = 0.5 * torch.mean((sol_pred[:, 0] - x_train)**2) + \
               0.5 * torch.mean((sol_pred[:, 1] - v_train)**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    with torch.no_grad():
        final_pred = forward_sim(ude, y0_train, t_train).numpy()
    
    plot_results(t_eval, x_true, x_noisy, final_pred[:, 0])

if __name__ == "__main__":
    train()