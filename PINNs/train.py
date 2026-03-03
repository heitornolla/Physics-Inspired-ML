import torch
import matplotlib.pyplot as plt
from model import PINN
from physics_utils import generate_data, true_solution
import losses

def train():
    # Setup
    t_data, h_data = generate_data()
    model = PINN(n_hidden=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Weights
    l_data, l_ode, l_ic = 0.6, 0.3, 0.1
    epochs = 2000

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_d = losses.data_loss(model, t_data, h_data)
        loss_p = losses.physics_loss(model, t_data)
        loss_i = losses.initial_condition_loss(model)

        total_loss = l_data * loss_d + l_ode * loss_p + l_ic * loss_i
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss.item():.6f}")

    plot_results(model, t_data, h_data)

def plot_results(model, t_data, h_data):
    model.eval()
    t_plot = torch.linspace(0, 2, 100).view(-1, 1)
    h_pred = model(t_plot).detach().numpy()
    h_true = true_solution(t_plot.numpy())

    plt.figure(figsize=(8, 5))
    plt.scatter(t_data.detach().numpy(), h_data.detach().numpy(), color='red', label='Noisy Data')
    plt.plot(t_plot, h_true, 'k--', label='Exact Solution')
    plt.plot(t_plot, h_pred, 'b', label='PINN Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()