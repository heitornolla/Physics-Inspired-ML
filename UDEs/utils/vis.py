import matplotlib.pyplot as plt

def plot_results(t_eval, x_true, x_noisy, x_final):
    plt.figure(figsize=(10, 5))
    plt.plot(t_eval, x_true, 'g--', label='Ideal x(t)', alpha=0.7)
    plt.plot(t_eval, x_noisy, 'ro', label='Noisy Data', markersize=3, alpha=0.5)
    plt.plot(t_eval, x_final, 'b-', label='UDE Prediction')
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.title("UDE Result: Physical Laws + Learned Neural Damping")
    plt.grid(True)
    plt.show()
