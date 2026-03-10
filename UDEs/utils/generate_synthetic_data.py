import numpy as np
from scipy.integrate import solve_ivp

def generate_synthetic_data(m=1.0, k=5.0, c_true=0.7, noise_level=0.05):
    def damped_spring_mass(t, y):
        x, x_dot = y
        x_ddot = -(c_true/m)*x_dot - (k/m)*x
        return [x_dot, x_ddot]

    t_span = [0.0, 10.0]
    t_eval = np.linspace(t_span[0], t_span[1], 200)
    y0 = [1.0, 0.0] # [displacement, velocity]

    sol = solve_ivp(damped_spring_mass, t_span, y0, t_eval=t_eval)
    
    x_noisy = sol.y[0, :] + noise_level * np.random.randn(len(t_eval))
    v_noisy = sol.y[1, :] + noise_level * np.random.randn(len(t_eval))
    
    return t_eval, sol.y[0, :], x_noisy, v_noisy
