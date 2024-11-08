import numpy as np
from collections.abc import Callable
from math import sqrt

B0 = 1
q = 4.803*1e-10
me = 9.1*1e-28
alpha = 0.05
c = 3*1e10
beta_p = 0.2
omega_c = q * B0 / (me * c)
eta_1_0 = 5
eta_2_0 = 5
zeta_0 = 5

def rkg_vector(
    func: Callable[[float, np.ndarray], np.ndarray],
    x_initial: float,
    state_initial: np.ndarray,
    step_size: float,
    x_final: float,
) -> np.ndarray:
    
    """
    Solve an Ordinary Differential Equations using Runge-Kutta-Gills Method of order 4.

    args:
    func: An ordinary differential equation (ODE) as function of x and y.
    x_initial: The initial value of x.
    y_initial: The initial value of y.
    step_size: The increment value of x.
    x_final: The final value of x.

    Returns:
        Solution of y at each nodal point.
    """

    if x_initial >= x_final:
        raise ValueError("The final value of x must be greater than the initial value of x.")
    
    if step_size <= 0:
        raise ValueError("Step size must be positive.")
    
    n = int((x_final - x_initial) / step_size)
    states = np.zeros((n + 1, len(state_initial)))
    states[0] = state_initial
    
    s = x_initial
    for i in range(n):
        k1 = step_size * func(s, states[i])
        k2 = step_size * func(s + step_size / 2, states[i] + k1 / 2)
        k3 = step_size * func(
            s + step_size / 2,
            states[i] + (-0.5 + 1 / sqrt(2)) * k1 + (1 - 1 / sqrt(2)) * k2
        )
        k4 = step_size * func(
            s + step_size, states[i] - (1 / sqrt(2)) * k2 + (1 + 1 / sqrt(2)) * k3
        )
        
        states[i + 1] = states[i] + (k1 + (2 - sqrt(2)) * k2 + (2 + sqrt(2)) * k3 + k4) / 6
        s += step_size
    
    return states

def particle_dynamics(beta_x, beta_y, beta_z, zeta, t, state):
    beta_x, beta_y, beta_z, zeta = state

    phi = 0
    beta = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
    gamma = 1/np.sqrt(1-beta**2)
    eta_1 = beta_p * t - alpha * zeta**2 + phi
    eta_2 = -eta_1

    # Compute magnetic and electric field components
    B_y1 = (alpha * zeta) * (np.tanh(eta_1) - 1)
    B_y2 = -(alpha * zeta) * (np.tanh(eta_2) + 1)

    B_z1 = (np.tanh(eta_1) - 1)
    B_z2 = (np.tanh(eta_2) + 1)

    E_x1 = -beta_p / 2 * (np.tanh(eta_1) - 1)
    E_x2 = beta_p / 2 * (np.tanh(eta_2) + 1)
    
    # Total fields (normalized)
    Ex_t = (E_x1 + E_x2) / B0
    By_t = (B_y1 + B_y2) / B0
    Bz_t = (B_z1 + B_z2) / B0
    
    # Equations of motion
    d_beta_x = 1/gamma * (Ex_t + (1 - beta_x**2) + (beta_y* Bz_t) - (beta_z * By_t))
    d_beta_y = -beta_x / gamma * (Bz_t + Ex_t*beta_y) 
    d_beta_z = -beta_x / gamma * (By_t + Ex_t*beta_z)
    d_zeta = omega_c**2 * beta_z
    
    return np.array([d_beta_x, d_beta_y, d_beta_z, d_zeta])


# Parameters for the dynamics
beta_x0 = 0
beta_y0 = -beta_p/4 * (np.tanh(eta_1_0) + np.tanh(eta_2_0) * (np.tanh(eta_1_0 -eta_2_0 - 2)))
beta_z0 = -beta_p * alpha * zeta_0/2 * (np.tanh(eta_1_0) - np.tanh(eta_2_0) - 2)
eta_1 = 0.5  # Adjust based on the actual model setup
eta_2 = 0.5  # Adjust based on the actual model setup
zeta = 0.5  # Adjust based on the actual model setup

# Initial conditions
initial_state = np.array([beta_x0, beta_y0, beta_z0, zeta_0])  # Adjust based on your initial conditions



# Run the solver with a lambda function to pass additional parameters
results = rkg_vector(
    func=lambda t, state: particle_dynamics(beta_x0, beta_y0, beta_z0, zeta_0, t, state),
    x_initial=0,
    state_initial=initial_state,
    step_size=0.01,
    x_final=10
)

# Display the results
print(results)