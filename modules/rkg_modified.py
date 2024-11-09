import numpy as np
from collections.abc import Callable
from math import sqrt

q = 3e9
me = 9.11e-28
alpha = 0.05
c = 3e8
beta_p = 0.2
eta_1_0 = 5
eta_2_0 = 5
zeta_0 = 5

def rkg_vector(
    func: Callable[[float, np.ndarray, int], np.ndarray],  # Updated function signature
    x_initial: float,
    state_initial: np.ndarray,
    step_size: float,
    x_final: float,
) -> np.ndarray:
    
    if x_initial >= x_final:
        raise ValueError("The final value of x must be greater than the initial value of x.")
    
    if step_size <= 0:
        raise ValueError("Step size must be positive.")
    
    n = int((x_final - x_initial) / step_size)
    states = np.zeros((n + 1, len(state_initial)))
    states[0] = state_initial
    
    s = x_initial
    for i in range(n):
        try:
            k1 = step_size * func(s, states[i],i)
            k2 = step_size * func(s + step_size / 2, states[i] + k1 / 2,i)
            k3 = step_size * func(
                s + step_size / 2,
                states[i] + (-0.5 + 1 / sqrt(2)) * k1 + (1 - 1 / sqrt(2)) * k2,i
            )
            k4 = step_size * func(
                s + step_size, states[i] - (1 / sqrt(2)) * k2 + (1 + 1 / sqrt(2)) * k3,i
            )
            states[i + 1] = states[i] + (k1 + (2 - sqrt(2)) * k2 + (2 + sqrt(2)) * k3 + k4) / 6
            s += step_size
        except ValueError as e:
            print(e)
            return states[:i + 1]
    
    return states


def particle_dynamics(t, state, iteration=None):
    beta_x, beta_y, beta_z, zeta = state
    phi = 0
    beta = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)

    if beta >= 1:
        print("Last valid state", state)
        raise ValueError(f"Error at iteration {iteration}: Beta squared is greater than or equal to 1, which makes gamma invalid.")



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
    Ex_t = (E_x1 + E_x2)
    By_t = (B_y1 + B_y2)
    Bz_t = (B_z1 + B_z2)
    
    # Equations of motion
    d_beta_x = 1/gamma * (Ex_t + (1 - beta_x**2) + (beta_y* Bz_t) - (beta_z * By_t))
    # print( "E_x", Ex_t)
    # print( "B_z", Bz_t)
    # print( "B_y", By_t)
    # print( "d_beta_x", d_beta_x)
    d_beta_y = -beta_x / gamma * (Bz_t + Ex_t*beta_y) 
    d_beta_z = -beta_x / gamma * (By_t + Ex_t*beta_z)
    d_zeta = beta_z
    
    return np.array([d_beta_x, d_beta_y, d_beta_z, d_zeta])


# Parameters for the dynamics
beta_x0 = 0
beta_y0 = -beta_p/4 * (np.tanh(eta_1_0) + np.tanh(eta_2_0) * (np.tanh(eta_1_0 -eta_2_0 - 2)))

beta_z0 = -beta_p * alpha * zeta_0/2 * (np.tanh(eta_1_0) - np.tanh(eta_2_0) - 2)

# Initial conditions
initial_state = np.array([beta_x0, beta_y0, beta_z0, zeta_0])  # Adjust based on your initial conditions


# Run the solver with a lambda function to pass additional parameters
results = rkg_vector(
    func=particle_dynamics,
    x_initial=0,
    state_initial=initial_state,
    step_size=0.01,
    x_final=10
)

# Display the results
#print(results)