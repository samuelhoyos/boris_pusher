import numpy as np
from collections.abc import Callable
from math import sqrt


me = 9.11e-28
alpha = 0.05
c = 3e8
beta_p = 0.2


def rkg_vector(
    func: Callable[[float, np.ndarray,int], np.ndarray],  # The particle dynamics function
    x_initial: float, # Initial time for the simulation
    state_initial: np.ndarray,
    step_size: float, # Step size for the simulation
    x_final: float, # Final time for the simulation
) -> np.ndarray:
    
    if x_initial >= x_final:
        raise ValueError("The final value of x must be greater than the initial value of x.")
    
    if step_size <= 0:
        raise ValueError("Step size must be positive.")
    
    n = int((x_final - x_initial) / step_size) # Number of iterations
    states = np.zeros((n + 1, len(state_initial))) # For each iteration, states contains the values of the parameters at each timestep
    states[0] = state_initial # Initial state of the parameters (n rows)
    times = np.zeros(n + 1)  # Array to store the time (t) values
    times[0] = x_initial

    
    s = x_initial # "s" for state
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
            times[i + 1] = s
        except ValueError as e:
            print(e)
            return states[:i + 1], times[:i + 1]
    
    return states


def particle_dynamics(t, state, iteration=None):
    beta_x, beta_y, beta_z, zeta, eta_1_ghost, eta_2_shost, zeta_ghost = state
    beta = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
    phi=0
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
    d_beta_x = 1/gamma * (Ex_t + (1 - beta_x**2) + (beta_y * Bz_t) - (beta_z * By_t))
    # print( "E_x", Ex_t)
    # print( "B_z", Bz_t)
    # print( "B_y", By_t)
    # print( "d_beta_x", d_beta_x)
    d_beta_y = -beta_x / gamma * (Bz_t + Ex_t * beta_y) 
    d_beta_z = -beta_x / gamma * (By_t + Ex_t * beta_z)
    d_zeta = beta_z
    
    return np.array([d_beta_x, d_beta_y, d_beta_z, d_zeta, eta_1, eta_2, zeta])


# Parameters for the dynamic
# Display the results
#print(results)