import numpy as np
from collections.abc import Callable
from math import sqrt


me = 9.11e-28
alpha = 0.05
c = 3e8
beta_p = 0.2


def rkg_vector(func, x_initial, state_initial, step_size, x_final):
    if x_initial >= x_final:
        raise ValueError("Final x must be greater than initial x.")
    if step_size <= 0:
        raise ValueError("Step size must be positive.")

    n = int((x_final - x_initial) / step_size)
    states = np.zeros((n + 1, len(state_initial)))
    states[0] = state_initial
    times = np.zeros(n + 1)
    times[0] = x_initial

    s = x_initial
    for i in range(n):
        print(
            f"Processing (small) iteration {i + 1} of {n}", end="\r"
        )  # Overwrites the same line

        try:
            k1 = step_size * func(s, states[i])
            k2 = step_size * func(s + step_size / 2, states[i] + k1 / 2)
            k3 = step_size * func(s + step_size / 2, states[i] - k1 / 2 + k2)
            k4 = step_size * func(s + step_size, states[i] - k2 + k3)

            states[i + 1] = states[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            s += step_size
            times[i + 1] = s

            # Halt if any variable grows too large
            if np.any(np.abs(states[i + 1]) > 1e6):
                raise ValueError("State values too large, stopping integration.")
        except ValueError as e:
            print(f"Integration stopped at step {i}: {e}")
            return states[: i + 1], times[: i + 1]

    return states, times


def particle_dynamics(t, state):
    # Extract variables
    beta_x, beta_y, beta_z, zeta = state[:4]

    beta = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)

    # Prevent overflow of beta
    if beta >= 1:
        beta = 0.999  # Clamp beta
        beta_x *= beta / np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
        beta_y *= beta / np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
        beta_z *= beta / np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)

    gamma = 1 / np.sqrt(1 - beta**2)

    # Field contributions
    eta_1 = beta_p * t - alpha * zeta**2
    eta_2 = -eta_1

    B_y = alpha * zeta * (np.tanh(eta_1) - np.tanh(eta_2))
    B_z = 0.5 * (np.tanh(eta_1) + np.tanh(eta_2))
    E_x = -beta_p / 2 * (np.tanh(eta_1) - np.tanh(eta_2))

    # Relativistic equations
    d_beta_x = gamma * (E_x + beta_y * B_z - beta_z * B_y)
    d_beta_y = -gamma * (beta_x * B_z)
    d_beta_z = -gamma * (beta_x * B_y)
    d_zeta = beta_z

    # Keep additional parameters constant
    return np.array([d_beta_x, d_beta_y, d_beta_z, d_zeta, 0, 0, 0])
