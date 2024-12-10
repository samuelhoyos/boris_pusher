import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.rkg_modified import beta_p, alpha, rkg_vector, particle_dynamics
import numpy as np
import matplotlib.pyplot as plt



# Adjust initial conditions
N = 50
x_initial = 0
x_final = 100
step_size = 0.001

results = []
for i in range(N):
    print(f"Processing iteration {i + 1} of {N}", end='\r')  # Overwrites the same line

    eta_1_0 = np.random.uniform(-10, 10)
    eta_2_0 = np.random.uniform(-10, 10)
    zeta_0 = np.random.uniform(-10, 10)

    beta_x0 = 0
    beta_y0 = -beta_p / 4 * (np.tanh(eta_1_0) + np.tanh(eta_2_0))
    beta_z0 = -beta_p * alpha * zeta_0 / 2 * (np.tanh(eta_1_0) - np.tanh(eta_2_0))

    initial_state = np.array([beta_x0, beta_y0, beta_z0, zeta_0, eta_1_0, eta_2_0, zeta_0])

    states, _ = rkg_vector(particle_dynamics, x_initial, initial_state, step_size, x_final)
    results.append(states)


for result in results:
    eta = result[:, 4]
    zeta = result[:, 3]
    plt.plot(eta, zeta)

plt.xlabel("Eta")
plt.ylabel("Zeta")
plt.title("Particle Trajectories")
plt.show()

