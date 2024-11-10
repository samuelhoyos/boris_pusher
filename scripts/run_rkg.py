import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from modules import rkg_modified
import matplotlib.pyplot as plt



N=50
q = 3e9
me = 9.11e-28
alpha = 0.05
c = 3e8
beta_p = 0.2
# eta_1_0 = np.random.uniform(-10, 10,size=N)
# eta_2_0 = np.random.uniform(-10, 10,size=N)
# zeta_0 = np.random.uniform(-10, 10,size=N)

tau_initial=0
tau_final=1000
step_size=0.001
# beta_x0 = np.zeros(N)
# beta_y0 = -beta_p/4 * (np.tanh(eta_1_0) + np.tanh(eta_2_0) * (np.tanh(eta_1_0 -eta_2_0 - 2)))

# beta_z0 = -beta_p * alpha * zeta_0/2 * (np.tanh(eta_1_0) - np.tanh(eta_2_0) - 2)

# Initial conditions
initial_state=[]

results = []
eta_total_array = []
zeta_total_array = []
if __name__ == "__main__":
    for i in range(N):
        np.random.seed(i)  # Use the loop index or another varying number as the seed
        right_or_left = False

        # Generate initial conditions for each iteration
        eta_1_0 = np.random.uniform(-10, 10)
        eta_2_0 = np.random.uniform(-10, 10)
        zeta_0 = np.random.uniform(-10, 10)
        
        beta_x0 = 0
        beta_y0 = -beta_p / 4 * (np.tanh(eta_1_0) + np.tanh(eta_2_0) * (np.tanh(eta_1_0 - eta_2_0 - 2)))
        beta_z0 = -beta_p * alpha * zeta_0 / 2 * (np.tanh(eta_1_0) - np.tanh(eta_2_0) - 2)

        initial_state = np.array([beta_x0, beta_y0, beta_z0, zeta_0, eta_1_0, eta_2_0, zeta_0])

        try:            
            results.append(rkg_modified.rkg_vector(
                func=rkg_modified.particle_dynamics,
                x_initial=tau_initial,
                state_initial=initial_state,
                step_size=step_size,
                x_final=tau_final
            ))

            eta_array = np.array([]) # The arrays are emptied after each iteration
            zeta_array = np.array([])

            if eta_1_0 > eta_2_0: # If the particle was going to the forward direction
                eta_array = results[-1][0][:,4]
            else: # If the particle was going to the backward direction
                eta_array = + results[-1][0][:,5]

            zeta_array = results[-1][0][:,6]            
            eta_total_array.append(eta_array)
            zeta_total_array.append(zeta_array)

        except ValueError as e:
            print(f"Error encountered: {e}")
    
    for i in range(len(eta_total_array)):
        plt.plot(eta_total_array[i], zeta_total_array[i])

    plt.xlabel("Eta")
    plt.ylabel("Zeta")
    plt.show()

