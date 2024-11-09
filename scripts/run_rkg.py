from modules import functions, rkg_modified
import numpy as np

q = 3e9
me = 9.11e-28
alpha = 0.05
c = 3e8
beta_p = 0.2
eta_1_0 = 5
eta_2_0 = 5
zeta_0 = 5

tau_initial=0
tau_final=10
step_size=0.01
beta_x0 = 0
beta_y0 = -beta_p/4 * (np.tanh(eta_1_0) + np.tanh(eta_2_0) * (np.tanh(eta_1_0 -eta_2_0 - 2)))

beta_z0 = -beta_p * alpha * zeta_0/2 * (np.tanh(eta_1_0) - np.tanh(eta_2_0) - 2)

# Initial conditions
initial_state = np.array([beta_x0, beta_y0, beta_z0, zeta_0])  

results = []
if __name__ == "__main__":
    try:
        results = rkg_modified.rkg_vector(
            func=rkg_modified.particle_dynamics,
            x_initial=tau_initial,
            state_initial=initial_state,
            step_size=step_size,
            x_final=tau_final
        )
    except ValueError as e:
        print(f"Error encountered: {e}")

