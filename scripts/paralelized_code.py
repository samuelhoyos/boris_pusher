import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.functions import update_v_relativistic, update_r


###################################
# Constants (in normalized units) #
###################################

e = -1.0  # Electron charge
me = 1.0  # Electron mass
c = 1.0  # Speed of light
B0 = 1  # Magnetic field strength
beta_p = 0.2  # Normalized shock speed (v_s/c)
a = 0.05  # Magnetic curvature coefficient

v_s = beta_p * c  # Shock speed
omega_ce = abs(e) * B0 / me  # Electron cyclotron frequency
k = omega_ce / c  # Wave number = inverse of the width of the shock front

num_particles = 32  # Number of test particles
final_time = 500

# Ranges for eta and zeta initialization
eta_ranges = [(-6, -4), (-6, -4), (-6, -4), (-6, -4), (4, 6), (4, 6), (4, 6), (4, 6)]
zeta_ranges = [(-7, -6), (6, 7), (-5, -4), (4, 5), (-7, -6), (6, 7), (-5, -4), (4, 5)]
particles_per_range = int(num_particles / len(eta_ranges))

# Random seed
seed = 363
np.random.seed(seed)

tolerance = 0.001  # Tolerance for the magnetic field


###########################################################
# Definition of the electric and magnetic field functions #
###########################################################

def electric_field(eta, zeta, t):
    eta_1 = k * (eta + v_s * t) - a * zeta**2
    eta_2 = k * (eta - v_s * t) + a * zeta**2
    Et_x = -(v_s * B0 / (2 * c)) * (np.tanh(eta_1) - np.tanh(eta_2) - 2)
    return np.array([Et_x, 0, 0])


def magnetic_field(eta, zeta, t):
    eta_1 = k * (eta + v_s * t) - a * zeta**2
    eta_2 = k * (eta - v_s * t) + a * zeta**2
    Bt_y = -(a * zeta * B0) * (np.tanh(eta_1) - np.tanh(eta_2) - 2)
    Bt_z = (B0 / 2) * (np.tanh(eta_1) + np.tanh(eta_2))
    return np.array([0, Bt_y, Bt_z])


########################
# Parallelized Routine #
########################

def simulate_particle(args):
    """Simulate the trajectory of a single particle."""
    r, v, particle_id = args
    new_trajectory, new_velocity, aux_time = [], [], []

    t = 0
    while t < final_time:
        B_field = magnetic_field(eta=r[1], zeta=r[2], t=t)
        if np.linalg.norm(B_field) > tolerance:
            dt = (0.005 * 0.1 * me) / (abs(e) * np.linalg.norm(B_field))

        v = update_v_relativistic(
            v=v,
            E=electric_field(eta=r[1], zeta=r[2], t=t),
            B=B_field,
            dt=dt,
        )

        r = update_r(v, r, dt)

        new_trajectory.append(r)
        new_velocity.append(v)
        aux_time.append(t)
        t += dt

    return particle_id, np.array(new_trajectory), np.array(new_velocity), np.array(aux_time)


def initialize_particles():
    """Initialize particle positions and velocities."""
    initial_positions_eta = np.concatenate([
        np.random.uniform(low, high, particles_per_range)
        for low, high in eta_ranges
    ])
    initial_positions_zeta = np.concatenate([
        np.random.uniform(low, high, particles_per_range)
        for low, high in zeta_ranges
    ])
    initial_positions_chi = np.zeros(num_particles)

    initial_positions = np.column_stack(
        (initial_positions_chi, initial_positions_eta, initial_positions_zeta)
    )
    initial_velocities = np.zeros((num_particles, 3))

    for i in range(num_particles):
        initial_velocities[i] = (
            np.cross(
                magnetic_field(eta=initial_positions[i, 1], zeta=initial_positions[i, 2], t=0),
                electric_field(eta=initial_positions[i, 1], zeta=initial_positions[i, 2], t=0),
            )
            / np.linalg.norm(
                magnetic_field(eta=initial_positions[i, 1], zeta=initial_positions[i, 2], t=0)
            )**2
        )
    return initial_positions, initial_velocities


##################
# Main Execution #
##################

if __name__ == "__main__":
    initial_positions, initial_velocities = initialize_particles()

    # Prepare inputs for parallel simulation
    particle_args = [
        (initial_positions[i], initial_velocities[i], i)
        for i in range(num_particles)
    ]

    # Run simulations in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_particle, particle_args), total=num_particles))

    # Collect results
    trajectories = [result[1] for result in sorted(results)]
    velocities = [result[2] for result in sorted(results)]
    time = [result[3] for result in sorted(results)]

    #############################
    # Plotting the trajectories #
    #############################

    eta_plot, zeta_plot = [], []
    for traj in trajectories:
        eta_plot.append(traj[:, 1])
        zeta_plot.append(traj[:, 2])

    # Eta - Zeta plot
    plt.figure(1)
    for i in range(num_particles):
        plt.plot(eta_plot[i], zeta_plot[i], color="red")
    plt.xlabel("Eta")
    plt.ylabel("Zeta")
    plt.title("Eta vs Zeta")
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)

    plt.show()
