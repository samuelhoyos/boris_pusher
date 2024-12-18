import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

# The path has some problems with the folder
import sys
import os

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


# Derived quantities (also normalized)
v_s = beta_p * c  # Shock speed
omega_ce = abs(e) * B0 / (me * c)  # Electron cyclotron frequency
k = omega_ce / c  # Wave number = inverse of the width of the shock front

num_particles = 5  # Number of test particles

# Final time for the simulation
final_time = 300
omega_ce = abs(e) * B0 / me  # Electron cyclotron frequency
k = omega_ce / c  # Width of the shock front

# Ranges for g1, g2, t, and z (optional)
g0_min, g0_max = -10.0, 10.0
zeta0_min, zeta0_max = -10.0, 10.0
t_min, t_max = 0.0, final_time

# Tolerance for the magnetic field
tolerance = 0.001

seed = 363 # Random seed
np.random.seed(seed)

# Used for plotting
number_of_snapshots = 50

####################################
# Ranges for the initial positions #
####################################

# Type 1 range (arbitrarily chosen)
# eta_ranges = [(-6, -4), (-6, -4), (-6, -4), (-6, -4), (4, 6), (4, 6), (4, 6), (4, 6)]
# zeta_ranges = [(-7, -6), (6, 7), (-5, -4), (4, 5), (-7, -6), (6, 7), (-5, -4), (4, 5)]

# Type 2 range
#eta_ranges = [(-7, -8), (-7, -8), (7, 8), (7, 8)]
# zeta_ranges = [(-7, -8), (7, 8), (-7, -8), (7, 8)]

# Type 3 range
#eta_ranges = [(0.5, -0.5)]
#zeta_ranges = [(0.5, -0.5)]

# Type 4 range
# eta_ranges = [(-1, -2), (-1, -2), (1, 2), (1, 2)]
# zeta_ranges = [(8, 9), (-8, -9), (8, 9), (-8, -9)]

# Type 5 range
#eta_ranges = [(-10, 10)]
#zeta_ranges = [(-10, 10)]

# Type 6 range
eta_ranges = [(-7, -8)]
zeta_ranges = [(-7, -8)]


# Number of particles per subrange
particles_per_range = int(num_particles/len(eta_ranges))


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


######################
# Initial conditions #
######################

initial_positions_chi= np.zeros(num_particles)

# Eta and zeta are initialized with random values between certain ranges that can be modified
initial_positions_eta = np.concatenate([np.random.uniform(low, high, particles_per_range) 
                                       for low, high in eta_ranges])

initial_positions_zeta = np.concatenate([np.random.uniform(low, high, particles_per_range) 
                                        for low, high in zeta_ranges])

print(f"Initial positions eta: {initial_positions_eta}")
print(f"Initial positions zeta: {initial_positions_zeta}")

initial_positions = np.column_stack(
    (initial_positions_chi, initial_positions_eta, initial_positions_zeta)
)

print(f"Initial positions: {initial_positions}")

initial_velocities = np.zeros((num_particles, 3))


# This array says if a particle is going to the right or to the left (not used)
is_initial_velocity_positive = []

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
    if initial_positions[i, 1] >= 0:
        is_initial_velocity_positive.append(False)
    else:
        is_initial_velocity_positive.append(True)

print(f"Initial velocities: {initial_velocities}")



############################
# Trajectories calculation #
############################

trajectories = []
velocities = []
time = []

#for i in tqdm(range(num_particles)):
for i in range(num_particles):
    new_trajectory = []
    new_velocity = []
    aux_time = []

    r = initial_positions[i]
    v = initial_velocities[i]

    t = 0
    j = 0
    #with tqdm(total=final_time) as pbar:     
    while (t < final_time):

        if (np.linalg.norm(magnetic_field(eta=r[1], zeta=r[2], t=t)) > tolerance): 
            # To ensure stability concerning dt   
            dt = (0.0005 * 0.1 * me) / (abs(e) * np.linalg.norm(magnetic_field(eta=r[1], zeta=r[2], t=t)))                

        v = update_v_relativistic(
            v=v,
            E=electric_field(eta=r[1], zeta=r[2], t=t),
            B=magnetic_field(eta=r[1], zeta=r[2], t=t),
            dt=dt,
        )

        r = update_r(v, r, dt)

        new_trajectory.append(r)
        new_velocity.append(v)

        j += 1

        aux_time.append(t)
        t += dt

            #pbar.update(dt)

    time.append(aux_time)
    trajectories.append(new_trajectory)
    velocities.append(new_velocity)


#############################
# Plotting the trajectories #
#############################

time = np.array(time, dtype=object)
trajectories = np.array(trajectories, dtype=object)
velocities = np.array(velocities, dtype=object)

chi_plot = []
chi_plot_aux = []
eta_plot = []
zeta_plot = []
eta_plot_aux = []
zeta_plot_aux = []

for i in range(num_particles):
    chi_plot_aux = np.array([row[0] for row in trajectories[i]])
    eta_plot_aux = np.array([row[1] for row in trajectories[i]])
    zeta_plot_aux = np.array([row[2] for row in trajectories[i]])
    chi_plot.append(chi_plot_aux)
    eta_plot.append(eta_plot_aux)
    zeta_plot.append(zeta_plot_aux)


# Eta - zeta plot

plt.figure(1)  # Create the first figure
for i in range(num_particles):
    plt.plot(eta_plot[i], zeta_plot[i], color = "red") 
    plt.xlabel("Eta")
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.ylabel("Zeta")
    plt.title("Eta vs Zeta")
    plt.legend()


# Chi - zeta plot

plt.figure(2)  # Create the first figure
for i in range(num_particles):
    plt.plot(chi_plot[i], zeta_plot[i], color = "red") 
    plt.xlabel("Chi")
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.ylabel("Zeta")
    plt.title("Chi vs Zeta")
    plt.legend()


#############
# Snapshots #
#############

# Eta - zeta
# Define snapshot times
snapshot_times = [i * final_time / number_of_snapshots for i in range(number_of_snapshots)]

# Color map for snapshots
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_times)))

# Eta - zeta plot
# plt.figure(3)
# for snapshot_idx, snapshot_time in enumerate(snapshot_times):
#     for i in range(num_particles):
#         # Get the time list for the current particle
#         particle_times = time[i]

#         # Find the index closest to the snapshot time
#         closest_idx = min(range(len(particle_times)), key=lambda idx: abs(particle_times[idx] - snapshot_time))

#         # Get the position of the particle at this time
#         eta_snapshot = eta_plot[i][closest_idx]
#         zeta_snapshot = zeta_plot[i][closest_idx]

#         # Plot the snapshot point
#         plt.scatter(eta_snapshot, zeta_snapshot, color=colors[snapshot_idx], label=f"t={snapshot_time:.1f}" if i == 0 else "")

# plt.xlabel("Eta")
# plt.xlim(-40, 40)
# plt.ylim(-40, 40)
# plt.ylabel("Zeta")
# plt.title("Eta vs Zeta (Snapshots)")
# plt.legend()
# plt.show()


