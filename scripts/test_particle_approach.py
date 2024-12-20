import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

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
B0 = 0.2 # Magnetic field strength
beta_p = 0.2  # Normalized shock speed (v_s/c)
a = 0.05  # Magnetic curvature coefficient


# Derived quantities (also normalized)
v_s = beta_p * c  # Shock speed
omega_ce = abs(e) * B0 / (me * c)  # Electron cyclotron frequency
k = omega_ce / c  # Wave number = inverse of the width of the shock front

num_particles = 32  # Number of test particles

# Final time for the simulation
final_time = 100/B0
omega_ce = abs(e) * B0 / me  # Electron cyclotron frequency
k = omega_ce / c  # Width of the shock front

# Ranges for g1, g2, t, and z (optional)
g0_min, g0_max = -10.0, 10.0
zeta0_min, zeta0_max = -10.0, 10.0
t_min, t_max = 0.0, final_time

# Tolerance for the magnetic field
min_tolerance_B = 1

seed = 22 # Random seed
np.random.seed(seed)

# Used for plotting
number_of_snapshots = 1


####################################
# Ranges for the initial positions #
####################################

# Type 1 range (arbitrarily chosen)
#eta_ranges = [(-6, -4), (-6, -4), (-6, -4), (-6, -4), (4, 6), (4, 6), (4, 6), (4, 6)]
#zeta_ranges = [(-7, -6), (6, 7), (-5, -4), (4, 5), (-7, -6), (6, 7), (-5, -4), (4, 5)]
# range_num = 1

# Type 2 range (this one is the best one)
# eta_ranges = [(-7, -8), (-7, -8), (7, 8), (7, 8)]
# zeta_ranges = [(-7, -8), (7, 8), (-7, -8), (7, 8)]
# range_num = 2

# Type 3 range
#eta_ranges = [(0.5, -0.5)]
#zeta_ranges = [(0.5, -0.5)]
# range_num = 3

# Type 4 range
# eta_ranges = [(-1, -2), (-1, -2), (1, 2), (1, 2)]
# zeta_ranges = [(8, 9), (-8, -9), (8, 9), (-8, -9)]
# range_num = 4

# Type 5 range
eta_ranges = [(-10, 10)]
zeta_ranges = [(-10, 10)]
range_num = 5

# Type 6 range
# eta_ranges = [(-4, -10), (-4, -10), (4, 10), (4, 10)]
# zeta_ranges = [(-4, -10), (4, 10), (-4, -10), (4, 10)]
# range_num = 6

# Type 7 range
# eta_ranges = [(-10, -8), (8, 10)]
# zeta_ranges = [(-10, 10), (-10, 10)]
# range_num = 7


# Number of particles per subrange
particles_per_range = int(num_particles/len(eta_ranges))

# Name for the data to be stored in a .csv file at the end of the execution
dataName = f"B0 = {B0}, N = {num_particles}, t = {final_time}, range = {range_num}"



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

for i in tqdm(range(num_particles)):
    new_trajectory = []
    new_velocity = []
    aux_time = []

    r = initial_positions[i]
    v = initial_velocities[i]

    t = 0
    j = 0
    with tqdm(total=final_time) as pbar:     
        while (t < final_time):
            B_field = magnetic_field(eta=r[1], zeta=r[2], t=t)

            #if (np.linalg.norm(magnetic_field(eta=r[1], zeta=r[2], t=t)) > min_tolerance_B): 
            gamma = 1/np.sqrt(1 - np.linalg.norm(v)**2/c**2)
            # To ensure stability concerning dt, and avoiding that it gets too small or too big  
            dt = (0.05 * 0.1 * me * gamma) / (abs(e) * np.linalg.norm(B_field))                

            v = update_v_relativistic(
                v=v,
                E=electric_field(eta=r[1], zeta=r[2], t=t),
                B=B_field,
                dt=dt,
            )

            r = update_r(v, r, dt)

            new_trajectory.append(r)
            new_velocity.append(v)

            j += 1

            aux_time.append(t)
            t += dt

            pbar.update(dt)

    time.append(aux_time)
    trajectories.append(new_trajectory)
    velocities.append(new_velocity)


#############################
# Plotting the trajectories #
#############################

time = np.array(time, dtype=object)
trajectories = np.array(trajectories, dtype=object)
velocities = np.array(velocities, dtype=object)

xi_plot = []
xi_plot_aux = []
eta_plot = []
eta_plot_aux = []
zeta_plot = []
zeta_plot_aux = []

for i in range(num_particles):
    xi_plot_aux = np.array([-row[0] for row in trajectories[i]]) # Detail : we think xi = -x
    eta_plot_aux = np.array([row[1] for row in trajectories[i]])
    zeta_plot_aux = np.array([row[2] for row in trajectories[i]])
    xi_plot.append(xi_plot_aux)
    eta_plot.append(eta_plot_aux)
    zeta_plot.append(zeta_plot_aux)


# Storing the data for plotting

df = pd.DataFrame({"time": time, "xi": xi_plot, "eta": eta_plot, "zeta": zeta_plot})
df.to_parquet(f"C:\\Users\\danie\\Desktop\\Images Ciardi\\{dataName}.csv")