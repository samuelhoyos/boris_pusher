import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
B0 = 10  # Magnetic field strength
beta_p = 0.2  # Normalized shock speed (v_s/c)
a = 0.05  # Magnetic curvature coefficient

# Derived quantities (also normalized)
v_s = beta_p * c  # Shock speed
omega_ce = abs(e) * B0 / (me * c)  # Electron cyclotron frequency
k = omega_ce / c  # Width of the shock front

# Final time for the simulation
final_time = 10

# Ranges for g1, g2, t, and z
g0_min, g0_max = -10.0, 10.0
zeta0_min, zeta0_max = -10.0, 10.0
t_min, t_max = 0.0, final_time
z_min, z_max = zeta0_min / k, zeta0_max / k

# Tolerance for the magnetic field
tolerance = 0.01

seed = 358 # Random seed
np.random.seed(seed)

num_particles = 1  # Number of test particles



##############################################
# Define the electromagnetic field functions #
##############################################


def electric_field(y, z, t):
    g1 = k * y + beta_p * omega_ce * t - a * k**2 * z**2
    g2 = k * y - beta_p * omega_ce * t + a * k**2 * z**2
    Et_x = -(v_s * B0 / (2.0 * c)) * (np.tanh(g1) - np.tanh(g2) - 2.0)
    return np.array([Et_x, 0, 0])


def magnetic_field(y, z, t):
    g1 = k * y + beta_p * omega_ce * t - a * k**2 * z**2
    g2 = k * y - beta_p * omega_ce * t + a * k**2 * z**2
    Bt_y = -(a * k * z * B0) * (np.tanh(g1) - np.tanh(g2) - 2.0)
    Bt_z = (B0 / 2.0) * (np.tanh(g1) + np.tanh(g2))
    return np.array([0, Bt_y, Bt_z])


######################
# Initial conditions #
######################

initial_positions_x = np.zeros(num_particles)
initial_positions_y = np.zeros(num_particles)


initial_positions_eta = np.random.uniform(g0_min, g0_max, num_particles)
print(f"Initial positions eta: {initial_positions_eta}")

initial_positions_zeta = np.random.uniform(zeta0_min, zeta0_max, num_particles)
print(f"Initial positions zeta: {initial_positions_zeta}")


initial_positions_z = initial_positions_zeta / k
for i, eta in enumerate(initial_positions_eta): 
    if eta > 0:
        # eta_1
        initial_positions_y[i] = (initial_positions_eta[i] + a * initial_positions_zeta[i]**2) / k 
    else:
        # eta_2
        initial_positions_y[i] = (initial_positions_eta[i] - a * initial_positions_zeta[i]**2) / k


initial_positions = np.column_stack(
    (initial_positions_x, initial_positions_y, initial_positions_z)
)

print(f"Initial positions: {initial_positions}")

initial_velocities = np.zeros((num_particles, 3))

# This array says if a particle is going to the right or to the left
is_initial_velocity_positive = []

for i in range(num_particles):
    initial_velocities[i] = (
        np.cross(
            magnetic_field(y=initial_positions[i, 1], z=initial_positions[i, 2], t=0),
            electric_field(y=initial_positions[i, 1], z=initial_positions[i, 2], t=0),
        )
        / np.linalg.norm(
            magnetic_field(y=initial_positions[i, 1], z=initial_positions[i, 2], t=0)
        )**2
    )
    if initial_positions[i, 1] >= 0:
        is_initial_velocity_positive.append(False)
    else:
        is_initial_velocity_positive.append(True)

print(f"Initial velocities: {initial_velocities}")


####################################################
# Finding the maximum value for the magnetic field #
####################################################


# def maximum_magnetic_field(final_time: float):
#     y_vals = np.linspace(y_min, y_max, 100)
#     z_vals = np.linspace(z_min, z_max, 100)
#     t_vals = np.linspace(0, final_time, 100)

#     # Initialize variables to track maximum
#     max_magnitude = 0
#     max_coords = (0, 0, 0)

#     # Grid search
#     for y in tqdm(y_vals):
#         for z in z_vals:
#             for t in t_vals:
#                 B = magnetic_field(y, z, t)
#                 magnitude = np.linalg.norm(B)  # Compute the magnitude
#                 if magnitude > max_magnitude:
#                     max_magnitude = magnitude
#                     max_coords = (y, z, t)

#     # Output the results
#     print(f"Maximum magnetic field magnitude: {max_magnitude}")
#     print(
#         f"At coordinates: y = {max_coords[0]}, z = {max_coords[1]}, t = {max_coords[2]}"
#     )

#     return max_magnitude


# Storage for trajectories, velocities and time
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

            omega_0 = abs(e) * np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t)) / me

            if (np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t)) > tolerance):           
                dt = (0.01 * 0.1 * me) / (abs(e) * np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t)))    

            # if omega_0 * dt < 0.1: 
            #     # print(f"Stability condition not satisfied at position = {j}")
            #     # print(f"Value of B = {np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t))}")
            #     # print(f"Current dt = {dt}")
            #     dt = (0.5 * 0.1 * me) / (abs(e) * np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t)))
            

            v = update_v_relativistic(
                v=v,
                E=electric_field(y=r[1], z=r[2], t=t),
                B=magnetic_field(y=r[1], z=r[2], t=t),
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

trajectories = np.array(trajectories, dtype=object)

eta_plot = []
zeta_plot = []

for i in range(num_particles):
    zeta_plot.append(np.array([row[2] for row in trajectories[i]]) * k)

    if is_initial_velocity_positive[i]:
        eta_plot.append(
            k * np.array([row[1] for row in trajectories[i]])
            + v_s * k * np.array(time[i])
            - a * k**2 * np.array([row[2] for row in trajectories[i]])**2
        )
    else:
        eta_plot.append(
            k * np.array([row[1] for row in trajectories[i]])
            - v_s * k * np.array(time[i])
            + a * k**2 * np.array([row[2] for row in trajectories[i]])**2
        )


for i in range(num_particles):
    plt.plot(eta_plot[i], zeta_plot[i]) 
    plt.xlabel("Eta")
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.ylabel("Zeta")
    plt.title("Eta vs Zeta")
    plt.legend()

plt.show()

################
# x, y, z plot #
################

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot each particle's trajectory
for i in range(num_particles):
    x = [row[0] for row in trajectories[i]]
    y = [row[1] for row in trajectories[i]]
    z = [row[2] for row in trajectories[i]]
    ax.plot(x, y, z, label=f"Particle {i + 1}")

# Add labels and legend
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Trajectories of Particles")
ax.legend()

# Show the plot
plt.show()