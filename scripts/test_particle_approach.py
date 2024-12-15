import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.functions import update_v_relativistic, update_r


# Constants (in the SI system)
# e = 1.609e-19  # Elementary charge (C)
# me = 9.11e-31  # Electron mass (kg)
# c = 3.0e8  # Speed of light (m/s)
# Normalized units
e = -1.0
me = 1.0
c = 1.0

B0 = 1 # Magnetic field strength (Normalized unit)
beta_p = 0.2  # Normalized shock speed (v_s/c)
a = 0.5  # Magnetic curvature coefficient

# Derived quantities
v_s = beta_p * c  # Shock speed
omega_ce = abs(e) * B0 / me  # Electron cyclotron frequency
k = omega_ce / c  # Wave vector

# Final time for the simulation
final_time = 1e3

# Ranges for g1, g2, t, and z
g1_min, g1_max = -10, 10
g2_min, g2_max = -10, 10
t_min, t_max = 0, final_time
z_min, z_max = -10 / k, 10 / k

# Calculate y_min
y_min = np.min(
    [
        (g1_min - beta_p * omega_ce * t_max + a * k**2 * z_min**2) / k,
        (g2_min + beta_p * omega_ce * t_max - a * k**2 * z_max**2) / k,
    ]
)

# Calculate y_max
y_max = np.max(
    [
        (g1_max - beta_p * omega_ce * t_min + a * k**2 * z_max**2) / k,
        (g2_max + beta_p * omega_ce * t_min - a * k**2 * z_min**2) / k,
    ]
)

# Output the range for y
print(f"Range for y: [{y_min}, {y_max}]")

# Range for y
max_abs_y = np.max([np.abs(y_min), np.abs(y_max)])
y_range = [-max_abs_y, max_abs_y] # Like this we are sure we are centered in 0


######################
# Initial conditions #
######################

num_particles = 3  # Number of test particles
initial_positions_x = np.zeros(num_particles)
seed = 40
np.random.seed(seed)
initial_positions_y = np.random.uniform(y_range[0], y_range[1], num_particles)
initial_positions_z = np.random.uniform(z_min, z_max, num_particles)

initial_positions = np.column_stack(
    (initial_positions_x, initial_positions_y, initial_positions_z)
)

print(initial_positions)

# initial_positions = np.random.uniform(-10, 10, (num_particles, 3))  # (x, y, z)
initial_velocities = np.zeros((num_particles, 3))

# This array says if a particle is going to the right or to the left
positive_negative_velocity = []

for i in range(num_particles):
    if initial_positions[i, 1] >= 0:
        positive_negative_velocity.append(True)
        initial_velocities[i, 1] = -v_s
    else:
        positive_negative_velocity.append(False)
        initial_velocities[i, 1] = v_s


# Define the electromagnetic field functions
def electric_field(y, z, t):
    g1 = k * y + beta_p * omega_ce * t - a * k**2 * z**2
    g2 = k * y - beta_p * omega_ce * t + a * k**2 * z**2
    Et_x = -(v_s * B0 / 2) * (np.tanh(g1) - np.tanh(g2) - 2)
    return np.array([Et_x, 0, 0])


def magnetic_field(y, z, t):
    g1 = k * y + beta_p * omega_ce * t - a * k**2 * z**2
    g2 = k * y - beta_p * omega_ce * t + a * k**2 * z**2
    Bt_y = -(a * k * z * B0) * (np.tanh(g1) - np.tanh(g2) - 2)
    Bt_z = (B0 / 2) * (np.tanh(g1) + np.tanh(g2))
    return np.array([0, Bt_y, Bt_z])


####################################################
# Finding the maximum value for the magnetic field #
####################################################


def maximum_magnetic_field(final_time: float):
    y_vals = np.linspace(y_min, y_max, 100)
    z_vals = np.linspace(z_min, z_max, 100)
    t_vals = np.linspace(0, final_time, 100)

    # Initialize variables to track maximum
    max_magnitude = 0
    max_coords = (0, 0, 0)

    # Grid search
    for y in tqdm(y_vals):
        for z in z_vals:
            for t in t_vals:
                B = magnetic_field(y, z, t)
                magnitude = np.linalg.norm(B)  # Compute the magnitude
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
                    max_coords = (y, z, t)

    # Output the results
    print(f"Maximum magnetic field magnitude: {max_magnitude}")
    print(
        f"At coordinates: y = {max_coords[0]}, z = {max_coords[1]}, t = {max_coords[2]}"
    )

    return max_magnitude




# Worst-case dt
# worst_dt = (0.5 *0.1 * me) / (e * np.linalg.norm(maximum_magnetic_field(final_time)))

# Storage for trajectories
trajectories = []
# longest_time = np.arange(0, final_time, worst_dt)

velocities = []
time = []
# trajectories = np.zeros((num_particles, len(longest_time), 3))
# velocities = np.zeros((num_particles, len(longest_time), 3))

for i in tqdm(range(num_particles)):
    # new_trajectory = np.zeros((len(longest_time), 3))
    # new_velocity = np.zeros((len(longest_time), 3))
    new_trajectory = []
    new_velocity = []
    aux_time = []

    r = initial_positions[i]
    v = initial_velocities[i]

    t = 0
    # idx = 0
    j = 0
    with tqdm(total=final_time) as pbar:
        while t < final_time:
            if np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t)) == 0: 
                print(f"number of iteration: {j}")

            dt = (0.5 * 0.1 * me) / (
                abs(e) * np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t))
            )
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

    # trajectories[i] = new_trajectory
    # velocities[i] = new_velocity

    # for it, t in enumerate(time):

    #     v = update_v_relativistic(
    #         v=v,
    #         E=electric_field(y=r[1], z=r[2], t=t),
    #         B=magnetic_field(y=r[1], z=r[2], t=t),
    #         dt=dt,
    #     )

    #     r = update_r(v, r, dt)

    #     new_trajectory[it] = r
    #     new_velocity[it] = v

    # trajectories[i] = new_trajectory
    # velocities[i] = new_velocity


#############################
# Plotting the trajectories #
#############################

trajectories = np.array(trajectories, dtype=object)

eta_plot = []
zeta_plot = []

for i in range(num_particles):
    zeta_plot.append(np.array([row[2] for row in trajectories[i]]) * k)

    if positive_negative_velocity: 
        eta_plot.append(k * np.array([row[1] for row in trajectories[i]])
        + beta_p * omega_ce * np.array(time[i])
        - a * k**2 * np.array([row[2] for row in trajectories[i]]) ** 2)
    else: 
        eta_plot.append(k * np.array([row[1] for row in trajectories[i]])
        - beta_p * omega_ce * np.array(time[i])
        + a * k**2 * np.array([row[2] for row in trajectories[i]]) ** 2)

if num_particles == 1: 
    eta_plot_sliced = np.copy(eta_plot[0][:-1])
    zeta_plot_sliced = np.copy(zeta_plot[0][:-1])

else: 
    eta_plot_sliced = []
    zeta_plot_sliced = []

    for i in range(num_particles): 
        eta_plot_sliced_aux = []
        zeta_plot_sliced_aux = []
        eta_plot_sliced_aux = np.copy(eta_plot[i][:-1])
        zeta_plot_sliced_aux = np.copy(zeta_plot[i][:-1])
        eta_plot_sliced.append(eta_plot_sliced_aux)
        zeta_plot_sliced.append(zeta_plot_sliced_aux)


if num_particles == 1: 
    fig = plt.figure(figsize=(10, 8))
    plt.plot(eta_plot_sliced, zeta_plot_sliced)
    plt.plot(eta_plot_sliced[-1], zeta_plot_sliced[-1], color="red", marker = "o")
    plt.xlabel("Eta")
    plt.ylabel("Zeta")
    plt.title("Eta vs Zeta")
    plt.legend()
    plt.show()

else:

    fig = plt.figure(figsize=(10, 8))

    # Loop with enumeration to get both the index and the data
    for i, (eta, zeta) in enumerate(zip(eta_plot_sliced, zeta_plot_sliced)):
        plt.plot(eta, zeta, label=f"Plot {i}")  # Use 'i' for labeling each plot

    # Add labels, title, and legend
    plt.xlabel("Eta")
    plt.ylabel("Zeta")
    plt.title("Eta vs Zeta")
    plt.legend()
    plt.xlim(-10, 10)
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
    ax.plot(x[-1], y[-1], z[-1], label=f"Particle {i + 1}", color = "red", marker = "o")

# Add labels and legend
ax.set_xlabel("X Position")
# ax.set_ylim(-1000, 1000)
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Trajectories of Particles")
ax.legend()

# Show the plot
plt.show()

# fig = plt.figure()
# plt.axes(projection="3d")  # Crear un eje 3D

# # Graficar cada trayectoria
# for idx in range(num_particles):
#     plt.scatter(
#         trajectories[idx][:, 0],
#         trajectories[idx][:, 1],
#         trajectories[idx][:, 2],
#         label=f"Particle {idx+1}"
#     )

# # Personalización del gráfico
# plt.title("Trajectories of Particles")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.legend()  # Mostrar la leyenda

# # Mostrar el gráfico
# plt.show()
"""
# Simulate each particle
for i in range(num_particles):
    y0 = [*initial_positions[i], *initial_velocities[i]]
    sol = solve_ivp(
        equations_of_motion, final_time, y0, t_eval=time_points, method="RK45"
    )
    trajectories.append(sol.y)
"""

#################################
# No animation (only snapshots) #
#################################


# # Visualization: Snapshots at specific times
# snapshot_times = [0, time_span[1] / 3, 2 * time_span[1] / 3, time_span[1]]
# plt.figure(figsize=(15, 10))

# for idx, snapshot_time in enumerate(snapshot_times):
#     plt.subplot(2, 2, idx + 1)
#     snapshot_positions_yz = []
#     snapshot_positions_xy = []
#     snapshot_positions_xz = []
#     for trajectory in trajectories:
#         # Find the index closest to the snapshot time
#         snapshot_index = np.abs(time_points - snapshot_time).argmin()
#         snapshot_positions_yz.append(
#             [trajectory[2][snapshot_index], trajectory[4][snapshot_index]]
#         )  # y, z
#         snapshot_positions_xy.append(
#             [trajectory[0][snapshot_index], trajectory[2][snapshot_index]]
#         )  # x, y
#         snapshot_positions_xz.append(
#             [trajectory[0][snapshot_index], trajectory[4][snapshot_index]]
#         )  # x, z
#     snapshot_positions_yz = np.array(snapshot_positions_yz)
#     snapshot_positions_xy = np.array(snapshot_positions_xy)
#     snapshot_positions_xz = np.array(snapshot_positions_xz)

#     # Plot for the y-z coordinates
#     plt.scatter(
#         snapshot_positions_yz[:, 0], snapshot_positions_yz[:, 1], s=10, alpha=0.7
#     )
#     plt.title(f"Snapshot at t = {snapshot_time:.2e} s")
#     plt.xlabel("y (m)")
#     plt.ylabel("z (m)")
#     plt.grid()

#     """
#     # Plot for the x-y coordinates
#     plt.scatter(snapshot_positions_xy[:, 0], snapshot_positions_xy[:, 1], s=10, alpha=0.7)
#     plt.title(f"Snapshot at t = {snapshot_time:.2e} s")
#     plt.xlabel("x (m)")
#     plt.ylabel("y (m)")
#     plt.grid()


#     # Plot for the x-z coordinates
#     plt.scatter(snapshot_positions_xz[:, 0], snapshot_positions_xz[:, 1], s=10, alpha=0.7)
#     plt.title(f"Snapshot at t = {snapshot_time:.2e} s")
#     plt.ylim(-0.5e-5, 5e-5)
#     plt.xlabel("x (m)")
#     plt.ylabel("z (m)")
#     plt.grid()

#     """
# plt.tight_layout()
# plt.show()


# # Trajectories in the y-z plane over time
# plt.figure(figsize=(10, 6))
# for trajectory in trajectories:
#     plt.plot(trajectory[2], trajectory[4], alpha=0.6)  # Plot y vs. z
# plt.title("Particle Trajectories in y-z Plane")
# plt.xlabel("y (m)")
# plt.ylabel("z (m)")
# plt.grid()
# plt.show()

# # Trajectories in the x-y plane over time
# plt.figure(figsize=(10, 6))
# for trajectory in trajectories:
#     plt.plot(trajectory[0], trajectory[2], alpha=0.6)  # Plot y vs. z
# plt.title("Particle Trajectories in x-y Plane")
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.grid()
# plt.show()

# # Trajectories in the x-z plane over time
# plt.figure(figsize=(10, 6))
# for trajectory in trajectories:
#     plt.plot(trajectory[0], trajectory[4], alpha=0.6)  # Plot y vs. z
# plt.title("Particle Trajectories in x-y Plane")
# plt.xlabel("x (m)")
# plt.ylabel("z (m)")
# plt.grid()
# plt.show()


"""
##################
# WITH ANIMATION #
################## 
# To plot data with an animation to better see the evolution
# Convert trajectories to a 3D array with consistent shape
trajectories = np.array(trajectories)  # Shape: (num_particles, 6, num_steps)

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-20, 20)
ax.set_ylim(-0.5e-5, 5e-5)
ax.set_xlabel("y (m)")
ax.set_ylabel("z (m)")
ax.set_title("Particle Motion Animation")
scat = ax.scatter([], [], s=10, alpha=0.7)

def init():
    scat.set_offsets(np.empty((0, 2)))  # Return an empty array for the initial frame
    return scat,

def update(frame):
    y = trajectories[:, 2, frame]  # y positions at this frame
    z = trajectories[:, 4, frame]  # z positions at this frame
    offsets = np.column_stack((y, z))  # Ensure correct shape
    scat.set_offsets(offsets)
    return scat,

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=20)

# Display the animation
plt.show()

# To save the animation as a file (optional):
# ani.save("particle_motion.mp4", writer="ffmpeg", dpi=300)
"""
