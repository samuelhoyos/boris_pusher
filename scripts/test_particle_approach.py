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
e = 1.609e-19  # Elementary charge (C)
me = 9.11e-31  # Electron mass (kg)
c = 3.0e8  # Speed of light (m/s)
B0 = 1  # Magnetic field strength (Normalized unit)
beta_p = 0.2  # Normalized shock speed (v_s/c)
a = 0.05  # Magnetic curvature coefficient

# Derived quantities
v_s = beta_p * c  # Shock speed
omega_ce = e * B0 / me  # Electron cyclotron frequency
k = omega_ce / c # Wave vector

# Final time for the simulation
final_time = 5e-7

# Ranges for g1, g2, t, and z
g1_min, g1_max = -10, 10
g2_min, g2_max = -10, 10
t_min, t_max = 0, final_time
z_min, z_max = -10 / k, 10 / k

# Calculate y_min
y_min = np.min([(g1_min - beta_p * omega_ce * t_max + a * k**2 * z_min**2) / k, (g2_min + beta_p * omega_ce * t_max - a * k**2 * z_max**2) / k])

# Calculate y_max
y_max = np.max([(g1_max - beta_p * omega_ce * t_min + a * k**2 * z_max**2) / k, (g2_max + beta_p * omega_ce * t_min - a * k**2 * z_min**2) / k])

# Output the range for y
print(f"Range for y: [{y_min}, {y_max}]")


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

def maximum_magnetic_field(final_time : float):
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
    print(f"At coordinates: y = {max_coords[0]}, z = {max_coords[1]}, t = {max_coords[2]}")

    return max_magnitude



"""
# Equations of motion
def equations_of_motion(t, y):
    x, vx, y, vy, z, vz = y

    v = np.sqrt(vx**2 + vy**2 + vz**2)
    gamma = 1 / np.sqrt(1 + (v / c) ** 2)

    v = np.array([vx, vy, vz])
    v = gamma * v

    # Fields at the particle's position
    E = electric_field(x, y, z, t)
    B = magnetic_field(x, y, z, t)

    # Lorentz force
    dv = (e / me) * (E + np.cross(v / gamma, B) / c)
    return [vx, dv[0], vy, dv[1], vz, dv[2]]
"""

# Initial conditions
num_particles = 2  # Number of test particles
initial_positions_x = np.zeros(num_particles)
initial_positions_y = np.random.uniform(y_min, y_max, num_particles)
initial_positions_z = np.random.uniform(z_min, z_max, num_particles)

initial_positions = np.column_stack((initial_positions_x, initial_positions_y, initial_positions_z))

# initial_positions = np.random.uniform(-10, 10, (num_particles, 3))  # (x, y, z)
initial_velocities = np.zeros((num_particles, 3))

for i in range(num_particles):
    if initial_positions[i, 1] >= 0:
        initial_velocities[i, 1] = -v_s
    else:
        initial_velocities[i, 1] = v_s

# Worst-case dt
# worst_dt = (0.5 *0.1 * me) / (e * np.linalg.norm(maximum_magnetic_field(final_time)))

# Storage for trajectories
trajectories = []
#longest_time = np.arange(0, final_time, worst_dt)

velocities = []
#trajectories = np.zeros((num_particles, len(longest_time), 3))
#velocities = np.zeros((num_particles, len(longest_time), 3))

for i in tqdm(range(num_particles)):
    #new_trajectory = np.zeros((len(longest_time), 3))
    #new_velocity = np.zeros((len(longest_time), 3))
    new_trajectory = []
    new_velocity = []
    
    r = initial_positions[i]
    v = initial_velocities[i]

    t = 0
    # idx = 0

    with tqdm(total=final_time) as pbar:
        while t < final_time:
            dt = (0.5 *0.1 * me) / (e * np.linalg.norm(magnetic_field(y=r[1], z=r[2], t=t)))
            v = update_v_relativistic(
                v=v,
                E=electric_field(y=r[1], z=r[2], t=t),
                B=magnetic_field(y=r[1], z=r[2], t=t),
                dt=dt,
            )

            r = update_r(v, r, dt)

            new_trajectory.append(r)
            new_velocity.append(v)

            #new_trajectory[idx] = r
            #new_velocity[idx] = v

            t += dt
            #idx += 1
        
            pbar.update(dt)


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

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each particle's trajectory
for i in range(num_particles):
    x = trajectories[i][:][0]
    y = trajectories[i][:][1]
    z = trajectories[i][:][2]
    ax.plot(x, y, z, label=f'Particle {i + 1}')

# Add labels and legend
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectories of Particles')
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
