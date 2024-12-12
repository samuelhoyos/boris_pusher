import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


# Constants
e = 1.6e-19  # Elementary charge (C)
me = 9.11e-31  # Electron mass (kg)
c = 3.0e8  # Speed of light (m/s)
B0 = 1e-3  # Magnetic field strength (T)
beta_p = 0.2  # Normalized shock speed (v_s/c)
a = 0.05  # Magnetic curvature coefficient

# Derived quantities
v_s = beta_p * c  # Shock speed
omega_ce = e * B0 / me  # Electron cyclotron frequency
k = omega_ce / c

# Define the electromagnetic field functions
def electric_field(x, y, z, t):
    g1 = k * y + beta_p * omega_ce * t - a * k**2 * z**2
    g2 = k * y - beta_p * omega_ce * t + a * k**2 * z**2
    Et_x = -(v_s * B0 / (2 * c)) * (np.tanh(g1) - np.tanh(g2) - 2)
    return np.array([Et_x, 0, 0])

def magnetic_field(x, y, z, t):
    g1 = k * y + beta_p * omega_ce * t - a * k**2 * z**2
    g2 = k * y - beta_p * omega_ce * t + a * k**2 * z**2
    Bt_y = -(a *k * z* B0) * (np.tanh(g1) - np.tanh(g2) - 2)
    Bt_z = (B0 / 2) * (np.tanh(g1) + np.tanh(g2))
    return np.array([0, Bt_y, Bt_z])

# Equations of motion
def equations_of_motion(t, y):
    x, vx, y, vy, z, vz = y
    v = np.array([vx, vy, vz])

    # Fields at the particle's position
    E = electric_field(x, y, z, t)
    B = magnetic_field(x, y, z, t)

    # Lorentz force
    dv = (e / me) * (E + np.cross(v, B))
    return [vx, dv[0], vy, dv[1], vz, dv[2]]

# Initial conditions
num_particles = 1000  # Number of test particles
initial_positions = np.random.uniform(-10, 10, (num_particles, 3))  # (x, y, z)
initial_velocities = np.zeros_like(initial_positions)  # Assume particles start at rest

# Time integration parameters
time_span = (0, 200 * 2 * np.pi / omega_ce)  # Integration time
num_steps = 100

# Storage for trajectories
trajectories = []
time_points = np.linspace(*time_span, num_steps)

# Simulate each particle
for i in range(num_particles):
    y0 = [*initial_positions[i], *initial_velocities[i]]
    sol = solve_ivp(equations_of_motion, time_span, y0, t_eval=time_points, method='RK45')
    trajectories.append(sol.y)


#################################
# No animation (only snapshots) #
#################################


# Visualization: Snapshots at specific times
snapshot_times = [0, time_span[1]/3, 2 * time_span[1]/3, time_span[1]]
plt.figure(figsize=(15, 10))

for idx, snapshot_time in enumerate(snapshot_times):
    plt.subplot(2, 2, idx + 1)
    snapshot_positions_yz = []
    snapshot_positions_xy = []
    snapshot_positions_xz = []
    for trajectory in trajectories:
        # Find the index closest to the snapshot time
        snapshot_index = np.abs(time_points - snapshot_time).argmin()
        snapshot_positions_yz.append([trajectory[2][snapshot_index], trajectory[4][snapshot_index]])  # y, z
        snapshot_positions_xy.append([trajectory[0][snapshot_index], trajectory[2][snapshot_index]])  # x, y
        snapshot_positions_xz.append([trajectory[0][snapshot_index], trajectory[4][snapshot_index]])  # x, z
    snapshot_positions_yz = np.array(snapshot_positions_yz)
    snapshot_positions_xy = np.array(snapshot_positions_xy)
    snapshot_positions_xz = np.array(snapshot_positions_xz)
    
    
    # Plot for the y-z coordinates
    plt.scatter(snapshot_positions_yz[:, 0], snapshot_positions_yz[:, 1], s=10, alpha=0.7)
    plt.title(f"Snapshot at t = {snapshot_time:.2e} s")
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.grid()
    
    """
    # Plot for the x-y coordinates
    plt.scatter(snapshot_positions_xy[:, 0], snapshot_positions_xy[:, 1], s=10, alpha=0.7)
    plt.title(f"Snapshot at t = {snapshot_time:.2e} s")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid()
    


    # Plot for the x-z coordinates
    plt.scatter(snapshot_positions_xz[:, 0], snapshot_positions_xz[:, 1], s=10, alpha=0.7)
    plt.title(f"Snapshot at t = {snapshot_time:.2e} s")
    plt.ylim(-0.5e-5, 5e-5)
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.grid()

    """
plt.tight_layout()
plt.show()


# Trajectories in the y-z plane over time
plt.figure(figsize=(10, 6))
for trajectory in trajectories:
    plt.plot(trajectory[2], trajectory[4], alpha=0.6)  # Plot y vs. z
plt.title("Particle Trajectories in y-z Plane")
plt.xlabel("y (m)")
plt.ylabel("z (m)")
plt.grid()
plt.show()

# Trajectories in the x-y plane over time
plt.figure(figsize=(10, 6))
for trajectory in trajectories:
    plt.plot(trajectory[0], trajectory[2], alpha=0.6)  # Plot y vs. z
plt.title("Particle Trajectories in x-y Plane")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid()
plt.show()

# Trajectories in the x-z plane over time
plt.figure(figsize=(10, 6))
for trajectory in trajectories:
    plt.plot(trajectory[0], trajectory[4], alpha=0.6)  # Plot y vs. z
plt.title("Particle Trajectories in x-y Plane")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.grid()
plt.show()


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