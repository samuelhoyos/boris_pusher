import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Copy the path directly from the file
dataName = "B0 = 0.1, N = 4, t = 150.0, range = 2"
num_of_particles = 4 # To introduce manually the number of particles to plot
num_of_snapshots = 50

# Extracted data from the file
df = pd.read_parquet(f"C:\\Users\\danie\\Desktop\\Images Ciardi\\{dataName}.csv")

###################
# Eta - zeta plot #
###################

plt.figure(1)

for i in range(num_of_particles): 

    plt.plot(df.eta.iloc[i], df.zeta.iloc[i], color = "red") 
    plt.xlabel("η")
    plt.ylabel("ζ")


###################
# Chi - zeta plot #
###################

plt.figure(2)

for i in range(num_of_particles): 

    plt.plot(df.xi.iloc[i], df.zeta.iloc[i], color = "red") 
    plt.xlabel("ξ")
    plt.ylabel("ζ")


##################
# Chi - eta plot #
##################

plt.figure(3)

for i in range(num_of_particles): 

    plt.plot(df.xi.iloc[i], df.eta.iloc[i], color = "red") 
    plt.xlabel("ξ")
    plt.ylabel("η")



snapshot_times = [i * df.time.iloc[0][-1] / num_of_snapshots for i in range(num_of_snapshots)]

# Color map for snapshots
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_times)))

# Eta - zeta plot
plt.figure(4)
for snapshot_idx, snapshot_time in enumerate(snapshot_times):
    for i in range(num_of_particles):
        # Get the time list for the current particle
        particle_times = df.time.iloc[i]

        # Find the index closest to the snapshot time
        closest_idx = min(range(len(particle_times)), key=lambda idx: abs(particle_times[idx] - snapshot_time))

        # Get the position of the particle at this time
        eta_snapshot = df.eta.iloc[i][closest_idx]
        zeta_snapshot = df.zeta.iloc[i][closest_idx]

        # Plot the snapshot point
        plt.scatter(eta_snapshot, zeta_snapshot, color=colors[snapshot_idx], label=f"t={snapshot_time:.1f}" if i == 0 else "")

plt.xlabel("η")
plt.ylabel("ζ")
plt.title("η vs ζ (Snapshots)")
plt.show()




plt.show()