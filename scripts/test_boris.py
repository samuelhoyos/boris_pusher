import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from modules import functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

initial_v = np.array([1, 0, 0])
initial_r = np.array([0, 0, 0])
E = np.array([0.1, 0.05, 0])
B = np.array([1, 1, 1])
dt = (0.01 * 9.11e-31) / (1.602e-19 * np.linalg.norm(B))
times = np.arange(0, 1e-10, dt)
velocity = np.zeros((len(times), 3))
position = np.zeros((len(times), 3))
energy = np.zeros(len(times))
for i, t in enumerate(times):
    if i == 0:
        velocity[i] = initial_v
        position[i] = initial_r

    else:
        velocity[i] = functions.update_v_relativistic(
            v=velocity[i - 1], E=E, B=B, dt=dt
        )
        position[i] = functions.update_r(v=velocity[i], r=position[i - 1], dt=dt)
    energy[i] = 0.5 * 9.11e-31 * np.linalg.norm(velocity[i], ord=2)


fig = plt.figure()
plt.axes(projection="3d").scatter(position[:, 0], position[:, 1], position[:, 2])
print(energy)

plt.show()
