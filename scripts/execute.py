import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import main
import numpy as np

q = 1.602e-19
m = 9.11e-31
steps = 1000000

# Each variable is an array with 3 rows (to account for the x, y and z directions) and a number of columns equal to "steps"
x0 = np.zeros((3, steps))
v0 = np.zeros((3, steps))
E0 = np.zeros((3, steps))
B0 = np.zeros((3, steps))

# Initial values for the magnetic and electric fields, and the velocity. The electric and magnetic fields are kept constant
# and the velocity is only given on the first point.
B0[2, :] = 1e-3  # Values for Bz
E0[1, :] = 1e2  # Values for Ey
v0[0, 0] = 10  # Initial value for Vx

dt = ((0.1 * m) / (np.abs(q) * np.linalg.norm(B0[:, 0]))) * 0.1

if __name__ == "__main__":
    print(dt)
    v = main.update_v(v=v0, E=E0, B=B0, dt=dt, steps=steps)
    x = main.update_x(v=v, x=x0, dt=dt, steps=steps)
