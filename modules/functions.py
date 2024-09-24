import pandas as pd
import numpy as np
from tqdm import tqdm

def update_x(v: np.array, x: np.array, dt: float, steps: int) -> np.array:
    for i in tqdm(range(0, steps - 1)):
        x[:, i + 1] = x[:, i] + v[:, i] * dt
    return x

def half_n(A: np.array, idx: int) -> np.array:

    x_component = np.mean(A[0, idx], A[0, idx + 1])
    y_component = np.mean(A[1, idx], A[1, idx + 1])
    z_component = np.mean(A[2, idx], A[2, idx + 1])

    return np.array(x_component, y_component, z_component)

def update_v(v: np.array,
             E: np.array,
             B: np.array,
             dt: float,
             steps: int,
             q=1.6e-19,
             m=9.11e-31
             ):

    s = np.zeros((3, steps))
    vm = np.zeros((3, steps))
    vp = np.zeros((3, steps))
    v_aux = np.zeros((3, steps))
    
    for i in tqdm(range(0, steps - 1)):
        # First step of the algorithm: add half of the electric impulse to v to obtain vm
        vm[:, i] = v[:, i] + ((q / m) * half_n(E, i) * dt / 2)

        # Second step: perform a rotation to obtain vp, thanks to v_aux. T and s are defined here.
        T = ((q * dt) / (2 * m)) * half_n(B, i)
        s[:, i] = (2 / (1 + np.linalg.norm(T[:, i]) ** 2)) * T[:, i]
        v_aux[:, i] = vm[:, i] + np.cross(vm[:, i], T[:, i]) 
        vp[:, i] = vm[:, i] + np.cross(v_aux[:, i], s[:, i])

        # Third step: add the remaining electric impulse to obtain the updated velocity V(n+1)
        v[:, i + 1] = vp[:, i] + ((q / m) * half_n(E, i) * dt / 2)

    return v
