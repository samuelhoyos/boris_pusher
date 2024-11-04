import pandas as pd
import numpy as np
from tqdm import tqdm



def update_x(v: np.array, x: np.array, dt: float, steps: float) -> np.array:
    for i in tqdm(range(0, steps - 1)):
        x[:, i + 1] = x[:, i] + v[:, i] * dt
    return x


def update_v(
    v: np.array,
    E: np.array,
    B: np.array,
    dt: float,
    steps: float,
    q=1.6e-19,
    m=9.11e-31,
    c=3e8
):
    s = np.zeros((3, steps))
    T = ((q * dt) / (2 * m)) * B
    # s = (2 / (1 + np.linalg.norm(T, axis=0) ** 2)) * T
    vm = np.zeros((3, steps))
    vp = np.zeros((3, steps))
    v_aux = np.zeros((3, steps))
    
    for i in tqdm(range(0, steps - 1)):

        v_mag = np.linalg.norm(v[:, i])
        gamma = 1 / np.sqrt(1 - (v_mag**2 / c**2))
        T[:, i] = ((q * dt) / (2 * gamma * m)) * B[:, i]
        s[:, i] = (2 / (1 + np.linalg.norm(T[:, i]) ** 2)) * T[:, i]
        vm[:, i] = v[:, i] + (q /(gamma* m)) * E[:, i] * dt * 0.5
        v_aux[:, i] = vm[:, i] + np.cross(vm[:, i], T[:, i])
        vp[:, i] = vm[:, i] + np.cross(v_aux[:, i], s[:, i])
        v[:, i + 1] = vp[:, i] + (q / (gamma*m)) * E[:, i] * 0.5 * dt

    return v
