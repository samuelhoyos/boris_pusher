import pandas as pd
import numpy as np
from tqdm import tqdm



def update_r(v: np.array, r: np.array, dt: float, steps: float) -> np.array:
    for i in tqdm(range(0, steps - 1)):
        r = r + v*dt
    return r

def update_v_relativistic(
    v: np.array,
    E: np.array,
    B: np.array,
    dt: float,
    steps: float,
    q:float=1.6e-19,
    m:float=9.11e-31,
    c:float=3e8
):
    s = np.zeros(3)
    T = ((q * dt) / (2 * m)) * B
    # s = (2 / (1 + np.linalg.norm(T, axis=0) ** 2)) * T
    vm = np.zeros(3)
    vp = np.zeros(3)
    v_aux = np.zeros(3)
    
    for i in tqdm(range(0, steps - 1)):

        v_mag = np.linalg.norm(v)
        gamma = 1 / np.sqrt(1 - (v_mag**2 / c**2))
        T = ((q * dt) / (2 * gamma * m)) * B
        s = (2 / (1 + np.linalg.norm(T) ** 2)) * T
        vm = v + (q /(gamma* m)) * E * dt * 0.5
        v_aux = vm + np.cross(vm, T)
        vp = vm + np.cross(v_aux, s)
        v = vp + (q / (gamma*m)) * E * 0.5 * dt

    return v


