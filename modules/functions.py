import numpy as np
from tqdm import tqdm


def update_r(v: np.array, r: np.array, dt: float, steps: float) -> np.array:
    return r + v*dt

def update_v_relativistic(
    v: np.array,
    E: np.array,
    B: np.array,
    dt: float,
    q:float=4.8032e-10,
    m:float=9.1094e-28,
    c:float=3e10
):
    s = np.zeros(3)
    T = ((q * dt) / (2 * m)) * B
    # s = (2 / (1 + np.linalg.norm(T, axis=0) ** 2)) * T
    vm = np.zeros(3)
    vp = np.zeros(3)
    v_aux = np.zeros(3)
    
    v_mag = np.linalg.norm(v)
    gamma = 1 / np.sqrt(1 - (v_mag*2 / c*2))
    T = ((q * dt) / (2 * gamma * m)) * B
    s = (2 / (1 + np.linalg.norm(T) ** 2)) * T
    vm = v + (q /(gamma* m)) * E * dt * 0.5
    v_aux = vm + np.cross(vm, T)
    vp = vm + np.cross(v_aux, s)
    v = vp + (q / (gamma*m)) * E * 0.5 * dt

    return v