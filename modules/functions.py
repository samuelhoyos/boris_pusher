import numpy as np
from tqdm import tqdm


def update_r(v: np.array, r: np.array, dt: float) -> np.array:
    return r + v * dt


def update_v_relativistic(
    v: np.array,
    E: np.array,
    B: np.array,
    dt: float,
    q: float = -1.0,
    m: float = 1.0,
    c: float = 1.0,
):
    
    v_mag = np.linalg.norm(v)
    gamma = 1.0 / np.sqrt(1.0 - (v_mag**2 / c**2))
    T = ((q * dt) / (2.0 * gamma * m)) * B
    s = (2.0 / (1.0 + np.linalg.norm(T) ** 2)) * T

    vm = v + ((q / (gamma * m)) * E * dt * 0.5)
    v_prime = vm + np.cross(vm, T)
    v_plus = vm + np.cross(v_prime, s)

    v_mag = np.linalg.norm(v_plus)

    gamma = 1.0 / np.sqrt(1.0 - (v_mag**2 / c**2))
    v = v_plus + (q / (gamma * m)) * E * 0.5 * dt

    return v
