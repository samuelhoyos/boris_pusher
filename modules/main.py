import pandas as pd
import numpy as np
from tqdm import tqdm
import constants

def update_x(v: np.array, x: np.array, dt: float, steps: int) -> np.array:
    for i in tqdm(range(0, steps - 1)):
        x[:, i + 1] = x[:, i] + v[:, i] * dt
    return x

def update_v(v: np.array,
             E: np.array,
             B: np.array,
             dt: float,
             steps: int
             ):

    s = np.zeros((3, steps))
    vm = np.zeros((3, steps))
    vp = np.zeros((3, steps))
    v_aux = np.zeros((3, steps))
    
    for i in tqdm(range(0, steps - 1)):
        # First step of the algorithm: add half of the electric impulse to v to obtain vm
        vm[:, i] = v[:, i] + ((constants.q / constants.m) * E[:, i] * dt / 2)

        # Second step: perform a rotation to obtain vp, thanks to v_aux. T and s are defined here.
        T = ((constants.q * dt) / (2 * constants.m)) * B[:, i]
        s[:, i] = (2 / (1 + np.linalg.norm(T[:, i]) ** 2)) * T[:, i]
        v_aux[:, i] = vm[:, i] + np.cross(vm[:, i], T[:, i]) 
        vp[:, i] = vm[:, i] + np.cross(v_aux[:, i], s[:, i])

        # Third step: add the remaining electric impulse to obtain the updated velocity V(n+1)
        v[:, i + 1] = vp[:, i] + ((constants.q / constants.m) * E[:, i] * dt / 2)

    return v
