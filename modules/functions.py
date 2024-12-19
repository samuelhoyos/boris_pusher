import numpy as np
from tqdm import tqdm


#####################################
# Update of the particle's position #
#####################################

def update_r(v: np.array, r: np.array, dt: float) -> np.array:
    return r + v * dt


#########################################################
# Relativistic Boris pusher using position and momentum #
#########################################################

def update_v_relativistic(
    v: np.array,
    E: np.array,
    B: np.array,
    dt: float,
    q: float = -1.0,
    m: float = 1.0,
    c: float = 1.0,
):
    
    
    gamma=1/np.sqrt(1-(np.linalg.norm(v)**2/c**2))
    p=gamma * m * v
    T = ((q * dt) / (2.0 * gamma)) * B
    s = (2.0 / (1.0 + np.linalg.norm(T)**2)) * T

    pm = p + (q * E * dt * 0.5)
    p_aux = pm + np.cross(pm, T)
    pp = pm + np.cross(p_aux, s)
    p = pp + (q * E * (0.5 * dt))

    gamma=np.sqrt(1+np.linalg.norm(p)**2/(m*c)**2)
    v=p/(gamma*m)

    return v
