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
<<<<<<< Updated upstream
    
    gamma=1/np.sqrt(1-(np.linalg.norm(v)**2/c**2))
    p=gamma * m * v
    T = ((q * dt) / (2.0)) * B
    s = (2.0 / (1.0 + np.linalg.norm(T)**2)) * T

    pm = p + (q * E * dt * 0.5)
    p_aux = pm + np.cross(pm, T)
    pp = pm + np.cross(p_aux, s)
    p = pp + (q * E * (0.5 * dt))

    gamma=np.sqrt(1+np.linalg.norm(p)**2/(m*c)**2)
    v=p/(gamma*m)

    return v
=======

    #v_mag = np.linalg.norm(v)
    gamma = np.sqrt(1 + (v**2 / c**2))
    u = gamma * v
    u_1 = u + ((q / (2 * m)) * E * dt)

    gamma_b = np.sqrt(1 + (u_1**2 / c**2))
    omega = (q * B) / (m * c)

    u_2 = (
        u_1 * (1 - ((omega * dt) / (2 * gamma_b)) ** 2)
        + (dt * np.cross(u_1, omega)) / gamma_b
        + 1 / 2 * (dt / gamma_b) ** 2 * np.dot(u_1, omega) * omega
    ) / (1 + ((omega * dt) / (2 * gamma_b)) ** 2)

    u = u_2 + (q * dt)/(2 * m) * E
    gamma_c = np.sqrt(1 + (u**2 / c**2))

    # We must return v, not u!
    return u/gamma_c
>>>>>>> Stashed changes
