from modules import functions
import numpy as np

q=1.602e-19
m=9.11e-31
steps=1000000

x0 = np.zeros((3,steps))
v0 = np.zeros((3,steps))
E0 = np.zeros((3,steps))
B0 = np.zeros((3,steps))
B0[2,:]=1e-3
E0[1,::]=1e2
v0[0,0]=10

dt=((0.1*m)/(np.abs(q)*np.linalg.norm(B0[:,0])))*0.1

if __name__ == "__main__":
    print(dt)
    v=functions.update_v(v=v0,E=E0,B=B0,dt=dt, steps=steps)
    x=functions.update_x(v=v,x=x0,dt=dt,steps=steps)