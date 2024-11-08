from modules import functions, rkg
import numpy as np

def f(t, x):
    return -t/x

x_initial = 0
y_initial = 1
step_size = 0.01
x_final = 1

if __name__ == "__main__":
    y_values = rkg.rkg(f, x_initial, y_initial, step_size, x_final)
    
