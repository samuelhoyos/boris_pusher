import numpy as np

# Constants (replace with actual values)
a = 0.05  # placeholder value
B0 = 1  # placeholder value

# Define the magnetic field function
def magnetic_field(eta1, eta2, zeta):
    Bt_y = -(a * zeta * B0) * (np.tanh(eta1) - np.tanh(eta2) - 2)
    Bt_z = (B0 / 2) * (np.tanh(eta1) + np.tanh(eta2))
    return Bt_y, Bt_z

# Define the range for eta1, eta2, and zeta
eta1_range = np.linspace(-10, 0, 1000)  # Example range for eta1
eta2_range = np.linspace(0, 10, 1000)  # Example range for eta2
zeta_range = np.linspace(-10, 10, 1000)  # Example range for zeta

# Create the meshgrid for eta1, eta2, and zeta
eta1_grid, eta2_grid, zeta_grid = np.meshgrid(eta1_range, eta2_range, zeta_range)

# Compute the magnetic field components Bt_y and Bt_z over the grid
Bt_y_grid, Bt_z_grid = magnetic_field(eta1_grid, eta2_grid, zeta_grid)

# Compute the magnitude of the magnetic field
Bt_magnitude = np.sqrt(Bt_y_grid**2 + Bt_z_grid**2)

# Find the maximum and minimum of the magnetic field magnitude
max_field = np.max(Bt_magnitude)
min_field = np.min(Bt_magnitude)

# Find the corresponding values of eta1, eta2, and zeta for max and min
max_index = np.unravel_index(np.argmax(Bt_magnitude), Bt_magnitude.shape)
min_index = np.unravel_index(np.argmin(Bt_magnitude), Bt_magnitude.shape)

max_eta1, max_eta2, max_zeta = eta1_grid[max_index], eta2_grid[max_index], zeta_grid[max_index]
min_eta1, min_eta2, min_zeta = eta1_grid[min_index], eta2_grid[min_index], zeta_grid[min_index]

# Output the results
print("Maximum magnetic field magnitude:", max_field)
print("Corresponding eta1, eta2, zeta for max:", max_eta1, max_eta2, max_zeta)

print("Minimum magnetic field magnitude:", min_field)
print("Corresponding eta1, eta2, zeta for min:", min_eta1, min_eta2, min_zeta)
