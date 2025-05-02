import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
L = 100          # Number of spatial points
T_steps = 500    # Number of time steps
dx = 0.1         # Spatial step
dt = 0.05        # Time step
c = 1.0          # Speed of propagation (set to 1)
m = 1.0          # Mass term
noise_amplitude = 0.05  # Initial quantum fluctuation amplitude

# Stability condition for finite difference (important!)
assert dt < dx/c, "Stability condition violated! (dt must be < dx/c)"

# Initialize field and its time derivative
phi = np.random.normal(0, noise_amplitude, size=L)
phi_old = np.copy(phi)  # Assume zero initial velocity
phi_new = np.zeros(L)

# Store evolution
phi_history = []

# Laplacian helper
def laplacian(arr):
    return (np.roll(arr, -1) - 2*arr + np.roll(arr, 1)) / dx**2

# Time evolution (Finite Difference Method)
for t in range(T_steps):
    lap = laplacian(phi)
    phi_new = (2*phi - phi_old) + (dt**2) * (c**2 * lap - m**2 * phi)
    
    phi_old = np.copy(phi)
    phi = np.copy(phi_new)
    
    phi_history.append(np.copy(phi))

phi_history = np.array(phi_history)

# ---- 2D Animation ----
fig2d, ax2d = plt.subplots()

line, = ax2d.plot([], [], lw=2)
ax2d.set_xlim(0, L*dx)
ax2d.set_ylim(-1, 1)
ax2d.set_title("2D Field Evolution (ϕ(x))")

def init_2d():
    line.set_data([], [])
    return line,

def update_2d(frame):
    x = np.linspace(0, L*dx, L)
    y = phi_history[frame]
    line.set_data(x, y)
    return line,

ani2d = animation.FuncAnimation(fig2d, update_2d, frames=T_steps, init_func=init_2d, blit=True)
plt.show()

# ---- 3D Surface Plot Animation ----
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

X = np.linspace(0, L*dx, L)
T = np.linspace(0, T_steps*dt, T_steps)
X, T = np.meshgrid(X, T)

def update_3d(frame):
    ax3d.clear()
    ax3d.plot_surface(X[:frame,:], T[:frame,:], phi_history[:frame,:], cmap="viridis")
    ax3d.set_xlabel('Space (x)')
    ax3d.set_ylabel('Time (t)')
    ax3d.set_zlabel('Field ϕ')
    ax3d.set_title("3D Field Surface Evolution")
    ax3d.set_zlim(-1, 1)

ani3d = animation.FuncAnimation(fig3d, update_3d, frames=T_steps, interval=30, repeat=False)
plt.show()

# ---- Heatmap Animation ----
fig_heat, ax_heat = plt.subplots()

cax = ax_heat.imshow(phi_history, aspect='auto', cmap='inferno', extent=[0, L*dx, 0, T_steps*dt])
ax_heat.set_xlabel('Space (x)')
ax_heat.set_ylabel('Time (t)')
ax_heat.set_title("Heatmap of Field Evolution")

plt.colorbar(cax, label='Field Value ϕ(x,t)')
plt.show()
