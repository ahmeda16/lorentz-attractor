import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Lorenz system parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

def lorenz(t, state):
    x, y, z = state

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    
    differentials = [dxdt, dydt, dzdt]

    return differentials

def generate_trajectories(n_trajectories, t_span, t_eval, base_state, perturbation):
    trajectories = []

    for i in range(n_trajectories):
        initial_state = base_state + np.random.uniform(-perturbation, perturbation, size=3)
        sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
        
        trajectories.append(sol.y)

    return trajectories

def main():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d', facecolor="black")
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([0, 50])
    ax.set_title("Lorenz Attractor (Chaotic Divergence)", color="white", fontsize=16)
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.tick_params(colors='white')

    # Trajectory parameters
    n_trajectories = 10
    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 2000) 
    base_state = np.array([1.0, 1.0, 1.0]) 
    perturbation = 1e-4

    trajectories = generate_trajectories(n_trajectories, t_span, t_eval, base_state, perturbation)

    lines = [ax.plot([], [], [], lw=0.7, alpha=0.8)[0] for _ in range(n_trajectories)]
    
    # Private update function
    def update(frame):
        for line, traj in zip(lines, trajectories):
            line.set_data(traj[0, :frame], traj[1, :frame])
            line.set_3d_properties(traj[2, :frame])
        return lines

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)

    fig.patch.set_facecolor('black')
    plt.style.use("dark_background")

    plt.show()

    return 0

if __name__ == "__main__":
    main()