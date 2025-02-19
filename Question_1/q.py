import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "Local_density_of_states_near_band_edge"
heatmap_dir = os.path.join(data_dir, "local density of states heatmap")

if not os.path.exists(heatmap_dir):
    os.makedirs(heatmap_dir)

def plot_heatmap(filename):
    """Plots and saves a heatmap of the LDOS data."""
    data = np.loadtxt(os.path.join(data_dir, filename))
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, cmap="inferno", cbar=True)
    plt.title(f"LDOS Heatmap - {filename}")
    plt.savefig(os.path.join(heatmap_dir, f"{filename}_heatmap.png"))
    plt.close()

for file in os.listdir(data_dir):
    if file.endswith(".txt"):
        plot_heatmap(file)

from mpl_toolkits.mplot3d import Axes3D

surface_dir = os.path.join(data_dir, "local density of states height")

if not os.path.exists(surface_dir):
    os.makedirs(surface_dir)

def plot_surface(filename):
    """Plots a 3D surface plot of the LDOS data."""
    data = np.loadtxt(os.path.join(data_dir, filename))
    X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, data, cmap="viridis")
    ax.set_title(f"LDOS Surface Plot - {filename}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Density")
    plt.savefig(os.path.join(surface_dir, f"{filename}_surface.png"))
    plt.close()

for file in os.listdir(data_dir):
    if file.endswith(".txt"):
        plot_surface(file)

subregion = (slice(10, 20), slice(10, 20))  # Select region (rows 10-20, cols 10-20)
subregion_means = []

for file in os.listdir(data_dir):
    if file.endswith(".txt"):
        data = np.loadtxt(os.path.join(data_dir, file))
        subregion_means.append(np.mean(data[subregion]))

plt.plot(range(len(subregion_means)), subregion_means, marker="o", linestyle="-")
plt.xlabel("File Index")
plt.ylabel("Average LDOS")
plt.title("Subregion LDOS Analysis")
plt.show()
