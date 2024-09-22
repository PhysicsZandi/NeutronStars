import matplotlib.pyplot as plt
import os
import numpy as np


# Import classes
from solver import *


# Plot all the equation of state in the compose archive
def all_eos():

    # Dictionary to keep track of the folder features
    folder_labels = {
        "delta_models": {"label": "Delta", "color": "red", "plotted": False},
        "hybrid_models": {"label": "Hybrid", "color": "blue", "plotted": False},
        "hyperon_models": {"label": "Hyperon", "color": "green", "plotted": False},
        "nucleonic_models": {"label": "Nucleonic", "color": "violet", "plotted": False},
    }

    folder_path = "compose_eos"

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Find only files with csv extension
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                folder_name = os.path.basename(root)

                # Load the equation of state
                pressures = []
                energy_densities = []
                with open(file_path, "r") as f:
                    next(f)
                    for line in f:
                        n, p, e = line.strip().split(",")
                        pressures.append(float(p))
                        energy_densities.append(float(e))
                # Plot equation of state with color according to the folder
                folder_info = folder_labels[folder_name]
                if not folder_info["plotted"]:
                    plt.plot(
                        pressures,
                        energy_densities,
                        label=folder_info["label"],
                        color=folder_info["color"],
                    )
                    folder_info["plotted"] = True
                else:
                    plt.plot(pressures, energy_densities, color=folder_info["color"])

    plt.ylabel("Energy density ($erg / cm^3$)")
    plt.yscale("log")
    plt.xlabel("Pressure ($dyne / cm^2$)")
    plt.xscale("log")
    plt.legend()
    plt.show()


# Plot all the mass versus radius plots in the compose archive
def all_MvsR(resolution=1e3, n_points=40):

    # Dictionary to keep track of the folder features
    folder_features = {
        "delta_models": {"label": "Delta", "color": "red", "plotted": False},
        "hybrid_models": {"label": "Hybrid", "color": "blue", "plotted": False},
        "hyperon_models": {"label": "Hyperon", "color": "green", "plotted": False},
        "nucleonic_models": {"label": "Nucleonic", "color": "violet", "plotted": False},
    }

    range_initial_pressures = np.logspace(33, 36, n_points)

    folder_path = "compose_eos"

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Find only files with csv extension
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                base_name = os.path.splitext(file_name)[0]
                folder_name = os.path.basename(root)

                # Compute mass and radius of the object
                eos = interpolation_eos(file_path, low_density_eos=True)
                solver = Solver_range(
                    eos,
                    resolution,
                    range_initial_pressures,
                    relativity_corrections=True,
                )
                solver.solve()
                r, m = solver.get()

                # Plot equation of state with color according to the folder
                folder_info = folder_features[folder_name]
                if not folder_info["plotted"]:
                    plt.plot(
                        r,
                        m,
                        label=folder_info["label"],
                        color=folder_info["color"],
                    )
                    folder_info["plotted"] = True
                else:
                    plt.plot(r, m, color=folder_info["color"])

    plt.xlabel("Radius (km)")
    plt.ylabel("Mass ($M_{\\odot}$)")
    plt.legend()
    plt.show()


# Plot all the equation of state
# in folder given by folder_initial_letters and name given by file_initial_letters
def all_eos_v2(file_initial_letters="", folder_initial_letters=""):
    folder_path = "compose_eos"

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Find only files with csv extension
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                base_name = os.path.splitext(file_name)[0]
                folder_name = os.path.basename(root)

                if file_name.startswith(
                    file_initial_letters
                ) and folder_name.startswith(folder_initial_letters):
                    # Load the equation of state
                    pressures = []
                    energy_densities = []
                    with open(file_path, "r") as f:
                        next(f)
                        for line in f:
                            n, p, e = line.strip().split(",")
                            pressures.append(float(p))
                            energy_densities.append(float(e))

                    plt.plot(pressures, energy_densities, label=base_name)

    plt.ylabel("Energy density ($erg / cm^3$)")
    plt.yscale("log")
    plt.xlabel("Pressure ($dyne / cm^2$)")
    plt.xscale("log")
    plt.legend()
    plt.show()


# Plot all the mass versus radius plots
# in folder given by folder_initial_letters and name given by file_initial_letters
def all_MvsR_v2(
    resolution=1e3, n_points=40, file_initial_letters="", folder_initial_letters=""
):
    range_initial_pressures = np.logspace(33, 36, n_points)

    folder_path = "compose_eos"

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Find only files with csv extension
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                base_name = os.path.splitext(file_name)[0]
                folder_name = os.path.basename(root)

                if file_name.startswith(
                    file_initial_letters
                ) and folder_name.startswith(folder_initial_letters):
                    # Compute mass and radius of the object
                    eos = interpolation_eos(file_path, low_density_eos=True)
                    solver = Solver_range(
                        eos,
                        resolution,
                        range_initial_pressures,
                        relativity_corrections=True,
                    )
                    solver.solve()
                    r, m = solver.get()

                    plt.plot(r, m, label=base_name)

    plt.xlabel("Radius (km)")
    plt.ylabel("Mass ($M_{\\odot}$)")
    plt.legend()
    plt.show()
