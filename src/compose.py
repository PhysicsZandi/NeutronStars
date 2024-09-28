import matplotlib.pyplot as plt
import os
import numpy as np
from solver import *


def search_file_path(file_initial_letters="", folder_initial_letters=""):
    """
    Search for csv files in the "compose_eos" directory that match the given
    initial letters for both file names and folder names.

    Parameters
    ----------
    file_initial_letters : str
        Starting letters to filter files by name. Default is empty, meaning no filter.
    folder_initial_letters : str
        Starting letters to filter folders by name. Default is empty, meaning no filter.

    Returns
    -------
    list: A list of lists, where each sublist contains the file path, folder name and file name
          of each matching file.
    """
    file_info = []

    for root, dirs, files in os.walk("compose_eos"):
        for file_name in files:
            # Find only files with csv extension
            if file_name.endswith(".csv"):
                # Find values according to folder and name
                file_path = os.path.join(root, file_name)
                folder_name = os.path.basename(root)
                file_name = os.path.splitext(file_name)[0]
                if file_name.startswith(
                    file_initial_letters
                ) and folder_name.startswith(folder_initial_letters):
                    file_info.append([file_path, folder_name, file_name])

    return file_info


def all_eos(file_initial_letters="", folder_initial_letters=""):
    """
    Plot the equation of state for different models based on the csv files found in the 'compose_eos' directory that match the given initial letters for both file names and folder names. If no filters are applied, it plots all
    models with color corresponding to the folder.

    Parameters
    ----------
    file_initial_letters : str
        Starting letters to filter files by name. Default is empty, meaning no filter.
    folder_initial_letters : str
        Starting letters to filter folders by name. Default is empty, meaning no filter.

    """

    # Dictionary to keep track of the folder features
    folder_labels = {
        "delta_models": {"label": "Delta", "color": "red", "plotted": False},
        "hybrid_models": {"label": "Hybrid", "color": "blue", "plotted": False},
        "hyperon_models": {"label": "Hyperon", "color": "green", "plotted": False},
        "nucleonic_models": {"label": "Nucleonic", "color": "grey", "plotted": False},
    }

    file_info = search_file_path(file_initial_letters, folder_initial_letters)

    for info in file_info:
        file_path, folder_name, file_name = info

        EoS = EquationOfState()
        EoS.load_from_file(file_path)
        pressures, energy_densities = EoS.get()

        if file_initial_letters == "" and folder_initial_letters == "":
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
        else:
            plt.plot(pressures, energy_densities, label=file_name)

    plt.ylabel("Energy density ($erg / cm^3$)")
    plt.yscale("log")
    plt.xlabel("Pressure ($dyne / cm^2$)")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.show()


def all_MvsR(
    last_initial_pressure=36, file_initial_letters="", folder_initial_letters=""
):
    """
    Compute and plot radius versus mass for various models based on the equations of state files found in the 'compose_eos' directory that match the given initial letters for both file names and folder names. If no filters are applied, it plots all models.

    Parameters
    ----------
    final_pressure : float
        Exponent in base 10 of the last initial pressure to integrate. Default is 36.
    file_initial_letters : str
        Starting letters to filter files by name. Default is empty, meaning no filter.
    folder_initial_letters : str
        Starting letters to filter folders by name. Default is empty, meaning no filter.

    """

    # Dictionary to keep track of the folder features
    folder_features = {
        "delta_models": {"label": "Delta", "color": "red", "plotted": False},
        "hybrid_models": {"label": "Hybrid", "color": "blue", "plotted": False},
        "hyperon_models": {"label": "Hyperon", "color": "green", "plotted": False},
        "nucleonic_models": {"label": "Nucleonic", "color": "grey", "plotted": False},
    }
    step_r = 1e3
    n_points = 40
    range_initial_pressures = np.logspace(33, last_initial_pressure, n_points)

    file_info = search_file_path(file_initial_letters, folder_initial_letters)

    for info in file_info:
        file_path, folder_name, file_name = info

        # Compute mass and radius of the object
        EoS = EquationOfState()
        EoS.load_from_file("data/low_density.csv")
        EoS.load_from_file(file_path)
        eos = EoS.interpolate()

        solver = SolverRangePressure(eos, relativity_corrections=True)
        solver.solve(step_r, range_initial_pressures)
        r, m, p = solver.get()
        if file_initial_letters == "" and folder_initial_letters == "":
            # Plot equation of state with color according to the folder
            folder_info = folder_features[folder_name]
            if not folder_info["plotted"]:
                plt.plot(r, m, label=folder_info["label"], color=folder_info["color"])
                folder_info["plotted"] = True
            else:
                plt.plot(r, m, color=folder_info["color"])
        else:
            plt.plot(r, m, label=file_name)

    plt.xlabel("Radius (km)")
    plt.ylabel("Mass ($M_{\\odot}$)")
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.show()
