import matplotlib.pyplot as plt
import os
import numpy as np
from solver import *


def numerical_derivative(range_x, range_y):
    """
    Computes the numerical derivative using a 5-point method.

    Parameters
    ----------
    range_x : NumPy array
        The independent variable.
    range_y : NumPy array
        The dependent variable.

    Returns
    --------
    NumPy array:
        A NumPy array containing the computed numerical derivatives.
    """
    range_y_prime = np.array([])

    for i in range(2, len(range_x) - 2):
        # Use central difference formula for numerical derivative
        derivative = (
            range_y[i - 2] - 8 * range_y[i - 1] + 8 * range_y[i + 1] - range_y[i + 2]
        ) / (12 * (range_x[i] - range_x[i - 1]))
        range_y_prime = np.append(range_y_prime, derivative)

    return range_y_prime


def compute_sound_speed(range_p, range_e):
    """
    Computes the square of sound, i.e. the derivative of pressure with respect to energy density.

    Parameters
    -----------
    range_p : NumPy array
        A NumPy array containing pressures.
    range_e : NumPy array
        A NumPy array containing energy densities.

    Returns
    -------
    NumPy array:
        A NumPy array containing the corresponding square speeds of sound in units of c^2.
    """
    p_prime = numerical_derivative(range_e, range_p)
    sound_speeds = np.array([])

    sound_speeds = np.maximum(p_prime, 0)  # if p > 0 append p, otherwise append 0

    return sound_speeds


def compute_redshift(r, m):
    """
    Compute the gravitational redshift using the general relativity formula.

    Parameters
    ----------
    r : float
        Radius in km.
    m : float
        Mass in solar masses.

    Returns
    -------
    float
        Dimensionless gravitational redshift.
    """
    r = r * km
    m = m * m_sun
    z = (1 - 2 * G * m / (r * c**2)) ** (-1 / 2) - 1
    return z


def compute_moment_inertia(r, m):
    """
    Compute the Newtonian moment of inertia of a sphere.

    Parameters
    ----------
    r : float
        Radius in km.
    m : float
        Mass in solar masses.

    Returns
    -------
    float
        Moment of inertia in g cm^2.
    """
    r = r * km
    m = m * m_sun
    I = (2 / 5) * m * r**2
    return I


def search_file_path(file_initial_letters="", folder_initial_letters=""):
    """
    Search for csv files in the "compose_eos" directory that match the given initial letters for both file names and folder names.

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


def plot_folder(folder_info, x_values, y_values):
    """
    Plot two NumPy arrays for a given folder with color according to the belonging folder of data. The first time data from a folder is plotted, add a label and use a color from a dictionary, then subsequent calls for the same folder plot data without a label but use same color.

    Parameters
    ----------
    folder_info : Dictionary
        Dictionary containing information if a folder label is already plotted with a Boolean variable.

    x_values : NumPy array
        NumPy array containing the values of the data points to be plotted on x-axis.

    y_values : NumPy array
         NumPy array containing the values of the data points to be plotted on y-axis.
    """

    # Plot equation of state with color according to the folder
    if not folder_info["plotted"]:
        plt.plot(
            x_values, y_values, label=folder_info["label"], color=folder_info["color"]
        )
        folder_info["plotted"] = True
    else:
        plt.plot(x_values, y_values, color=folder_info["color"])


def all_eos(file_initial_letters="", folder_initial_letters=""):
    """
    Plot the equation of state for different models based on the csv files found in the "compose_eos" directory that match the given initial letters for both file names and folder names. If no filters are applied, it plots all models with color corresponding to the folder.

    Parameters
    ----------
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

    file_info = search_file_path(file_initial_letters, folder_initial_letters)

    for info in file_info:
        file_path, folder_name, file_name = info

        EoS = EquationOfState()
        EoS.load_from_file(file_path)
        n, pressures, energy_densities = EoS.get()

        if file_initial_letters == "" and folder_initial_letters == "":
            folder_info = folder_features[folder_name]
            plot_folder(folder_info, pressures, energy_densities)
        else:
            plt.plot(pressures, energy_densities, label=file_name)

    plt.ylabel("Energy density ($erg / cm^3$)")
    plt.yscale("log")
    plt.xlabel("Pressure ($dyne / cm^2$)")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.title("Equation of state")
    plt.show()


def all_MvsR(
    first_initial_pressure=33,
    last_initial_pressure=36,
    file_initial_letters="",
    folder_initial_letters="",
    add_low_density=False,
):
    """
    Compute and plot radius versus mass for various models based on the equations of state files found in the "compose_eos" directory that match the given initial letters for both file names and folder names. If no filters are applied, it plots all models.

    Parameters
    ----------
    first_initial_pressure : float
        Exponent in base 10 of the first initial pressure to integrate. Default is 33.
    last_initial_pressure : float
        Exponent in base 10 of the last initial pressure to integrate. Default is 36.
    file_initial_letters : str
        Starting letters to filter files by name. Default is empty, meaning no filter.
    folder_initial_letters : str
        Starting letters to filter folders by name. Default is empty, meaning no filter.
    add_low_density : bool
        Boolean variable to indicate if low density - crust equation of state must be included.

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
    range_initial_pressures = np.logspace(
        first_initial_pressure, last_initial_pressure, n_points
    )

    file_info = search_file_path(file_initial_letters, folder_initial_letters)

    for info in file_info:
        file_path, folder_name, file_name = info

        # Compute mass and radius of the object
        EoS = EquationOfState()
        if add_low_density:
            EoS.load_from_file("data/low_density.csv")
        EoS.load_from_file(file_path)
        eos = EoS.interpolate()

        solver = SolverTOVRangePressure(eos, relativity_corrections=True)
        solver.solve(step_r, range_initial_pressures)
        r, m, p = solver.get()

        if file_initial_letters == "" and folder_initial_letters == "":
            folder_info = folder_features[folder_name]
            plot_folder(folder_info, r, m)
        else:
            plt.plot(r, m, label=file_name)

    plt.xlabel("Radius (km)")
    plt.ylabel("Mass ($M_{\\odot}$)")
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.title("Mass versus radius")
    plt.show()


def all_v(file_initial_letters="", folder_initial_letters=""):
    """
    Compute and plot square sound speed versus pressure for various models based on the equations of state files found in the "compose_eos" directory that match the given initial letters for both file names and folder names. If no filters are applied, it plots all models.

    Parameters
    ----------
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

    file_info = search_file_path(file_initial_letters, folder_initial_letters)

    for info in file_info:
        file_path, folder_name, file_name = info

        EoS = EquationOfState()
        EoS.load_from_file(file_path)
        n, pressures, energy_densities = EoS.get()

        sound_speeds = compute_sound_speed(pressures, energy_densities)
        # Derivative is computed with 5th points method so we need to remove first 2 and last 2 pressures
        pressures = pressures[:-2]
        pressures = pressures[2:]
        if file_initial_letters == "" and folder_initial_letters == "":
            folder_info = folder_features[folder_name]
            plot_folder(folder_info, pressures, sound_speeds)
        else:
            plt.plot(pressures, sound_speeds, label=file_name)

    plt.ylabel("Square sound speed ($c^2$)")
    plt.xlabel("Pressure ($dyne / cm^2$)")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.title("Square sound speed versus pressure")
    plt.show()


def all_z(
    first_initial_pressure=33,
    last_initial_pressure=36,
    file_initial_letters="",
    folder_initial_letters="",
    add_low_density=False,
):
    """
    Compute and plot redshift versus initial pressure for various models based on the equations of state files found in the "compose_eos" directory that match the given initial letters for both file names and folder names. If no filters are applied, it plots all models.

    Parameters
    ----------
    first_initial_pressure : float
        Exponent in base 10 of the first initial pressure to integrate. Default is 33.
    last_initial_pressure : float
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
    initial_pressures = np.logspace(
        first_initial_pressure, last_initial_pressure, n_points
    )

    file_info = search_file_path(file_initial_letters, folder_initial_letters)

    for info in file_info:
        file_path, folder_name, file_name = info

        # Compute mass and radius of the object
        EoS = EquationOfState()
        if add_low_density:
            EoS.load_from_file("data/low_density.csv")
        EoS.load_from_file(file_path)
        eos = EoS.interpolate()

        solver = SolverTOVRangePressure(eos, relativity_corrections=True)
        solver.solve(step_r, initial_pressures)
        radii, masses, p = solver.get()

        redshifts = np.array([])

        for i in range(len(initial_pressures)):
            z = compute_redshift(radii[i], masses[i])
            redshifts = np.append(redshifts, z)

        if file_initial_letters == "" and folder_initial_letters == "":
            folder_info = folder_features[folder_name]
            plot_folder(folder_info, initial_pressures, redshifts)
        else:
            plt.plot(initial_pressures, redshifts, label=file_name)

    plt.xlabel("Pressure ($dyne / cm^2$)")
    plt.xscale("log")
    plt.ylabel("Redshift")
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.title("Redshift versus pressure")
    plt.show()


def all_I(
    first_initial_pressure=33,
    last_initial_pressure=36,
    file_initial_letters="",
    folder_initial_letters="",
    add_low_density=False,
):
    """
    Compute and plot momenta of inertia versus initial pressure for various models based on the equations of state files found in the "compose_eos" directory that match the given initial letters for both file names and folder names. If no filters are applied, it plots all models.

    Parameters
    ----------
    first_initial_pressure : float
        Exponent in base 10 of the first initial pressure to integrate. Default is 33.
    last_initial_pressure : float
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
    initial_pressures = np.logspace(
        first_initial_pressure, last_initial_pressure, n_points
    )

    file_info = search_file_path(file_initial_letters, folder_initial_letters)

    for info in file_info:
        file_path, folder_name, file_name = info

        # Compute mass and radius of the object
        EoS = EquationOfState()
        if add_low_density:
            EoS.load_from_file("data/low_density.csv")
        EoS.load_from_file(file_path)
        eos = EoS.interpolate()

        solver = SolverTOVRangePressure(eos, relativity_corrections=True)
        solver.solve(step_r, initial_pressures)
        radii, masses, p = solver.get()

        momenta_inertia = np.array([])

        for i in range(len(initial_pressures)):
            I = compute_moment_inertia(radii[i], masses[i])
            momenta_inertia = np.append(momenta_inertia, I)

        if file_initial_letters == "" and folder_initial_letters == "":
            folder_info = folder_features[folder_name]
            plot_folder(folder_info, initial_pressures, momenta_inertia)
        else:
            plt.plot(initial_pressures, momenta_inertia, label=file_name)

    plt.xlabel("Pressure ($dyne / cm^2$)")
    plt.xscale("log")
    plt.ylabel("Moment of inertia ($g cm^2$)")
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.title("Momentum of inertia versus pressure")
    plt.show()
