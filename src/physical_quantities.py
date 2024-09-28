from physical_constants import *
from solver import *
import numpy as np


# Numerical derivative implementation
def numerical_derivative(range_x, range_y):
    y_prime = []
    new_y = []

    for i in range(len(range_x)):
        if (
            i == 0
            or i == 1
            or i == 2
            or i == len(range_x)
            or i == len(range_x) - 1
            or i == len(range_x) - 2
        ):
            continue
        derivative = (
            (range_y[i - 2] - 8 * range_y[i - 1] + 8 * range_y[i + 1] - range_y[i + 2])
            / 12
            / (range_x[i] - range_x[i - 1])
        )
        y_prime.append(derivative)
        new_y.append(range_y[i])

    return new_y, y_prime


# Compute the derivative of the pressure over energy dentity
def compute_sound_velocity(range_p, range_e):
    new_p, p_prime = numerical_derivative(range_e, range_p)
    velocities = []
    for p in p_prime:
        if p < 0:
            v = 0
        else:
            v = np.sqrt(p * c**2)
        velocities.append(v)
    return new_p, velocities


# Compute the sound velocity, given an equation of state
def sound_velocity(file_path):
    pressures = []
    energy_densities = []

    with open(f"{file_path}", "r") as file:
        next(file)  # Skip header
        for line in file:
            n, p, e = line.strip().split(",")
            pressures.append(float(p))
            energy_densities.append(float(e))

    p_prime, v = compute_sound_velocity(pressures, energy_densities)
    return p_prime, v


# Compute the gravitational redshift with the GR formula
def compute_redshift(r, m):
    return (1 - 2 * G * m / r / c**2) ** (-1 / 2) - 1


# Compute the gravitational redshift, given an equation of state
def redshift(file_path):
    eos = interpolation_eos(file_path)
    resolution = 1e3
    range_initial_pressures = np.logspace(33, 36, 100)
    solver = Solver_range(
        eos, resolution, range_initial_pressures, relativity_corrections=True
    )
    solver.solve()
    r, m = solver.get()
    radii = [rs * km for rs in r]
    masses = [ms * m_sun for ms in m]

    redshifts = []
    for i in range(len(range_initial_pressures)):
        z = compute_redshift(radii[i], masses[i])
        redshifts.append(z)

    return range_initial_pressures, redshifts


# Compute Newtonian moment of inertia
def compute_moment_inertia(r, m):
    return 2 * m * r**2 / 5


# Compute the moment of inertia, given an equation of state
def moment_inertia(file_path):
    eos = interpolation_eos(file_path)
    resolution = 1e3
    range_initial_pressures = np.logspace(33, 36, 100)
    solver = Solver_range(
        eos, resolution, range_initial_pressures, relativity_corrections=True
    )
    solver.solve()
    r, m = solver.get()
    radii = [rs * km for rs in r]
    masses = [ms * m_sun for ms in m]

    moments = []
    for i in range(len(range_initial_pressures)):
        I = compute_moment_inertia(radii[i], masses[i])
        moments.append(I)

    return range_initial_pressures, moments
