import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Import physical constants
from physical_constants import *


# Interpolation of the equation of state from data by file_path
# Input: file_path - where the datas are located in
# Output: eos - the equation of state function e(p), pressure as input and energy density as output
def interpolation_eos(file_path, low_density_eos=False, plot=False):
    pressures = []
    energy_densities = []

    # For high density equations of state, the low density part is added
    if low_density_eos == True:
        with open(f"/Users/matte/projects/TOV/src/data/low_density.csv", "r") as file:
            next(file)  # Skip header
            for line in file:
                n, p, e = line.strip().split(",")
                pressures.append(float(p))
                energy_densities.append(float(e))

    # Open file and save pressures and energies
    with open(f"{file_path}", "r") as file:
        next(file)  # Skip header
        for line in file:
            n, p, e = line.strip().split(",")
            pressures.append(float(p))
            energy_densities.append(float(e))

    if plot == True:
        plt.plot(pressures, energy_densities)
        plt.ylabel("Energy density ($erg / cm^3$)")
        plt.yscale("log")
        plt.xlabel("Pressure ($dyne / cm^2$)")
        plt.xscale("log")
        plt.show()

    # Interpolation made with CubicSpline of scipy.interpolate
    eos = CubicSpline(pressures, energy_densities)

    return eos


# System of ordinary differential equations: dm/dr = g(p, m, r) and dp/dr = f(p, m, r)
# Input: eos - the equation of state function
#        relativity_corrections - boolean variable, if true TOV equation otherwise Newtonian equation
# Output: f and g are the equations in the system
class System:
    def __init__(self, eos, relativity_corrections=True):
        self.eos = eos

        # Choose if add any general relativity corrections
        if relativity_corrections == True:
            self.f = self.dpdr_rel
        else:
            self.f = self.dpdr
        self.g = self.dmdr

    # Mass equation
    def dmdr(self, r, p, m):
        # Stop if pressure is negative because we are at the surface
        if p <= 0:
            return 0

        e = self.eos(p)
        return 4 * np.pi * r**2 * e / c**2

    # Newtonian equation
    def dpdr(self, r, p, m):
        # Stop if pressure is negative because we are at the surface
        if p <= 0:
            return 0

        e = self.eos(p)
        return -G / c**2 / r**2 * m * e

    # TOV equation
    def dpdr_rel(self, r, p, m):
        # Stop if pressure is negative because we are at the surface
        if p <= 0:
            return 0

        e = self.eos(p)
        const = G * e * m / c**2 / r**2
        first_term = 1 + p / e
        second_term = 1 + 4 * np.pi * r**3 * p / m / c**2
        third_term = (1 - 2 * G * m / c**2 / r) ** (-1)
        return -const * first_term * second_term * third_term


# Solver for the system of ordinary differential equations of single initial pressure
# Input: system - class System
#        resolution - length of the radius step (also the resolution of the radius of the object)
#        p0 - initial pressure at the center of the object
# Output: getter and printer - respectively return and print the radius and the mass of the object
#         plotter - plot of the pressure and the mass versus the radius
class Solver:
    def __init__(self, eos, resolution, p0, relativity_corrections=True):
        self.system = System(eos, relativity_corrections)
        self.step_r = resolution

        # Initial conditions for the Cauchy problem
        self.p0 = p0
        self.m0 = 1e-6
        self.r0 = 1e-6

        # Initialise array to keep values in memory
        self.r_values = []
        self.p_values = []
        self.m_values = []

        # Add first value to the arrays
        self.r_values.append(self.r0)
        self.p_values.append(self.p0)
        self.m_values.append(self.m0)

    # Implementation of fourth-order Runge-Kutta method
    def runge_kurra_4_step(self, r, dr, p, m):
        kp_1 = self.system.f(r, p, m) * dr
        km_1 = self.system.g(r, p, m) * dr

        kp_2 = self.system.f(r + dr / 2, p + kp_1 / 2, m + km_1 / 2) * dr
        km_2 = self.system.g(r + dr / 2, p + kp_1 / 2, m + km_1 / 2) * dr

        kp_3 = self.system.f(r + dr / 2, p + kp_2 / 2, m + km_2 / 2) * dr
        km_3 = self.system.g(r + dr / 2, p + kp_2 / 2, m + km_2 / 2) * dr

        kp_4 = self.system.f(r + dr, p + kp_3, m + km_3) * dr
        km_4 = self.system.g(r + dr, p + kp_3, m + km_3) * dr

        p_new = p + (kp_1 + 2 * kp_2 + 2 * kp_3 + kp_4) / 6
        m_new = m + (km_1 + 2 * km_2 + 2 * km_3 + km_4) / 6

        return p_new, m_new

    # Algorithm that computes pressure and mass at any radius,
    # it breaks when pressure becomes negative (which means that we are at the surface), tracked by the variable breaking_condition
    def solve(self):
        breaking_condition = True

        while breaking_condition:
            # Take last values
            r = self.r_values[-1]
            p = self.p_values[-1]
            m = self.m_values[-1]

            # Find new values
            p_new, m_new = self.runge_kurra_4_step(r, self.step_r, p, m)
            r_new = r + self.step_r

            # Check if pressure is negative
            if p_new <= 0:
                breaking_condition = False

            # Add new values
            self.r_values.append(r_new)
            self.p_values.append(p_new)
            self.m_values.append(m_new)

    # Print radius in km and mass in m_sun
    def print(self):
        print("The radius is R =", self.r_values[-1] / km, "km")
        print("The mass is M =", self.m_values[-1] / m_sun, "M_sun")

    # Get radius in km and mass in m_sun
    def get(self):
        return self.r_values[-1] / km, self.m_values[-1] / m_sun

    # Plot pressure and mass versus radius (save_file allows to save file in the pdf folder)
    def plot(self, save_file=False):
        # Convert radii in km
        self.r_values = [r / km for r in self.r_values]

        # Convert masses in m_sun
        self.m_values = [m / m_sun for m in self.m_values]

        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        axs[0].plot(self.r_values, self.p_values)
        axs[0].set_xlabel("Radius ($km$)")
        axs[0].set_ylabel("Pressure ($dyne / cm^2$)")
        axs[1].plot(self.r_values, self.m_values)
        axs[1].set_xlabel("Radius ($km$)")
        axs[1].set_ylabel("Mass ($M_{\\odot}$)")
        plt.tight_layout()

        if save_file == True:
            plt.savefig("pdf/plot.pdf")

        plt.show()


# Solver for the system of ordinary differential equations of different initial pressures
# Input: system - class System
#        resolution - length of the radius step (also the resolution of the radius of the object)
#        range_initial_pressures - array with all initial pressures to integrate
# Output: printer - print maximum mass, radius and central pressure of the object
#         getter - return radii and masses
#         plotter - plot radius and mass versus pressure
#         plotter_RvsM - plot mass versus radius
class Solver_range:
    def __init__(
        self, eos, resolution, range_initial_pressures, relativity_corrections=True
    ):
        self.eos = eos
        self.resolution = resolution
        self.range_initial_pressures = range_initial_pressures
        self.relativity_corrections = relativity_corrections

        # Initialise array to keep values in memory
        self.masses = []
        self.radii = []

    # Compute the mass and the radius of the object for all initial pressures
    def solve(self):
        for p0 in self.range_initial_pressures:
            self.solver_single_p = Solver(
                self.eos, self.resolution, p0, self.relativity_corrections
            )
            self.solver_single_p.solve()
            r, m = self.solver_single_p.get()
            self.masses.append(m)
            self.radii.append(r)

    # Print maximum mass in m_sun, radius of that object in km and central pressure in dyne/cm^2
    def print(self):
        # Get index corresponding to maximum mass
        index = self.masses.index(max(self.masses))
        print("The maximum mass is M =", self.masses[index], "M_sun, ")
        print("which corresponds with a radius of R =", self.radii[index], "km")
        print(
            "and a central pressure of P =",
            self.range_initial_pressures[index],
            "dyne/cm^2",
        )

    # Get array of radii and masses
    def get(self):
        return self.radii, self.masses

    # Plot radius and mass versus pressure (save_file allows to save file in the pdf folder)
    def plot(self, save_file=False):
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))

        axs[0].plot(self.range_initial_pressures, self.radii)
        axs[0].set_xlabel("Pressure ($dyne / cm^2$)")
        axs[0].set_ylabel("Radius ($km$)")
        axs[0].set_xscale("log")

        axs[1].plot(self.range_initial_pressures, self.masses)
        axs[1].set_xlabel("Pressure ($dyne / cm^2$)")
        axs[1].set_ylabel("Mass ($M_{\\odot}$)")
        axs[1].set_xscale("log")

        if save_file == True:
            plt.savefig("pdf/plot.pdf")

        plt.tight_layout()
        plt.show()

    # Plot radius versus mass (save_file allows to save file in the pdf folder)
    def plot_RvsM(self, save_file=False):

        plt.plot(self.radii, self.masses)
        plt.xlabel("Radius ($km$)")
        plt.ylabel("Mass ($M_{\\odot}$)")

        if save_file == True:
            plt.savefig("pdf/plot_RvsM.pdf")

        plt.tight_layout()
        plt.show()
