import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Import physical constants for computations
from physical_constants import *


class EquationOfState:
    """
    Class to represent an equation of state.

    Attributes
    ----------
    number_densities : NumPy array
        NumPy array that stores number densities in cm^-3.
    pressures : NumPy array
        NumPy array that stores pressures in dyne/cm^2.
    energy_densities : NumPy array
        NumPy array that stores energy densities in erg/cm^3.

    Methods
    -------
    load_from_file(file_path)
        Read data from a file and store number densities, pressures and energy densities in corresponding NumPy arrays.
    plot()
        Plot energy densities versus pressure.
    interpolate()
        Interpolate using cubic spline and return the equation of state in the form of energy density as a function of pressure.
    get()
        Return NumPy arrays containing pressure and energy density.
    """

    def __init__(self):
        """
        Initialise NumPy arrays to store number densities, pressures and energy densities.
        """
        self.number_densities = np.array([])
        self.pressures = np.array([])
        self.energy_densities = np.array([])

    def load_from_file(self, file_path):
        """
        Open file given its path, read data and store quantities in corresponding NumPy arrays. The first row is skipped because it is the header. The delimiter must be comma for csv files. In the first column there must be number densities in 1/cm^3, in the second column there must be pressures in dyne/cm^2, in the third column there must be energy densities in erg/cm^3.

        Parameters
        ----------
        file_path : str
            Path of the file containing number densities, pressures and energy densities in respective columns.

        Raises
        ------
        InputError
            The file cannot be opened.
        ValueError
            The file is empty.
            Pressures are not strictly increasing.
        """
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            self.number_densities = np.append(self.number_densities, data[:, 0])
            self.pressures = np.append(self.pressures, data[:, 1])
            self.energy_densities = np.append(self.energy_densities, data[:, 2])
        except Exception as ex:
            raise IOError(f"Failed to load file {file_path}: {ex}")

        if self.pressures.size == 0 or self.energy_densities.size == 0:
            raise ValueError("File is empty.")

        # Ensure pressures are strictly increasing for interpolation
        is_sorted = np.all(np.diff(self.pressures) > 0)
        if not is_sorted:
            raise ValueError("Pressures must be strictly increasing.")

    def plot(self):
        """
        Plot energy density versus pressure using logarithmic scale for both x and y axis with a grid.

        Raises
        ------
        ValueError
            No data is loaded to plot.
        """
        if self.pressures.size == 0 or self.energy_densities.size == 0:
            raise ValueError("No data loaded.")

        plt.plot(self.pressures, self.energy_densities)
        plt.ylabel("Energy Density ($erg / cm^3$)")
        plt.yscale("log")
        plt.xlabel("Pressure ($dyne / cm^2$)")
        plt.xscale("log")
        plt.grid(True)
        plt.show()

    def interpolate(self):
        """
        Interpolate pressures and energy densities into an equation of state using cubic spline (imported from SciPy).

        Returns
        -------
        CubicSpline
            CubicSpline representation of the equation of state, which gives energy density as output once pressure is given as input.

        Raises
        ------
        ValueError
            No data is loaded to interpolate.
        """
        if self.pressures.size == 0 or self.energy_densities.size == 0:
            raise ValueError("No data loaded.")

        eos = CubicSpline(self.pressures, self.energy_densities)
        return eos

    def get(self):
        """
        Return NumPy arrays containing pressures and energy densities.

        Returns
        -------
        tuple
            NumPy arrays of pressures and energy densities.
        """
        return self.pressures, self.energy_densities


class TOVSystem:
    """
    Class to represent the system of ordinary differential equations for a spherically symmetric physical object
    which is in static gravitational equilibrium. The system is composed in 3 equations: the mass equation, the pressure equation
    and the equation of state. The pressure equation can be either Newtonian or relativistic (the latter called properly TOV equation).

    Attributes
    ----------
    eos : CubicSpline
        A CubicSpline representation of the equation of state.
    relativity_corrections : bool
        A Boolean variable to determined whether to include general relativity corrections. If True the TOV equation is used, if False the Newtonian equation is used.

    Methods
    -------
    check_physical_constraints(r, p, m)
        Control if physical quantities, like mass, pressure and radius, are positive.
    dmdr(r, p, m)
        Compute and return the mass equation accordingly to current radius, pressure and mass.
    dpdr(r, p, m)
        Compute and return the pressure equation accordingly to current radius, pressure, mass. It can consider relativity corrections if attribute relativity_corrections is True.
    """

    def __init__(self, eos, relativity_corrections=True):
        """
        Take an equation of state and an option for relativity corrections and copy them as attributes of the class.

        Parameters:
        eos : EquationOfState
            A CubicSpline representation of the equation of state.
        relativity_corrections : bool
            A Boolean variable to determined whether to include general relativity corrections. If True the TOV equation is used, if False the Netwonian equation is used.
        """
        self.eos = eos
        self.relativity_corrections = relativity_corrections

    def check_physical_constraints(self, r, p, m):
        """
        Check the physical constraints for radius, pressure and mass, which must be all positive. When pressure is negative, return False to signal that the surface of the object is reached and integration has to stop.

        Parameters
        ----------
        r : float
          Radius in cm.
        p : float
          Pressure in dyne/cm^2.
        m : float
          Mass in g.

        Raises
        ------
        ValueError
            If radius is negative.
            If mass is negative.

        Returns
        -------
        bool
            True if all physical constraints are satisfied, False if pressure is negative.
        """
        # Stop if pressure is negative (indicating that we are at the surface)
        if p < 0:
            return False

        if r < 0:
            raise ValueError("Radius cannot be negative.")

        if m < 0:
            raise ValueError("Mass cannot be negative.")

        return True

    def dmdr(self, r, p, m):
        """
        Compute the mass equation (dm/dr) based on the current radius, pressure and mass.

        Parameters
        ---------
        r : float
          Radius in cm.
        p : float
          Pressure in dyne/cm^2.
        m : float
          Mass in g.

        Returns
        -------
        float
            Mass equation of the system (dm/dr).
        """
        if not self.check_physical_constraints(r, p, m):
            return 0

        e = self.eos(p)  # Get energy density from equation of state
        dmdr = 4 * np.pi * r**2 * e / c**2
        return dmdr

    def dpdr(self, r, p, m):
        """
        Compute the pressure equation (dp/dr) based on the current radius, pressure and mass.
        If relativity corrections are included, return the TOV equation, otherwise return the Newtonian pressure equation.

        Parameters
        ----------
        r : float
            Radius in cm.
        p : float
          Pressure in dyne/cm^2.
        m : float
            Mass in g.

        Returns
        -------
        float
            Pressure equation of the system (dp/dr).
        """
        # If pressure is negative, we are at the surface and return 0
        if not self.check_physical_constraints(r, p, m):
            return 0

        e = self.eos(p)  # Get energy density from equation of state

        if self.relativity_corrections:
            first_term = G * e * m / c**2 / r**2
            second_term = 1 + p / e
            third_term = 1 + 4 * np.pi * r**3 * p / m / c**2
            fourth_term = (1 - 2 * G * m / c**2 / r) ** (-1)
            dpdr = -first_term * second_term * third_term * fourth_term  # TOV equation
        else:
            dpdr = -G / c**2 / r**2 * m * e  # Newtonian pressure equation

        return dpdr


class SolverSinglePressure:
    """
    Class to solve numerically the system of ordinary differential equations given a single initial pressure.
    Implement the 4th-order Runge-Kutta with a breaking condition when the pressure becomes zero or negative, because it means that the surface of the object is reached.

    Attributes
    ----------
    radii : NumPy array
        NumPy array that stores radii in km.
    masses : NumPy array
        NumPy array that stores masses in solar masses.
    pressures : NumPy array
        NumPy array that stores pressures in dyne/cm^2.

    Methods
    -------
    runge_kutta_4th_step(r, dr, p, m)
        Implement the Runge-Kutta 4th order algorithm for a single step to solve numerically a system of two differential equations, according to current pressure, mass, radius and radius step.
    solve(step_r, p0)
        Solve the system until pressure drops to zero given the initial pressure and radius step. Store radii, masses and pressures in corresponding NumPy arrays.
    print_mass_radius()
        Print the mass and the radius of the object.
    get()
        Return radius, mass and pressure NumPy arrays.
    plot()
        Plot mass and radius versus pressure.
    """

    def __init__(self, eos, relativity_corrections=True):
        """
        Initialize the class TOVSystem with an equation of state and an option for relativity corrections.

        Parameters
        ----------
        eos : CubicSpline
            A CubicSpline representation of the equation of state.
        relativity_corrections : bool
            A Boolean variable to determined whether to include general relativity corrections. If True the TOV equation is considered, if False the Netwonian equation is considered.
        """
        self.system = TOVSystem(eos, relativity_corrections)

    def runge_kutta_4th_step(self, r, dr, p, m):
        """
        Perform a single step of the 4th-order Runge-Kutta algorithm. Given the current radius, pressure, mass and radius step, compute the 4 coefficients of the algorithm, using the equations in the methods of TOVSystem, and then compute and return the new calculated value of pressure and mass.

        Parameters
        ----------
        r : float
            Current radius in cm.
        dr : float
            Radius step in cm.
        p : float
            Current pressure in dyne/cm^2.
        m : float
            Current mass in g.

        Returns
        -------
        tuple
            NumPy arrays of pressures and masses.
        """
        kp_1 = self.system.dpdr(r, p, m) * dr
        km_1 = self.system.dmdr(r, p, m) * dr

        kp_2 = self.system.dpdr(r + dr / 2, p + kp_1 / 2, m + km_1 / 2) * dr
        km_2 = self.system.dmdr(r + dr / 2, p + kp_1 / 2, m + km_1 / 2) * dr

        kp_3 = self.system.dpdr(r + dr / 2, p + kp_2 / 2, m + km_2 / 2) * dr
        km_3 = self.system.dmdr(r + dr / 2, p + kp_2 / 2, m + km_2 / 2) * dr

        kp_4 = self.system.dpdr(r + dr, p + kp_3, m + km_3) * dr
        km_4 = self.system.dmdr(r + dr, p + kp_3, m + km_3) * dr

        p_new = p + (kp_1 + 2 * kp_2 + 2 * kp_3 + kp_4) / 6
        m_new = m + (km_1 + 2 * km_2 + 2 * km_3 + km_4) / 6

        return p_new, m_new

    def solve(self, step_r, p0):
        """
        Apply the 4th-order Runge-Kutta step using a while loop for the given initial pressure until pressure drops to zero. For every iteration, update the radius, pressure and mass values in the corresponding NumPy arrays. Change unit of measure of mass from g to solar masses and radius from cm to km.

        Parameters
        ----------
        step_r : float
            Radius step in cm.
        p0 : float
            Initial central pressure in dyne/cm^2.

        Raises
        ------
        ValueError
            If radius step is negative.
            If initial pressure is negative.
        """
        if step_r < 0:
            raise ValueError("Radius step cannot be negative.")

        if p0 < 0:
            raise ValueError("Initial pressure cannot be negative.")

        # Initial conditions for the Cauchy problem
        p0 = p0
        m0 = 1e-6  # Small initial mass to prevent division by zero
        r0 = 1e-6  # Small initial radius to avoid division by zero

        # Initialize NumPy arrays with the initial conditions
        self.r_values = np.array([r0])
        self.p_values = np.array([p0])
        self.m_values = np.array([m0])

        pressure_is_negative = False

        while not pressure_is_negative:
            # Take last values
            r = self.r_values[-1]
            p = self.p_values[-1]
            m = self.m_values[-1]

            # Find new values using 4-th order Runge-Kutta algorithm
            p_new, m_new = self.runge_kutta_4th_step(r, step_r, p, m)
            r_new = r + step_r

            # Check if pressure is zero or negative
            if p_new <= 0:
                pressure_is_negative = True  # Condition to break the while loop

            # Add new values to the arrays
            self.r_values = np.append(self.r_values, r_new)
            self.p_values = np.append(self.p_values, p_new)
            self.m_values = np.append(self.m_values, m_new)

        # Convert radius to km and mass to solar masses
        self.r_values = self.r_values / km
        self.m_values = self.m_values / m_sun

    def print_mass_radius(self):
        """
        Print the radius in km and mass in solar masses of the object, rounded to fourth significant digits.

        Raises
        ------
        ValueError
            No data is computed to print.
        """
        if (
            self.r_values.size == 0
            or self.p_values.size == 0
            or self.m_values.size == 0
        ):
            raise ValueError("No data is computed.")

        print("The radius is R =", round(self.r_values[-1], 4), "km")
        print("The mass is M =", round(self.m_values[-1], 4), "M_sun")

    def get(self):
        """
        Return radius, mass and pressure NumPy arrays.

        Returns
        --------
        tuple
            NumPy array of radius in km, mass in solar masses and pressures in dyne/cm^2.

        Raises
        ------
        ValueError
            No data is computed to get.
        """
        if (
            self.r_values.size == 0
            or self.p_values.size == 0
            or self.m_values.size == 0
        ):
            raise ValueError("No data is computed.")

        return self.r_values, self.m_values, self.p_values

    def plot(self):
        """
        Plot mass and radius versus pressure.
        Show two subplots: radius versus pressure and mass versus pressure with a grid.

        Raises
        -------
        ValueError
            No data is computed to plot.
        """
        if (
            self.r_values.size == 0
            or self.p_values.size == 0
            or self.m_values.size == 0
        ):
            raise ValueError("No data is computed.")

        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        axs[0].plot(self.r_values, self.p_values)
        axs[0].set_xlabel("Radius ($km$)")
        axs[0].set_ylabel("Pressure ($dyne / cm^2$)")
        axs[0].grid(True)
        axs[1].plot(self.r_values, self.m_values)
        axs[1].set_xlabel("Radius ($km$)")
        axs[1].set_ylabel("Mass ($M_{\\odot}$)")
        axs[1].grid(True)
        plt.show()


class SolverRangePressure:
    """
    Class to solve numerically the system of ordinary differential equations given a range of initial pressures.
    It uses the class SolverSinglePressure for each initial pressure.

    Attributes
    ----------
    eos : CubicSpline
        A CubicSpline representation of the equation of state.
    relativity_corrections : bool
        A Boolean variable to determined whether to include general relativity corrections. If True the TOV equation is considered, if False the Netwonian equation is considered.
    radii : NumPy array
        NumPy array that stores radii in km.
    masses : NumPy array
        NumPy array that stores masses in solar masses.
    initial_pressures : NumPy array
        NumPy array that stores initial pressures in dyne/cm^2.

    Methods
    -------
    solve(step_r, initial_pressures)
        Integrate the system until pressure drops to zero given the initial pressure and radius step. Store radii, masses and pressures in corresponding NumPy arrays.
    print_max_mass()
        Print the maximum mass, its radius and central pressure.
    get()
        Return NumPy arrays containing radius, mass and initial pressure.
    plot_MRvsP()
        Plot mass and radius versus initial pressure.
    plot_RvsM()
        Plot radius versus mass.
    """

    def __init__(self, eos, relativity_corrections=True):
        """
        Take an equation of state and an option for relativity corrections and copy them to attributes of the class.
        Initialise NumPy arrays to store radii in km, pressures in dyne/cm^2 and masses in solar masses.

        Parameters
        ----------
        eos : CubicSpline
            A CubicSpline representation of the equation of state.
        relativity_corrections : bool
            A Boolean variable to determined whether to include general relativity corrections. If True the TOV equation is considered, if False the Netwonian equation is considered.
        """
        self.eos = eos
        self.relativity_corrections = relativity_corrections

        self.radii = np.array([])
        self.masses = np.array([])
        self.initial_pressures = np.array([])

    def solve(self, step_r, initial_pressures):
        """
        Solve the system for a range of initial pressures. Using the class SolverSinglePressure, compute the mass and the radius and store them for each initial pressure.

        Parameters
        ----------
        step_r : float
            Radius step in cm.
        initial_pressures : NumPy array
            NumPy array of initial pressures in dyne/cm^2.

        Raises
        -------
        ValueError
            If radius step is negative.
            If initial pressure is negative.
        """
        if step_r < 0:
            raise ValueError("Radius step cannot be negative.")

        if initial_pressures.any() < 0:
            raise ValueError("Initial pressures cannot be negative.")

        self.initial_pressures = np.append(self.initial_pressures, initial_pressures)

        for p0 in initial_pressures:
            # Solve for a single pressure p0
            self.solver_single_p = SolverSinglePressure(
                self.eos, self.relativity_corrections
            )
            self.solver_single_p.solve(step_r, p0)

            # Save mass and radius
            r, m, p = self.solver_single_p.get()
            self.masses = np.append(self.masses, m[-1])
            self.radii = np.append(self.radii, r[-1])

    def print_max_mass(self):
        """
        Print the maximum mass, its corresponding radius and central pressure.

        Raises
        -------
        ValueError
            No data is computed to print.
        """
        if (
            self.radii.size == 0
            or self.masses.size == 0
            or self.initial_pressures.size == 0
        ):
            raise ValueError("No data is computed.")

        index = np.argmax(self.masses)  # Get index of maximum mass
        print("The maximum mass is M =", round(self.masses[index], 3), "M_sun, ")
        print("which corresponds to a radius of R =", round(self.radii[index], 3), "km")
        print(
            "and a central pressure of P =",
            round(self.initial_pressures[index], 3),
            "dyne/cm^2",
        )

    def get(self):
        """
        Return the NumPy arrays of radius, mass and initial pressure.

        Returns
        --------
        tuple
            NumPy array of radius in km, mass in solar masses and initial pressure in dyne/cm^2.

        Raises
        ------
        ValueError
            No data is computed to get
        """
        if (
            self.radii.size == 0
            or self.masses.size == 0
            or self.initial_pressures.size == 0
        ):
            raise ValueError("No data is computed.")

        return self.radii, self.masses, self.initial_pressures

    def plot_MRvsP(self):
        """
        Plot mass and radius versus initial pressure.
        Show two subplots: radius versus initial pressure and mass versus initial pressure in logarithmic scal for the x axis with a grid.

        Raises
        -------
        ValueError
            No data is computed to plot
        """
        if (
            self.radii.size == 0
            or self.masses.size == 0
            or self.initial_pressures.size == 0
        ):
            raise ValueError("No data is computed.")

        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        axs[0].plot(self.initial_pressures, self.radii)
        axs[0].set_xlabel("Pressure ($dyne / cm^2$)")
        axs[0].set_ylabel("Radius ($km$)")
        axs[0].set_xscale("log")
        axs[0].grid(True)
        axs[1].plot(self.initial_pressures, self.masses)
        axs[1].set_xlabel("Pressure ($dyne / cm^2$)")
        axs[1].set_ylabel("Mass ($M_{\\odot}$)")
        axs[1].set_xscale("log")
        axs[1].grid(True)
        plt.show()

    def plot_RvsM(self):
        """
        Plot radius versus mass with a grid.

        Raises
        ------
        ValueError
            No data is computed to plot
        """
        if (
            self.radii.size == 0
            or self.masses.size == 0
            or self.initial_pressures.size == 0
        ):
            raise ValueError("No data is computed.")

        plt.plot(self.radii, self.masses)
        plt.xlabel("Radius ($km$)")
        plt.ylabel("Mass ($M_{\\odot}$)")
        plt.grid(True)
        plt.show()
