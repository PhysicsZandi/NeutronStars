import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Import physical constants
from physical_constants import *


# Polytropic equation of state p = k e^gamma or e = (p / k)^(1 / gamma)
def polytropic_eos(p, k, gamma):
    return (p / k) ** (1 / gamma)


# Range of initial pressure and number desnity that we consider
range_initial_pressures = np.logspace(20, 42, 200)
range_number_densities = np.logspace(26.5, 44, 200)

# Degenerate ideal Fermi gas of neutrons

# Constants

e_0_e = m_e**4 * c**5 / np.pi**2 / hbar**3
e_0_n = m_n**4 * c**5 / np.pi**2 / hbar**3
e_0_p = m_p**4 * c**5 / np.pi**2 / hbar**3


# Non-relativistic case
def fermi_nonrel(range_initial_pressures):
    k_nonrel = hbar**2 / 15 / np.pi**2 / m_n * (3 * np.pi**2 / m_n / c**2) ** (5 / 3)
    gamma_nonrel = 5 / 3

    energy_densities_nonrel = [
        polytropic_eos(p, k_nonrel, gamma_nonrel) for p in range_initial_pressures
    ]

    return energy_densities_nonrel


# Generic case
def fermi_gen(range_initial_pressures):
    # Compute the pressure given x = k / (m * c ) and a pressure as input to match
    # If it returns zero, then the pressure is exactly the desidered one
    def compute_pressure_gen(x, p):
        return (
            e_0_n
            / 24
            * ((2 * x**3 - 3 * x) * (1 + x**2) ** (1 / 2) + 3 * np.arcsinh(x))
            - p
        )

    # Compute the energy density given x = k / (m * c )
    def compute_energy_density_gen(x):
        k_f = x * m_n * c
        n = k_f**3 / 3 / np.pi**2 / hbar**3
        e_n = n * m_n
        e_e = e_0_n / 8 * ((2 * x**3 + x) * (1 + x**2) ** (1 / 2) - np.arcsinh(x))
        return e_n + e_e

    # Given a desidered pressure, find the corresponding x using the toms748 algorithm
    def compute_x_gen(p):
        x = opt.toms748(compute_pressure_gen, 1e-4, 1e2, args=(p,))
        return x

    # Given a desidered pressure, return the corresponding energy density
    def eos_gen(p):
        x = compute_x_gen(p)
        e = compute_energy_density_gen(x)
        return e

    energy_densities_gen = [eos_gen(p) for p in range_initial_pressures]

    return energy_densities_gen


# Degenerate ideal Fermi gas of neutron, protons and electrons
def fermi_npe(range_number_densities):
    # Integral appearing in the pressure as function of x = k / (m * c) and prefactor e_0
    def integral_pressure(x, e_0):
        p = e_0 / 24 * ((2 * x**3 - 3 * x) * (1 + x**2) ** (1 / 2) + 3 * np.arcsinh(x))
        return p

    # Integral appearing in the energy density as function of x = k / (m * c) and prefactor e_0
    def integral_energy_density(x, e_0):
        return e_0 / 8 * ((2 * x**3 + x) * (1 + x**2) ** (1 / 2) - np.arcsinh(x))

    # Condition of beta equailibrium allows to compute k_p as a function of k_n
    def beta_equilibrium(k_p, k_n):
        mu_p = (k_p**2 * c**2 + m_p**2 * c**4) ** (1 / 2)
        mu_e = (k_p**2 * c**2 + m_e**2 * c**4) ** (1 / 2)
        mu_n = (k_n**2 * c**2 + m_n**2 * c**4) ** (1 / 2)
        return mu_n - mu_p - mu_e

    # Compute k_p as a function of k_n using fsolve algorithm
    def compute_k_p(k_n):
        k_p = opt.fsolve(beta_equilibrium, k_n, args=(k_n,))
        return k_p[0]

    # Compute the total energy density given k_n as sum of component energy densities
    def compute_energy_density_npe(k_n):
        # Find k_p given k_n
        k_p = compute_k_p(k_n)

        # Find x for each component
        x_e = k_p / m_e / c
        x_p = k_p / m_p / c
        x_n = k_n / m_n / c

        # Find e for each component
        e_e = integral_energy_density(x_e, e_0_e)
        e_n = integral_energy_density(x_n, e_0_n)
        e_p = integral_energy_density(x_p, e_0_p)

        return e_n + e_p + e_e

    # Compute the total pressure given k_n as sum of component pressures (Pascal's principle)
    def compute_pressure_npe(k_n):
        # Find k_p given k_n
        k_p = compute_k_p(k_n)

        # Find x for each component
        x_e = k_p / m_e / c
        x_p = k_p / m_p / c
        x_n = k_n / m_n / c

        # Find p for each component
        p_e = integral_pressure(x_e, e_0_e)
        p_n = integral_pressure(x_n, e_0_n)
        p_p = integral_pressure(x_p, e_0_p)

        return p_n + p_e + p_p

    # Separate the case in which there are no neutrons
    def compute_energy_density_pe(k_p):
        # Find x for each component
        x_e = k_p / m_e / c
        x_p = k_p / m_p / c

        # Find e for each component
        e_e = integral_energy_density(x_e, e_0_e)
        e_p = integral_energy_density(x_p, e_0_p)

        return e_p + e_e

    def compute_pressure_pe(k_p):
        # Find x for each component
        x_e = k_p / m_e / c
        x_p = k_p / m_p / c

        # Find p for each component
        p_e = integral_pressure(x_e, e_0_e)
        p_p = integral_pressure(x_p, e_0_p)

        return p_e + p_p

    # Compute the Fermi momentum given the number density
    def k(n):
        return hbar * (3 * np.pi**2 * n) ** (1 / 3)

    # Range of number densities that we consider
    k_range = [k(n) for n in range_number_densities]

    # Compute pressure and energy for where there are all three components
    pressures_npe_above = [compute_pressure_npe(k) for k in k_range]
    energy_densities_npe_above = [compute_energy_density_npe(k) for k in k_range]

    # Compute pressure and energy for where there are no neutrons
    pressures_pe_below = [compute_pressure_pe(k) for k in k_range]
    energy_densities_pe_below = [compute_energy_density_pe(k) for k in k_range]

    # Combine the above and below limit pressure 3.038e24
    energy_densities_npe = []
    pressures_npe = []
    for i in range(len(pressures_pe_below)):
        if pressures_pe_below[i] < 3.038e24:
            p = pressures_pe_below[i]
            pressures_npe.append(p)
            e = energy_densities_pe_below[i]
            energy_densities_npe.append(e)

    for i in range(len(pressures_npe_above)):
        if pressures_npe_above[i] > 3.038e24:
            p = pressures_npe_above[i]
            pressures_npe.append(p)
            e = energy_densities_npe_above[i]
            energy_densities_npe.append(e)

    return pressures_npe, energy_densities_npe


# Empirical interactions

# Constants
A = -118.2 * MeV
B = 65.39 * MeV
sigma = 2.112

E_0 = 22.1 * MeV
n_0 = 0.16 / fm**3
S_0 = 30 * MeV


def emp(range_initial_pressures):
    # Compute pressure as function of u = n / n_0 and given pressure
    def compute_pressure_emp(u, p):
        return (
            n_0
            * (
                2 / 3 * E_0 * u ** (5 / 3)
                + A / 2 * u**2
                + B * sigma / (sigma + 1) * u ** (sigma + 1)
            )
            + n_0
            * ((2 ** (2 / 3) - 1) * E_0 * (2 / 3 * u ** (5 / 3) - u**2) + S_0 * u**2)
            - p
        )

    # Compute energy density as function of u = n / n_0
    def compute_energy_density_emp(u):
        return (
            n_0
            * u
            * (
                m_n * c**2
                + 2 ** (2 / 3) * E_0 * u ** (2 / 3)
                + A / 2 * u
                + B * u**sigma / (sigma + 1)
                + u * (S_0 - (2 ** (2 / 3) - 1) * E_0)
            )
        )

    # Compute u as a function of pressure using fsolve algorithm
    def compute_u_emp(p):
        u = opt.fsolve(compute_pressure_emp, 5, args=(p,))
        return u[0]

    # Given a desidered pressure, return the corresponding energy density
    def eos_emp(p):
        u = compute_u_emp(p)
        e = compute_energy_density_emp(u)
        return e

    energy_density_emp = [eos_emp(p) for p in range_initial_pressures]

    return energy_density_emp


# Skyrme Hatree-Fock interactions

# Constants
t_0 = 1024.1 * MeV * fm**3
t_3 = 14600.8 * MeV * fm**6


def sky(range_initial_pressures):
    # Compute pressure as function of u = n / n_0 and given pressure
    def compute_pressure_sky(u, p):
        n = u * n_0
        return (
            2 / 10 / m_n * (3 * np.pi**2 * hbar**3) ** (2 / 3) * n ** (5 / 3)
            + t_3 / 12 * n**3
            - t_0 / 4 * n**2
            - p
        )

    # Compute energy density as function of u = n / n_0
    def compute_energy_density_sky(u):
        n = u * n_0
        return (
            m_n * n * c**2
            + 3 / 10 / m_n * (3 * np.pi**2 * hbar**3) ** (2 / 3) * n ** (5 / 3)
            + t_3 / 24 * n**3
            - t_0 / 4 * n**2
        )

    # Compute u as a function of pressure using fsolve algorithm
    def compute_u_sky(p):
        u = opt.toms748(compute_pressure_sky, 1e-21, 1e5, args=(p,))
        return u

    # Given a desidered pressure, return the corresponding energy density
    def eos_sky(p):
        u = compute_u_sky(p)
        e = compute_energy_density_sky(u)
        return e

    energy_density_sky = [eos_sky(p) for p in range_initial_pressures]

    return energy_density_sky


# plt.plot(range_initial_pressures, energy_densities_nonrel, label="fermi_nonrel")
# plt.plot(range_initial_pressures, energy_densities_gen, label="fermi_gen")
# plt.plot(pressures_npe, energy_densities_npe, label="fermi_npe")
# plt.ylabel("Energy density ($erg / cm^3$)")
# plt.yscale("log")
# plt.xlabel("Pressure ($dyne / cm^2$)")
# plt.xscale("log")
# plt.legend()
# plt.savefig("code/plot/ideal_eos_n.pdf")
# plt.show()


# Plot
e_fermi_nonrel = fermi_nonrel(range_initial_pressures)
e_fermi_gen = fermi_gen(range_initial_pressures)
p_fermi_npe, e_fermi_npe = fermi_npe(range_number_densities)
e_emp = emp(range_initial_pressures)
e_sky = sky(range_initial_pressures)

plt.plot(range_initial_pressures, e_fermi_nonrel, label="fermi_nonrel")
plt.plot(range_initial_pressures, e_fermi_gen, label="fermi_gen")
plt.plot(p_fermi_npe, e_fermi_npe, label="fermi_npe")

plt.plot(range_initial_pressures, e_emp, label="emp")
plt.plot(range_initial_pressures, e_sky, label="sky")

plt.ylabel("Energy density ($erg / cm^3$)")
plt.yscale("log")
plt.xlabel("Pressure ($dyne / cm^2$)")
plt.xscale("log")
plt.legend()
plt.show()


# Save into file csv
with open("src/data/fermi_nonrel.csv", "w") as file:
    for i in range(len(range_initial_pressures)):
        file.write(f"0,{range_initial_pressures[i]},{e_fermi_nonrel[i]}\n")

with open("src/data/fermi_gen.csv", "w") as file:
    for i in range(len(range_initial_pressures)):
        file.write(f"0,{range_initial_pressures[i]},{e_fermi_gen[i]}\n")

with open("src/data/fermi_npe.csv", "w") as file:
    for i in range(len(p_fermi_npe)):
        file.write(f"0,{p_fermi_npe[i]},{e_fermi_npe[i]}\n")


with open("src/data/emp.csv", "w") as file:
    for i in range(len(range_initial_pressures)):
        file.write(f"0,{range_initial_pressures[i]},{e_emp[i]}\n")

with open("src/data/sky.csv", "w") as file:
    for i in range(len(range_initial_pressures)):
        file.write(f"0,{range_initial_pressures[i]},{e_sky[i]}\n")
