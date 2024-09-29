In this repository, you will find a project in python which studies neutron stars by solving the system of ordinary differential equations, composed by a mass equation
```math
 \frac{dm}{dr} = \frac{4 \pi r^2 \epsilon(r)}{r} ~,
```
and a pressure equation
```math
\quad \frac{d p}{dr} = \frac{G \epsilon(r) m(r)}{c^2 r^2} ~, \qquad \frac{d p}{dr} = \frac{G \epsilon(r) m(r)}{c^2 r^2} \Big ( 1 + \frac{p(r)}{\epsilon(r)} \Big ) \Big ( 1 + \frac{4 \pi r^3 p(r)}{m(r)c^2} \Big ) \Big ( 1 - \frac{2 G m(r)}{c^2 r} \Big) ~,
```
where the first one is calculated in Newtonian regime and the second one using general relativity. The boundary conditions are
```math
p(r = R_{object})= 0 ~, \quad p(r = 0) = p_0 ~, \quad m(r = 0) = 0 ~.
```

Solving the system means to find physical quantities (like mass, radius, gravitational redshift or moment of inertia) given a central pressure $p_0$ of the neutron star.

The code to solve this system given an equation of state (energy density as a function of pressure) can be found in 
[solver.py](https://github.com/PhysicsZandi/NeutronStars/blob/main/src/solver.py).

We have considered 2 kinds of equations of state:
1. Degenerate ideal Fermi gas of neutrons, with the addition of protons and electrons or interactions (empirical or Skyrme ones). The code to generate such equations of state can be found in [fermi_eos.py](https://github.com/PhysicsZandi/NeutronStars/blob/main/src/fermi_eos.py) and examples with plots in [fermi_eos_examples.ipynb](https://github.com/PhysicsZandi/NeutronStars/blob/main/src/fermi_eos_examples.ipynb).

![alt text](https://github.com/PhysicsZandi/NeutronStars/blob/main/src/fermi_eos.png)

2. Realistic equations of state in the archive [CompOSE](https://compose.obspm.fr). The code to study such equations of state can be found in [compose.py](https://github.com/PhysicsZandi/NeutronStars/blob/main/src/compose.py) and examples with plots in [compose_examples.ipynb](https://github.com/PhysicsZandi/NeutronStars/blob/main/src/compose_examples.ipynb).

![alt text](https://github.com/PhysicsZandi/NeutronStars/blob/main/src/compose.png)


The documentation of the project is in [documentation](https://github.com/PhysicsZandi/NeutronStars/tree/main/src/documentation).