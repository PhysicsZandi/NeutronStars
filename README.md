In this repository, you will find a project in python which studies neutron stars by solving the system of ordinary differential equations, composed by a mass equation
$$ \frac{dm}{dr} = \frac{4 \pi r^2 \epsilon(r)}{r} $$
and a pressure equation
$$ \quad \frac{d p}{dr} = \frac{G \epsilon(r) m(r)}{c^2 r^2} \qquad \frac{d p}{dr} = \frac{G \epsilon(r) m(r)}{c^2 r^2} \Big ( 1 + \frac{p(r)}{\epsilon(r)} \Big ) \Big ( 1 + \frac{4 \pi r^3 p(r)}{m(r)c^2} \Big ) \Big ( 1 - \frac{2 G m(r)}{c^2 r} \Big) ~,$$
where the first one is calculated in Newtonian regime and the second one using general relativity,