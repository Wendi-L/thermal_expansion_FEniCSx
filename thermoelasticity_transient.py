from dolfinx import *
from ufl import *
from basix.ufl import element, mixed_element
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

L = 1.
R = 0.1
N = 50  # mesh density

#domain = Rectangle(Point(0., 0.), Point(L, L)) - Circle(Point(0., 0.), R, 100)
msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)

T0 = fem.Constant(msh, 293.)
DThole = fem.Constant(msh, 10.)
E = 70e3
nu = 0.3
lmbda = E*nu/((1+nu)*(1-2*nu))
mu = E/2/(1+nu)
rho = 2700.     # density
alpha = 2.31e-5  # thermal expansion coefficient
kappa = fem.Constant(msh, alpha*(2*mu + 3*lmbda))
cV = fem.Constant(msh, 910e-6*rho)  # specific heat per unit volume at constant strain
k = fem.Constant(msh, 237e-6)  # thermal conductivity


k = 2
Vue = element('CG', msh.basix_cell(), k) # displacement finite element
Vte = element('CG', msh.basix_cell(), k-1) # temperature finite element
V = fem.FunctionSpace(msh, mixed_element([Vue, Vte]))

def inner_boundary(x):
    return np.isclose(x[0]**2+x[1]**2, R**2)
def bottom(x):
    return np.isclose(x[1], 0)
def left(x):
    return np.isclose(x[0], 0)

fdim = msh.topology.dim - 1
facets_inner = mesh.locate_entities_boundary(msh, fdim, marker=lambda x: np.isclose(x[0]**2+x[1]**2, R**2))
Q, _ = V.sub(1).collapse()
dofs_inner = fem.locate_dofs_topological((V.sub(1), Q), fdim, facets_inner)

facets_bottom = mesh.locate_entities_boundary(msh, fdim, marker=lambda x: np.isclose(x[1], 0))
P, _ = V.sub(0).collapse()
dofs_bottom = fem.locate_dofs_topological((V.sub(0), P), fdim, facets_bottom)

facets_left = mesh.locate_entities_boundary(msh, fdim, marker=lambda x: np.isclose(x[0], 0))
dofs_left = fem.locate_dofs_topological((V.sub(0), P), fdim, facets_left)

bc1 = DirichletBC(0.0, dofs_bottom, V.sub(0).sub(1)) 
bc2 = DirichletBC(0.0, dofs_left, V.sub(0).sub(0))
bc3 = DirichletBC(DThole, dofs_inner, V.sub(1))
bcs = [bc1, bc2, bc3]

(u_, Theta_) = TestFunctions(V)
(du, dTheta) = TrialFunctions(V)
(uold, Thetaold) = Function(V)

def eps(v):
    return sym(grad(v))


def sigma(v, Theta):
    return (lmbda*tr(eps(v)) - kappa*Theta)*Identity(2) + 2*mu*eps(v)


dt = fem.Constant(msh, 0.)
mech_form = inner(sigma(du, dTheta), eps(u_))*dx
therm_form = (cV*(dTheta-Thetaold)/dt*Theta_ +
              kappa*T0*ufl.tr(eps(du-uold))/dt*Theta_ +
              ufl.dot(k*ufl.grad(dTheta), ufl.grad(Theta_)))*dx
form = mech_form + therm_form
a, L = ufl.lhs(form), ufl.rhs(form)
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


Nincr = 100
t = np.logspace(1, 4, Nincr+1)
Nx = 100
x = np.linspace(R, L, Nx)
T_res = np.zeros((Nx, Nincr+1))
U = Function(V)
for (i, dti) in enumerate(np.diff(t)):
    print("Increment " + str(i+1))
    dt.value = dti
    Esh = problem.solve()
    Uold.value = U
    T_res[:, i+1] = [U(xi, 0.)[2] for xi in x]


u, Theta = split(U)
plt.figure()
p = plot(sigma(u, Theta)[1, 1], title="$\sigma_{yy}$ stress near the hole")
plt.xlim((0, 3*R))
plt.ylim((0, 3*R))
plt.colorbar(p)
plt.show()

plt.figure()
p = plot(Theta, title="Temperature variation")
plt.xlim((0, 3*R))
plt.ylim((0, 3*R))
plt.colorbar(p)
plt.show()




