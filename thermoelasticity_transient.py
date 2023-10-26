# import fenicsx modules
import basix
import dolfinx 
import ufl 
from dolfinx.fem import petsc

# import other modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from mpi4py import MPI
from petsc4py import PETSc

# MPI
mesh_comm = MPI.COMM_WORLD

# Geometry parameters
Length = 1.
Radius = 0.1
nLength = 50  # mesh density
nRadius = 10 

#domain = Rectangle(Point(0., 0.), Point(L, L)) - Circle(Point(0., 0.), R, 100)
#msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.quadrilateral)
msh = dolfinx.mesh.create_rectangle(comm=mesh_comm, points=( (0.0, 0.0), (Length,Radius) ), n=(nLength,nRadius), cell_type=dolfinx.mesh.CellType.quadrilateral)

# Conditions
TZero = constants.zero_Celsius
T0 = dolfinx.fem.Constant(msh, (TZero+20.))
DThole = 10.

# Material Properties
E = 70e3
nu = 0.3
lmbda = E*nu/((1+nu)*(1-2*nu))
mu = E/2/(1+nu)
rho = 2700.     # density
alpha = 2.31e-5  # thermal expansion coefficient
kappa = dolfinx.fem.Constant(msh, alpha*(2*mu + 3*lmbda))
cV = dolfinx.fem.Constant(msh, 910e-6*rho)  # specific heat per unit volume at constant strain
k = dolfinx.fem.Constant(msh, 237e-6)  # thermal conductivity


# Elements and functions spaces
k = 2
V_ue = basix.ufl.element("Lagrange", msh.basix_cell(), k, shape=(msh.geometry.dim,) ) # displacement finite element
V_te = basix.ufl.element("Lagrange", msh.basix_cell(), k-1) # temperature finite element
V_mi = basix.ufl.mixed_element([V_ue, V_te])
V = dolfinx.fem.functionspace(msh, V_mi)


# Test and trial functions
(du, dTheta) = ufl.TrialFunctions(V)
(u_, Theta_) = ufl.TestFunctions(V)
vector, _ = V.sub(0).collapse()
uold = dolfinx.fem.Function(vector)
scalar, _ = V.sub(1).collapse()
Thetaold = dolfinx.fem.Function(scalar)

# define eps and sigma the 
def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v, Theta):
    return (lmbda*ufl.tr(eps(v)) - kappa*Theta)*ufl.Identity(2) + 2*mu*eps(v)
# Setting up the matrix
dt = dolfinx.fem.Constant(msh, 0.)
mech_form = ufl.inner(sigma(du, dTheta), eps(u_))*ufl.dx

therm_form = (cV*(dTheta-Thetaold)/dt*Theta_ +
              kappa*T0*ufl.tr(eps(du-uold))/dt*Theta_ +
              ufl.dot(k*ufl.grad(dTheta), ufl.grad(Theta_)))*ufl.dx

form = mech_form + therm_form

a, L = ufl.lhs(form), ufl.rhs(form)

# 
def inner_boundary(x):
    return np.isclose(x[0]**2+x[1]**2, R**2)
def bottom(x):
    return np.isclose(x[1], 0)
def left(x):
    return np.isclose(x[0], 0)

# Boundary conditions - Left - Displacement and temperature
fdim = msh.topology.dim - 1
facets_left = dolfinx.mesh.locate_entities_boundary(msh, fdim, marker=lambda x: np.isclose(x[0], 0.0))
dofs_leftVector = dolfinx.fem.locate_dofs_topological((V.sub(0), vector), fdim, facets_left)
dofs_leftScalar = dolfinx.fem.locate_dofs_topological((V.sub(1), scalar), fdim, facets_left)

def fe1(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = 0.0
    return values


fe_x1 = dolfinx.fem.Function(vector)
fe_x1.interpolate(fe1)
bc_leftVector = dolfinx.fem.dirichletbc( fe_x1, dofs_leftVector, vector.sub(0) )

def ft1(x):
    values = np.zeros(x.shape[0])
    values[1, :] = DThole
    return values


ft_x1 = dolfinx.fem.Function(scalar)
with ft_x1.vector.localForm() as loc:
    loc.set(DThole)
#ft_x1.interpolate(ft1)
#ft_x1 = np.array([10.0], dtype=PETSc.ScalarType)   # type: ignore
bc_leftScalar = dolfinx.fem.dirichletbc( ft_x1, dofs_leftScalar)


# Boundary conditions - Bottom - Displacement
facets_bottom = dolfinx.mesh.locate_entities_boundary(msh, fdim, marker=lambda x: np.isclose(x[1], 0.0))
dofs_bottom = dolfinx.fem.locate_dofs_topological((V.sub(0), vector), fdim, facets_bottom)

def fe2(x):
    values = np.zeros((2, x.shape[0]))
    values[1, :] = 0.0
    return values


fe_x2 = dolfinx.fem.Function(vector)
fe_x2.interpolate(fe2)
bc_bottom = dolfinx.fem.dirichletbc( fe_x2, dofs_bottom, vector.sub(1) ) 

# Boundary conditions - Top - Displacement
facets_top = dolfinx.mesh.locate_entities_boundary(msh, fdim, marker=lambda x: np.isclose(x[1], Radius))
dofs_top = dolfinx.fem.locate_dofs_topological((V.sub(0), vector), fdim, facets_top)

def fe3(x):
    values = np.zeros((2, x.shape[0]))
    values[1, :] = 0.0
    return values


fe_x3 = dolfinx.fem.Function(vector)
fe_x3.interpolate(fe3)
bc_top = dolfinx.fem.dirichletbc( fe_x3, dofs_top, vector.sub(1) ) 

bcs = [bc_leftVector, bc_leftScalar, bc_bottom, bc_top]




problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


Nincr = 100
t = np.logspace(1, 4, Nincr+1)
Nx = nLength
x = ufl.SpatialCoordinate(msh)
T_res = np.zeros((Nx, Nincr+1))
U = dolfinx.fem.Function(V)
#u.name = "stress"
#Theta.name = "temperature"

xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results_theromelasticity.xdmf","w")
xdmf.write_mesh(msh)

for (i, dti) in enumerate(np.diff(t)):
    print("Increment " + str(i+1)+str(' ')+str(t[i]))
    dt.value = dti
    Esh = problem.solve()
    uold.value = U
    u, Theta = ufl.split(Esh)
    xdmf.write_function(u,dti)
    xdmf.write_function(Theta,dti)

    #T_res[:, i+1] = [U(xi, 0.0)[2] for xi in x]

'''
plt.figure()
plt.plot(Nx,sigma(u, Theta)[1, 1], title="$\sigma_{yy}$ stress near the hole")
plt.xlim((0, 3*Radius))
plt.ylim((0, 3*Radius))
plt.colorbar(p)
plt.show()

plt.figure()
plt.plot(Nx,Theta, title="Temperature variation")
plt.xlim((0, Length))
plt.ylim((TZero,T0))
plt.colorbar(p)
plt.show()
'''



