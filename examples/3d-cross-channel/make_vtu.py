import numpy
from firedrake import *
from fir3dab import info_green

"""
This script takes the already saved solutions in a .xml.gz file, interpolates them to
DG spaces and saves them as pvd files.

We do this as it allows us to grid-sequence the solutions on a finer mesh using a number of cores,
even if the original coarse-mesh solutions were saved using a different number of cores.
"""

comm = COMM_WORLD
width = 1. # aspect ratio

Ng = 20
meshg = BoxMesh(Ng, Ng, Ng, width, 1.0, 1.0, comm=comm)

# Function spaces to save the solution in
Ve = VectorElement("DG", meshg.ufl_cell(), 1, dim=3) # velocity
Pe = FiniteElement("DG", meshg.ufl_cell(), 0) # pressure
Ce = FiniteElement("DG", meshg.ufl_cell(), 0) # control
Re = FiniteElement("R",  meshg.ufl_cell(), 0) # reals
Ze = MixedElement([Ce, Ve, Pe, Re])
Zf  = FunctionSpace(meshg, Ze)

# Function spaces of the solutions already saved.
Ve = FiniteElement("BDM", meshg.ufl_cell(), 1) # velocity
Pe = FiniteElement("DG", meshg.ufl_cell(), 0) # pressure
Ce = FiniteElement("DG", meshg.ufl_cell(), 0) # control
Re = FiniteElement("R",  meshg.ufl_cell(), 0) # reals
Ze = MixedElement([Ce, Ve, Pe, Re])
Zc = FunctionSpace(meshg, Ze)
scratch = "output/"
soln = "mu-0.000000000000e+00-dofs-391201-params-[0.1, 25000.0, 0.1]-solver-BensonMunson/"

# For all the branches, load the solution, interpolate, and save them as pvd files.
for branch in range(3):
    zc = Function(Zc)
    h5 = HDF5File(scratch+soln+"%s.xml.gz"%branch, "r", comm=comm)
    h5.read(zc, "/guess")
    del h5

    zf = Function(Zf)
    zf.split()[0].assign(zc.split()[0])
    zf.split()[1].interpolate(zc.split()[1])
    zf.split()[2].assign(zc.split()[2])

    (rho, u, p, l) = zf.split()
    info_green("lmbda-value-branch-%s = %s"%(branch,comm.allgather(zc.split()[3].vector().get_local())[0]))

    rho.rename("Solution")
    u.rename("Solution")
    p.rename("Solution")
    File("pvd/rho-%s.pvd"%branch).write(rho)
    File("pvd/p-%s.pvd"%branch).write(p)
    File("pvd/u-%s.pvd"%branch).write(u)
