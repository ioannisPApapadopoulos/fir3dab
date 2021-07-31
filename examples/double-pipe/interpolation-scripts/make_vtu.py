import numpy
from firedrake import *
from fir3dab import info_green


"""
This script takes the already saved solutions in a .xml.gz file, interpolates them to
DG spaces and saves them as pvd files.

We do this as it allows us to p-sequence the solutions on a finer mesh using a number of cores,
even if the original coarse-mesh solutions were saved on a different number of cores.
"""

comm = COMM_WORLD
N = 50

# Create base mesh
def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
meshg = RectangleMesh(N, N, 1.5, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)
mh = MeshHierarchy(meshg, 1, reorder=True, callbacks=(before,after))
meshg = mh[-1]

for porder in range(2,5):
    # Function spaces to save the solution in
    Ve = VectorElement("DG", meshg.ufl_cell(), porder, dim=2) # velocity
    Pe = FiniteElement("DG", meshg.ufl_cell(), 0) # pressure
    Ce = FiniteElement("DG", meshg.ufl_cell(), 0) # control
    Re = FiniteElement("R",  meshg.ufl_cell(), 0) # reals
    Ze = MixedElement([Ce, Ve, Pe, Re])
    Zf  = FunctionSpace(meshg, Ze)


    # Function spaces of the solutions already saved.
    Ve = FiniteElement("BDM", meshg.ufl_cell(), 1, variant="integral") # velocity
    Pe = FiniteElement("DG", meshg.ufl_cell(), 0) # pressure
    Ce = FiniteElement("DG", meshg.ufl_cell(), 0) # control
    Re = FiniteElement("R",  meshg.ufl_cell(), 0) # reals
    Ze = MixedElement([Ce, Ve, Pe, Re])
    Zc = FunctionSpace(meshg, Ze)

    scratch = "../aL2-N-50-nref-1-output/"
    soln = "mu-0.000000000000e+00-dofs-100401-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/"

    for branch in range(2):
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
        File("pvd/rho-%s-%s.pvd"%(branch,porder)).write(rho)
        File("pvd/p-%s-%s.pvd"%(branch,porder)).write(p)
        File("pvd/u-%s-%s.pvd"%(branch,porder)).write(u)
