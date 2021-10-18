# -*- coding: utf-8 -*-
from firedrake import *
from deflatedbarrier import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
from petsc4py import PETSc
import petsc4py
petsc4py.PETSc.Sys.popErrorHandler()

doublepipe_th = __import__("double-pipe-th")
doublepipe_bdm = __import__("double-pipe-bdm")

"""
This script loads the previous level solution of the BDM and Taylor-Hood discretizations
prolongs the solution onto the next level and then resolves the double-pipe topology
optimization problem using the coarse-level solutions as an initial guess.

At the end we create a table that displays the values of the L2 norm of the
divergence of the velocity.
"""

def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)


def prolong_solutions_BDM():
    problem = doublepipe_bdm.BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)

    width = 1.5
    N = 20 # mesh resolution
    nref = 6
    for base_ref in [2,3,4,5,6]:

        class DoublePipe(object):
            def mesh(self, comm):
                  distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
                  mesh = RectangleMesh(N, N, width, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)
                  self.mh = MeshHierarchy(mesh, nref, reorder=True, callbacks=(before,after))
                  mesh = self.mh[base_ref]
                  return mesh


            def initial_guesses(self, Z, params):
                """
                 Use as initial guess the constant rho that satisfies the integral constraint.
                 Solve the Stokes equations for the values of (u, p, p0).
                """
                comm = Z.comm
                Zc = FunctionSpace(self.mh[base_ref-1],self.Ze)
                base_dofs = Zc.dim()
                zc = Function(Zc)

                scratch = "output/BDM-N-20-nref-%s-output/"%(base_ref-1)
                soln = "mu-0.000000000000e+00-dofs-%s-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/"%base_dofs
                h5 = HDF5File(scratch + soln + "0.xml.gz", "r", comm=comm)
                h5.read(zc, "/guess")
                del h5

                z0 = Function(Z)
                prolong(zc, z0)

                h5 = HDF5File(scratch + soln + "1.xml.gz", "r", comm=comm)
                h5.read(zc, "/guess")
                del h5

                z1 = Function(Z)
                prolong(zc, z1)
                return [z0, z1]

            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = DoublePipe()
        saving_folder = "output/BDM-N-%s-nref-%s-"%(N,base_ref)
        deflatedbarrier(newproblem, params, mu_start= 0, max_halfstep = 0, saving_folder = saving_folder)

def prolong_solutions_TH():
    problem = doublepipe_th.BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)

    width = 1.5
    N = 20 # mesh resolution
    nref = 6
    for base_ref in [2,3,4,5]:

        class DoublePipe(object):
            def mesh(self, comm):
                  distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
                  mesh = RectangleMesh(N, N, width, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)
                  self.mh = MeshHierarchy(mesh, nref, reorder=True, callbacks=(before,after))
                  mesh = self.mh[base_ref]
                  return mesh


            def initial_guesses(self, Z, params):
                """
                 Use as initial guess the constant rho that satisfies the integral constraint.
                 Solve the Stokes equations for the values of (u, p, p0).
                """
                comm = Z.comm
                Zc = FunctionSpace(self.mh[base_ref-1],self.Ze)
                base_dofs = Zc.dim()
                zc = Function(Zc)

                scratch = "output/TH-N-20-nref-%s-output/"%(base_ref-1)
                soln = "mu-0.000000000000e+00-dofs-%s-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/"%base_dofs
                h5 = HDF5File(scratch + soln + "0.xml.gz", "r", comm=comm)
                h5.read(zc, "/guess")
                del h5

                z0 = Function(Z)
                prolong(zc, z0)

                h5 = HDF5File(scratch + soln + "1.xml.gz", "r", comm=comm)
                h5.read(zc, "/guess")
                del h5

                z1 = Function(Z)
                prolong(zc, z1)
                return [z0, z1]

            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = DoublePipe()
        saving_folder = "output/TH-N-%s-nref-%s-"%(N,base_ref)
        deflatedbarrier(newproblem, params, mu_start= 0, max_halfstep = 0, saving_folder = saving_folder)


# This method creates a table of the values of the L2 norm of the divergence
# of the velocity for both discretizations over all the levels

def create_table():
    table = np.array([[r'$h$', r'BDM-$\|\mathrm{div}(u_h)\|_{L^2(\Omega)}$', r'Taylor-Hood-$\|\mathrm{div}(u_h)\|_{L^2(\Omega)}$', r'BDM-$\|\mathrm{div}(u_h)\|_{L^2(\Omega)}$', r'Taylor-Hood-$\|\mathrm{div}(u_h)\|_{L^2(\Omega)}$']])

    N = 20
    nref = 6
    width = 1.5
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    mesh = RectangleMesh(N, N, width, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)
    mh = MeshHierarchy(mesh, nref, reorder=True, callbacks=(before,after))

    # BDM element
    Ve = FiniteElement("BDM", mesh.ufl_cell(), 1, variant = "integral") # velocity
    Pe = FiniteElement("DG", mesh.ufl_cell(), 0) # pressure
    Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
    Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals
    Ze_BDM = MixedElement([Ce, Ve, Pe, Re])

    # Taylor-Hood element
    Ve = VectorElement("CG", mesh.ufl_cell(), 2, dim=2) # velocity
    Pe = FiniteElement("CG", mesh.ufl_cell(), 1) # pressure
    Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
    Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals
    Ze_TH = MixedElement([Ce, Ve, Pe, Re])

    # Run through 5 meshes
    for base_ref in [1,2,3,4,5]:
        mesh = mh[base_ref]

        # Use heuristic to get mesh size
        hmin = sqrt((1.5/(N*2**base_ref))**2 + (1.0/(N*2**base_ref))**2)
        div_list = [hmin]

        Z_BDM  = FunctionSpace(mesh, Ze_BDM)
        Z_TH  = FunctionSpace(mesh, Ze_TH)
        comm = Z_BDM.comm

        # Locate files of stored solutions
        base_dofs_bdm = Z_BDM.dim()
        z_bdm = Function(Z_BDM)
        scratch_bdm = "output/BDM-N-20-nref-%s-output/"%(base_ref)
        soln_bdm = "mu-0.000000000000e+00-dofs-%s-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/"%base_dofs_bdm

        base_dofs_th = Z_TH.dim()
        z_th = Function(Z_TH)
        scratch_th = "output/TH-N-20-nref-%s-output/"%(base_ref)
        soln_th = "mu-0.000000000000e+00-dofs-%s-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/"%base_dofs_th


        # Load BDM branch 0
        h5 = HDF5File(scratch_bdm + soln_bdm + "0.xml.gz", "r", comm=comm)
        h5.read(z_bdm, "/guess")
        del h5
        div_list.append(sqrt(assemble(inner(div(z_bdm.split()[1]), div(z_bdm.split()[1]))*dx)))


        # Load Taylor-Hood branch 0
        h5 = HDF5File(scratch_th + soln_th + "0.xml.gz", "r", comm=comm)
        h5.read(z_th, "/guess")
        del h5
        div_list.append(sqrt(assemble(inner(div(z_th.split()[1]), div(z_th.split()[1]))*dx)))


        # Load BDM branch 1
        h5 = HDF5File(scratch_bdm + soln_bdm + "1.xml.gz", "r", comm=comm)
        h5.read(z_bdm, "/guess")
        del h5
        div_list.append(sqrt(assemble(inner(div(z_bdm.split()[1]), div(z_bdm.split()[1]))*dx)))


        # Load Taylor-Hood branch 1
        h5 = HDF5File(scratch_th + soln_th + "1.xml.gz", "r", comm=comm)
        h5.read(z_th, "/guess")
        del h5
        div_list.append(sqrt(assemble(inner(div(z_th.split()[1]), div(z_th.split()[1]))*dx)))


        out = np.array(div_list)
        table = np.append(table, [out], axis = 0)

    # Create table with L^2-norm of the divergence term
    fig, ax = plt.subplots()
    columns = ('', r'Straight channels','', r'Double-ended wrench','')
    the_table = plt.table(cellText=table, colLabels=columns, loc='top')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(18)
    the_table.scale(4, 4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.xlabel(r"$L^2(\Omega)$-norm of the divergence of the velocity", fontsize = 30)
    plt.savefig("table_divergence.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()


prolong_solutions_BDM()
prolong_solutions_TH()
create_table()
