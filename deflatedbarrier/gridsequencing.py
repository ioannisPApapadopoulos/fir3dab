# -*- coding: utf-8 -*-
from firedrake import *
from firedrake.petsc import PETSc
from .deflation import defsolve
from .compatibility import make_comm
from .drivers import deflatedbarrier
from .mlogging import *
import os
import resource
import shutil



def gridsequencing(problem, sharpness_coefficient, branches, params=None, pathfile = None,
                   initialpathfile = None, comm=COMM_WORLD,
                   mu_start_refine = 0.0, mu_start_continuation = 0.0, param_end = 5e-3,
                   iters_total = 20, parameter_update = None, greyness_tol = 1e-1,
                   grid_refinement = 10, parameter_continuation = True):

    # FIXME refinement of mesh does not work yet, no equivalent of refine in Firedrake
    if grid_refinement > 0:
        info_red(r"Mesh refinement not working yet, no equivalent of refine in Firedrake")
        grid_refinement = 0

    mu = Constant(0.0)
    epsilon_original = params[sharpness_coefficient]
    if pathfile == None: pathfile = "gs_output"

    for branch in branches:
        # Initialise FEM 
        params[sharpness_coefficient] = epsilon_original
        mesh = problem.mesh(comm)
        gsproblem = GridSequenceProblem(problem, mesh)
        pvd = File(pathfile+"/paraview/refine-branch-%s.pvd"%branch)
        initialstring = pathfile + "/tmp/%s.xml.gz"%(branch)
        tmppathfile = pathfile + "/tmp"
        
        # Formulate problem for initial check of starting solution
        (nvs_cont, vi, dofs, z) = requirements(mesh, problem, mu, params)
        if initialpathfile == None:
            h5 = HDF5File("output/mu-%.12e-dofs-%s-params-%s-solver-BensonMunson/%s.xml.gz" % (float(mu), dofs, params , branch), "r", comm=comm)
        else:
            h5 = HDF5File(initialpathfile, "r", comm=comm)
        h5.read(z, "/guess"); del h5

        epsilon = params[sharpness_coefficient]
        info_blue(r"Checking solution to initial grid for branch %s, sharpness coefficient = %s"%(branch,epsilon))
        (success,_,_) = defsolve(nvs_cont, deflateop = None, vi=vi)
        if success:
            save_pvd(pvd, z, mu)
        else:
            info_red(r"Solution not found")
            break

        iters = 0
        gr = 1
        while (epsilon > param_end) and (iters < iters_total):
            
            if gr <= grid_refinement:
                mesh_ = interface_refine(z, mesh, greyness_tol)
                Z_ = problem.function_space(mesh_)
                z_ = Function(Z_)
                info_blue(r"Degrees of freedom: %s" %Z_.dim())
            else:
                mesh_ = mesh
                z_ = z

            gsproblem = GridSequenceProblem(problem, mesh_)

            # solve for refined grid before updating epsilon
            if gr <= grid_refinement:
                gr += 1
                info_blue(r"Refining grid for branch %s, sharpness coefficient = %s"%(branch,epsilon))
                prolong(z,z_)
                exists = os.path.isfile(tmppathfile +"/%s.xml.gz"%branch)
                if exists: os.remove(tmppathfile +"/%s.xml.gz"%branch)
                problem.save_solution(comm,z_,branch,tmppathfile)
                ([z_],_) = deflatedbarrier(gsproblem, params, mu_start=mu_start_refine, mu_end = 1e-10, max_halfstep = 1, initialstring = initialstring)
                problem.save_pvd(pvd, z_, mu)
            # newton(F, J, z_, bcs, params, sp, None, None, None, vi)

            if parameter_continuation == True:
                if parameter_update == None:
                    raise("Require rules for update in continuation parameter")
                epsilon = parameter_update(epsilon, z_)
                params[sharpness_coefficient] = epsilon
                info_blue(r"Solve for new sharpness coefficient = %s for branch %s"%(epsilon, branch))
                exists = os.path.isfile(tmppathfile +"/%s.xml.gz"%branch)
                if exists: os.remove(tmppathfile +"/%s.xml.gz"%branch)
                problem.save_solution(comm,z_,branch,tmppathfile)
                ([z_],_) = deflatedbarrier(gsproblem, params, mu_start=mu_start_continuation, mu_end = 1e-10, max_halfstep = 0, initialstring = initialstring)

            z = z_
            mesh = mesh_
            problem.save_pvd(pvd, z, mu)
            problem.save_solution(comm,z,branch,pathfile+"/cont-%s/branch-%s/iter-%s"%(epsilon,branch,iters))
            #vwr = PETSc.Viewer().createHDF5(pathfile+"/cont-%s/branch-%s/iter-%s/mesh.h5"%(epsilon, branch,iters), "w", comm=comm)
            #dm = mesh._plex
            #dm.view(vwr); del vwr
            iters += 1
        # shutil.rmtree("gs_output/tmp", ignore_errors=True)
        info_blue(r"Reached target refinement, terminating algorithm")

    shutil.rmtree(tmppathfile, ignore_errors=True)
    return None

def requirements(mesh, problem, mu, params): 
    Z = problem.function_space(mesh)
    dofs = Z.dim()
    z = Function(Z, name = "Solution")
    (lb, ub)  = problem.bounds(mesh, mu, params)
    vi = problem.bounds_vi(Z, mu, params)
    nref = 0
    nvs_cont = problem.solver(z, lb, ub, mu, nref, params, "ContinuationTask")
    return (nvs_cont,vi, dofs, z)

def interface_refine(z, mesh, greyness_tol):
    #rho = z.split()[0]

    #for i in range(1):
    #    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    #    cell_markers.set_all(False)
    #    for cell in cells(mesh):
    #        p = cell.midpoint()
    #        if greyness_tol < rho(p) < 1. - greyness_tol:
    #            cell_markers[cell] = True
    #    mesh = refine(mesh, cell_markers)

    return mesh

def prolong(z,z_):
    subz  = z.split()
    subz_ = z_.split()

    for (u, u_) in zip(subz, subz_):
        ele = u.function_space().ufl_element()
        if ele.family() == "Real":
            with u_.dat.vec_wo as dest, u.dat.vec_ro as src:
                src.copy(dest)
        else:
            # is there a better way to do this?
            u_.project(u)

    for (i, u_) in enumerate(subz_):
        z_.sub(i).assign(u_)

class GridSequenceProblem(object):
    def __init__(self, problem, mesh_):
        self.mesh_ = mesh_
        self.problem = problem
    def mesh(self, comm):
        return self.mesh_
    def save_solution(self, comm, z, branch, pathfile):
        pass
    def save_pvd(self, pvd, u, mu):
        pass
    def number_solutions(self, mu, params):
        return 1
    def __getattr__(self, attr):
        return getattr(self.problem, attr)
