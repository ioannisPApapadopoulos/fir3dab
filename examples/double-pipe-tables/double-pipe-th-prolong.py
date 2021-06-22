# -*- coding: utf-8 -*-
from firedrake import *
from deflatedbarrier import *

delta = 1.5 # aspect ratio
N = 20 # mesh resolution
nref = 6
base_ref = 1

def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

def InflowOutflow(mesh):
    """
    Smoothed out BCs which guarantee the analytical solutions for the velocity
    and pressure live in (H^2)^2 x H^1
    """
    x = SpatialCoordinate(mesh)
    l = 1.0/6.0
    gbar = 1.0

    x_on_boundary = Or(lt(x[0], 1e-10), gt(x[0], delta-1e-10))
    
    y_in_first_pipe = And(gt(x[1], 1/4 - l/2), lt(x[1], 1/4 + l/2)) 
    val_in_first_pipe = exp(gbar - gbar/(1.-(12*x[1] - 3.)**2))
    
    y_in_second_pipe = And(gt(x[1], 3/4 - l/2), lt(x[1], 3/4 + l/2))
    val_in_second_pipe = exp(gbar - gbar/(1.-(12*x[1] - 9.)**2))

    x_component = conditional(x_on_boundary,
                              conditional(y_in_first_pipe, val_in_first_pipe,
                                          conditional(y_in_second_pipe, val_in_second_pipe,
                                                      Constant(0))),
                              Constant(0))
    y_component = Constant(0)
    return as_vector([x_component, y_component])

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
        mesh = RectangleMesh(N, N, delta, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)
        self.mh = MeshHierarchy(mesh, nref, reorder=True, callbacks=(before,after))
        mesh = self.mh[base_ref]
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", mesh.ufl_cell(), 2, dim=2) # velocity
        Pe = FiniteElement("CG", mesh.ufl_cell(), 1) # pressure
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals
        Ze = MixedElement([Ce, Ve, Pe, Re])
        self.Ze = Ze

        Z  = FunctionSpace(mesh, Ze)
        self.Z = Z
        info_blue("Number of degrees of freedom: ", Z.dim())
        # Take some data. First, BCs
        self.expr = InflowOutflow(mesh)
        Ge = MixedElement([Ve, Pe])
        self.G = FunctionSpace(mesh, Ge)
        self.Gbcs = [DirichletBC(self.G.sub(0), InflowOutflow(mesh), "on_boundary")]
        return Z


    def lagrangian(self, z, params):
        (rho, u, p, lmbda) = split(z)
        (gamma, alphabar, q) = params

        L = (
            + 0.5 * inner(grad(u), grad(u))*dx
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            - inner(lmbda, gamma - rho)*dx
            )

        return L

    def nullspace(self, Z, params):
        # Remove nullspace of pressure block
        nsp = MixedVectorSpaceBasis(Z, [Z.sub(0),
                                        Z.sub(1),
                                        VectorSpaceBasis(constant=True),
                                        Z.sub(3)]
                                    )
        return nsp

    def cost(self, z, params):
        rho, u, p, lmbda = split(z)
        L = (
              0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), self.expr, "on_boundary")]

    def number_initial_guesses(self, params):
        return 2


    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the Stokes equations for the values of (u, p, p0).
        """
        comm = Z.comm
        Zc = FunctionSpace(self.mh[base_ref-1],self.Ze)
        base_dofs = Zc.dim()
        zc = Function(Zc)

        scratch = "/scratch/papadopoulos/3d/double-pipe-analysis/TH-N-20-nref-%s-output/"%(base_ref-1)
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

    def number_solutions(self, mu, params):
        return 2

    def update_mu(self, z, mu, iters, k, k_mu_old, params):
        if float(mu) > 20:
            k_mu = 0.9
        else:
            k_mu = 0.8
        theta_mu = 1.2
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu


    def solver_parameters(self, mu, branch, task, params):
        (gamma,alphabar, q) = params
        if task == 'ContinuationTask':
            linesearch = "l2"
            max_it = 20
            damping = 1.0
        elif task == 'DeflationTask':
            linesearch = "l2"
            max_it = 100
            damping = 0.9
        elif task == 'PredictorTask':
            linesearch = "basic"
            max_it = 5
            damping = 1.0
        #snes_atol = 1e-6 if float(mu) > 0 else 1e-7 #7e-7
        snes_atol = 1e-5 if float(mu) > 0 else 1e-5 #7e-7
        args = {
                #"snes_view":None,
                "snes_max_it": max_it, #
                "snes_atol": snes_atol,
                "snes_stol": 1e-20,
                "snes_rtol": 1e-20,
                "snes_converged_reason": None,
                "snes_divergence_tolerance": 1e10,
                "snes_monitor": None,
                "snes_linesearch_type": linesearch,
                "snes_linesearch_monitor": None,
                "snes_linesearch_damping": damping,

                # fGMRES outermost solver
                "ksp_type": "fgmres",
                "ksp_norm_type": "unpreconditioned",
                "ksp_gmres_restart": 500,
                "ksp_gmres_modifiedgramschmidt": None,
                "ksp_monitor_true_residual": None,
                "ksp_converged_reason": None,
                "ksp_atol": 1e-7 if float(mu) > 0 else 1e-9,
                "ksp_rtol": 1e-7,

                # We want to do a fieldsplit to handle the R-block since firedrake
                # cannot assemble aij matrices with R-block
                "mat_type": "matfree",
                "mat_mumps_icntl_14": 500, # for some reason the coarse-grid LU MUMPS solve is grabbing it from here
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_0_fields": "0,1,2",
                "pc_fieldsplit_1_fields": "3",

                # field 1 is just an R-block, gmres should converge in 1 iteration
                "fieldsplit_1_ksp_type": "gmres",
                "fieldsplit_1_ksp_max_it": 1,
                "fieldsplit_1_ksp_convergence_test": "skip",
                "fieldsplit_1_pc_type": "none",

                # field 0 is the big guy, here we try and tackle it by more
                # Schur fieldsplits, First we take the material distribution
                # as the top left block

                "fieldsplit_0":{
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled":{
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps",
                        "mat_mumps_icntl_14": 500,},}
                }
        return args

    def alpha(self, rho, params):
        (gamma, alphabar,q) = params
        return alphabar * ( 1. - rho*(1+q)/(rho+q))

    def stokes(self, u, p, rho, params):
        """The Stokes functional, without constraints"""
        J = (
              inner(grad(u), grad(u))*dx
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        return J

    def bounds_vi(self, Z, mu, params):
        inf = 1e100
        lb = Function(Z); ub = Function(Z)
        lb.split()[0].interpolate(Constant(0))
        ub.split()[0].interpolate(Constant(1))
        lb.split()[1].vector().set_local(-inf)
        ub.split()[1].vector().set_local(+inf)
        for i in range(2,4):
            lb.split()[i].interpolate(Constant(-inf))
            ub.split()[i].interpolate(Constant(+inf))
        return (lb, ub)

    def save_pvd(self, pvd, z, mu):
        (rho_, u_, p_, lmbda_) = z.split()
        rho_.rename("Control")
        u_.rename("Velocity")
        p_.rename("Pressure")
        pvd.write(rho_, u_, p_)

    def volume_constraint(self, params):
        return params[0]

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return nothing(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)

if __name__ == "__main__":

    problem=BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
    saving_folder = "/scratch/papadopoulos/3d/double-pipe-analysis/TH-N-%s-nref-%s-"%(N,base_ref)
    solutions = deflatedbarrier(problem, params, mu_start=0, mu_end = 1e-5, max_halfstep = 1, saving_folder = saving_folder)
