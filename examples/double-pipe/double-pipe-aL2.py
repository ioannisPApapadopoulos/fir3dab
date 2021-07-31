# -*- coding: utf-8 -*-
from firedrake import *
from alfi import *
from alfi.transfer import *
from fir3dab import *

"""
This script implements the double-pipe problem with a Brezzi-Douglas-Marini
discretization for the velocity-pressure pair.

We use the preconditioning for the linear systems to reduce the solve to

1. Solve the diagonal material distribution block
2. Solve the block-diagonal pressure mass matrix
3. Solve the augmented momentum block

In this script we use the strategy:

aL2) 1. and 2. are inverted with MUMPS LU but 3. is approximated by GMRES preconditioned
with a (2-grid) geometric MG cycle with star patch relaxation and a representation of the
active set on the coarse level.

In total we find 2 solutions.
"""

width = 1.5 # aspect ratio
gamma_al = 1e4 # augmented Lagrangian parameter
dgtransfer = DGInjection() # some transfer operators from alfi

# before and after methods are for mesh hierarchy generation
def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

def InflowOutflow(mesh):
    # Boundary conditions of double-pipe. This models 2 inlets
    # and 2 outlets with parabolic bcs.
    x = SpatialCoordinate(mesh)
    l = 1.0/6.0
    gbar = 1.0

    x_on_boundary = Or(lt(x[0], 1e-10), gt(x[0], width-1e-10))
    y_in_first_pipe = And(gt(x[1], 1/4 - l/2), lt(x[1], 1/4 + l/2))
    val_in_first_pipe = gbar*(1 - (2*(x[1] - 1.0/4)/l)**2)
    y_in_second_pipe = And(gt(x[1], 3/4 - l/2), lt(x[1], 3/4 + l/2))
    val_in_second_pipe = gbar*(1 - (2*(x[1] - 3.0/4)/l)**2)

    x_component = conditional(x_on_boundary,
                              conditional(y_in_first_pipe, val_in_first_pipe,
                                          conditional(y_in_second_pipe, val_in_second_pipe,
                                                      Constant(0))),
                              Constant(0))
    y_component = Constant(0)
    return as_vector([x_component, y_component])

class Mass(AuxiliaryOperatorPC):
    # Class for pressure mass matrix approximation of the Schur complement
    def form(self, pc, test, trial):
        K = 1./gamma_al * inner(test,trial)*dx
        return (K, None)

class BDMTransferManager(TransferManager):
   # Transfer operators for robust MG cycle
   def __init__(self, *, native_transfer=None, use_averaging=True):
        self.native_transfers = {VectorElement("DG", triangle, 1): (dgtransfer.prolong, restrict, dgtransfer.inject),
                                 FiniteElement("DG", triangle, 0): (prolong, restrict, inject),
                                 FiniteElement("R", triangle, 0): (prolong, restrict, inject),
                                 }
        self.use_averaging = use_averaging
        self.caches = {}

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        # arguments for mesh hierarchy
        distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
        # Load "simple" rectangle mesh.
        mesh = RectangleMesh(N, N, width, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)
        # generate mesh hierarchy
        self.mh = MeshHierarchy(mesh, nref, reorder=True, callbacks=(before,after))
        mesh = self.mh[-1]
        # The coarse to fine element mappings are required in the MG cycle (if being used)
        # in order to represent the active set on the coarser levels. We choose to save
        # this mapping here to self.
        self.c2f_mapping = self.mh.coarse_to_fine_cells
        return mesh

    def function_space(self, mesh):
        # (Integral-evaluated) first-order BDM discretization for the velocity
        # and pressure pair, with DG0 for the material distribution.
        Ve = FiniteElement("BDM", mesh.ufl_cell(), 1, variant = "integral") # velocity
        Pe = FiniteElement("DG", mesh.ufl_cell(), 0) # pressure
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals
        Ze = MixedElement([Ce, Ve, Pe, Re])

        Z  = FunctionSpace(mesh, Ze)
        self.Z = Z

        # Here we save the function spaces at the different levels to self.
        # This is utilized later on during the MG cycle in order to assemble
        # the augmented momentum block.
        self.function_space_hierarchy = []
        for i in range(nref+1):
            self.function_space_hierarchy.append(FunctionSpace(self.mh[i], MixedElement([Ce,Ve])))

        info_blue("Number of degrees of freedom: ", Z.dim())
        # BCs
        self.expr = InflowOutflow(mesh)

        # Used in the initial guess
        Ge = MixedElement([Ve, Pe])
        self.G = FunctionSpace(mesh, Ge)
        self.Gbcs = [DirichletBC(self.G.sub(0), InflowOutflow(mesh), "on_boundary")]
        return Z

    def gradIP(self, z, w, index):
        # This method is an implementation of the broken gradient.
        u = split(z)[index]; v = split(w)[index]

        Z = z.function_space()
        mesh = Z.mesh()
        Re = 1
        g = self.expr

        sigma = Constant(10) * max(Z.sub(index).ufl_element().degree()**2, 1)
        n = FacetNormal(z.ufl_domain())
        h = CellDiameter(z.ufl_domain())

        A = (
             1/Re * inner(grad(u), grad(v))*dx
           - 1/Re * inner(avg(grad(u)), 2*avg(outer(v, n))) * dS
           - 1/Re * inner(avg(grad(v)), 2*avg(outer(u, n))) * dS
           + 1/Re * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
           - 1/Re * inner(outer(v,n), grad(u))*ds
           - 1/Re * inner(outer(u-g,n), grad(v))*ds
           + 1/Re * (sigma/h)*inner(v,u-g)*ds
        )
        return A

    def lagrangian(self, z, params):
        (rho, u, p, lmbda) = split(z)
        (gamma, alphabar, q) = params
        # Lagrangian of the problem (minus the broken gradient which will be
        # added later).
        L = (
            + 0.5 * gamma_al*inner(div(u), div(u))*dx # augmented Lagrangian term
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            - inner(lmbda, gamma - rho)*dx
            )

        return L

    def residual(self, z, w, lb, ub, mu, params):
        # Compute derivative of the barrier functional and append with broken gradient terms
        J = self.lagrangian(z, params) + self.penalty(z, lb, ub, mu, params)
        F = derivative(J, z, w) + self.gradIP(z, w, 1)
        return F

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
        Z = z.function_space()
        mesh = Z.mesh()
        g = self.expr

        sigma = Constant(10) * max(Z.sub(1).ufl_element().degree()**2, 1)
        n = FacetNormal(z.ufl_domain())
        h = CellDiameter(z.ufl_domain())

        A = (
             0.5 * inner(grad(u), grad(u))*dx
           - 0.5 * inner(avg(grad(u)), 2*avg(outer(u, n))) * dS
           + 0.5 * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(u,n))) * dS
           - 0.5 * inner(outer(u-g,n), grad(u))*ds
           + 0.5 * (sigma/h)*inner(u-g,u-g)*ds
        )
        L = A + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), self.expr, "on_boundary")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the augmented Stokes equations for the values of (u, p, p0).
        """
        comm = Z.comm
        info_blue("Computing initial guess.")
        gamma = Constant(params[0])
        rho_guess = gamma

        g = Function(self.G)
        (u, p) = split(g)
        J = self.stokes(u, p, rho_guess, params)
        k = TestFunction(self.G)
        F = derivative(J, g, k) + self.gradIP(g,k,0)

        solver_params=({"ksp_type": "preonly",
                         "mat_type": "aij",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps",
                         "snes_monitor": None,})
        nsp = MixedVectorSpaceBasis(self.G, [self.G.sub(0), VectorSpaceBasis(constant=True)])
        solve(F == 0, g, bcs = self.Gbcs, nullspace=nsp, solver_parameters = solver_params)
        info_blue("Initial guess computed, projecting ...")

        lmbda_guess = Constant(10)
        rho_guess = variable(rho_guess)

        z = Function(Z)
        (u, p) = g.split()
        z.split()[0].interpolate(rho_guess)
        z.split()[1].assign(u)
        z.split()[2].assign(p)
        z.split()[3].interpolate(lmbda_guess)

        info_blue("Initial guess projected.")
        return [z]

    def stokes(self, u, p, rho, params):
        """The Stokes functional, without constraints"""

        J = (
            + gamma_al * inner(div(u),div(u))*dx
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )

        return J

    def number_solutions(self, mu, params):
        return 2

    def update_mu(self, z, mu, iters, k, k_mu_old, params):
        # update scheme for barrier parameter
        if float(mu) > 20:
            k_mu = 0.9
        else:
            k_mu = 0.8
        theta_mu = 1.2
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu


    def solver_parameters(self, mu, branch, task, params):
        # Solver parameters
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

        if nref >= 2:
            snes_atol = 1e-5
        else:
            snes_atol = 1e-6 if float(mu) > 0 else 1e-7

        return self.aL2_parameters(params, mu, max_it, snes_atol, linesearch, damping)

    def aL2_parameters(self, params, mu, max_it, snes_atol, linesearch, damping):
        args = {
                "snes_max_it": max_it,
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
                "ksp_atol": 1e-7,
                "ksp_rtol": 1e-7,
                #"ksp_max_it": 3,

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
                # Schur fieldsplits, first we take the material distribution
                # as the top left block

                "fieldsplit_0":{
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled":{
                        "ksp_type": "preonly",
                        "pc_type": "fieldsplit",
                        "pc_fieldsplit_type": "schur",
                        "pc_fieldsplit_schur_fact_type": "full",
                        "pc_fieldsplit_schur_precondition": "selfp",
                        "pc_fieldsplit_0_fields": "0",
                        "pc_fieldsplit_1_fields": "1,2",

                        # Need this for the solver to pick up the correct transfer operator
                        "mg_transfer_manager": __name__ + ".BDMTransferManager",

                        # material distribution block is a diagonal matrix and should
                        # not be difficult to invert with LU
                        "fieldsplit_0":{
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                            "mat_mumps_icntl_14": 500,},

                        # Momentum-pressure block can be approximated by another Schur complement
                        # factorization since it looks like Stokes! The top left block is
                        # the momentum block
                        "fieldsplit_1":{
                            "ksp_type": "preonly",
                            "pc_type": "fieldsplit",
                            "pc_fieldsplit_type": "schur",
                            "pc_fieldsplit_schur_fact_type": "full",
                            "pc_fieldsplit_0_fields": "0",
                            "pc_fieldsplit_1_fields": "1",
                            "mg_transfer_manager": __name__ + ".BDMTransferManager",

                            # The momentum block is approximated by a specialized multigrid scheme
                            "fieldsplit_0":{
                                    # Use FGMRES to accurately invert the momentum block
                                    # to a relatively sharp tolerance so that the outermost
                                    # FGMRES iterations remain low.
                                    "ksp_type": "fgmres",
                                    "ksp_norm_type": "unpreconditioned",
                                    "ksp_converged_reason": None,
                                    "ksp_atol": 1e-8, #
                                    "ksp_rtol": 1e-9,
                                    "ksp_max_it": 40,
                                    "ksp_gmres_restart": 200,

                                    "pc_type": "mg",
                                    "pc_mg_type": "full",
                                    "mg_transfer_manager": __name__ + ".BDMTransferManager",
                                    # DABCoarseGridPC is a class implemented in the
                                    # fir3dab library. It correctly assembles the
                                    # augmented momentum block on the coarsest level
                                    # including the representation of the active set.
                                    "mg_coarse":{
                                        "ksp_type": "preonly",
                                        "pc_type": "python",
                                        # custom coarse-grid PC
                                        "pc_python_type": __name__ + ".DABCoarseGridPC",
                                        "dab": {
                                           "nref": nref,
                                           "sigma": 10,
                                           "params": params,
                                           "gamma_al": gamma_al,},
                                    },
                                    # DABFineGridPC is a class implemented in the
                                    # fir3dab library. It correctly assembles the
                                    # augmented momentum block on the finer levels
                                    # including the representation of the active set.
                                    # It can also be composed with exotic solver, e.g.
                                    # asm patch solvers.
                                    "mg_levels": {
                                        "ksp_type": "gmres",
                                        "ksp_norm_type": "unpreconditioned",
                                        "ksp_convergence_test": "skip",
                                        "ksp_max_it": 5,
                                        "pc_type": "python",
                                        "pc_python_type": __name__ + ".DABFineGridPC",
                                        "dab": {
                                           "nref": nref,
                                           "sigma": 10,
                                           "params": params,
                                           "gamma_al": gamma_al,
                                           "ksp_type": "preonly",
                                           "pc_type": "python",
                                           "pc_python_type": "firedrake.ASMStarPC",
                                           #"pc_star_backend": "tinyasm",
                                           "patch_pc_patch_save_operators": True,
                                           "patch_pc_patch_dense_inverse": True,
                                           "patch_pc_patch_partition_of_unity": False,
                                           "patch_pc_patch_sub_mat_type": "seqdense",
                                           "patch_pc_patch_construct_dim": 0,
                                           "patch_pc_patch_construct_type": "star",
                                           "patch_sub_ksp_type": "preonly",
                                           "patch_sub_pc_type": "lu",
                                           "patch_sub_pc_factor_shift_type": "nonzero",},
                                },
                            },
                    # Due to the augmented Lagrangian term, this Schur complement looks like
                    # the pressure mass matrix * 1/gamma_al. For now we invert the
                    # mass matrix with LU, but can be done many others ways quite easily
                            "fieldsplit_1":{
                                "ksp_type": "preonly",
                                "ksp_norm_type": "unpreconditioned",
                                "pc_type": "python",
                                "pc_python_type": __name__ + ".Mass",
                                "aux_pc_type": "lu",
                                "aux_pc_factor_mat_solver_type": "mumps",
                                "aux_mat_mumps_icntl_14": 500,}
                        },
                  },
             },
         }
        return args

    def alpha(self, rho, params):
        (gamma, alphabar,q) = params
        return alphabar * ( 1. - rho*(1+q)/(rho+q))

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
    nref = 1
    for N in [50, 100]:
        saving_folder = "aL2-N-%s-nref-%s-"%(N, nref)
        solutions = deflatedbarrier(problem, params, mu_start=105, mu_end = 1e-5, max_halfstep = 1, saving_folder=saving_folder)
