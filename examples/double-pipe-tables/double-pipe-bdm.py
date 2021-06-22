# -*- coding: utf-8 -*-
from firedrake import *
from deflatedbarrier import *

"""
A deflated barrier method implementation applied to the double-pipe problem
discretized with a divergence-free DG FEM. Here we use a piecewise constant
discretization for the material distribution and a Brezzi-Douglas-Marini
discretization for the velocity-pressure pair.
"""

# Mesh properties
width = 1.5 # aspect ratio
N = 20 # mesh resolution
nref = 6
base_ref = 1

# Augmented Lagrangian parameter
gamma_al = 1e4

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

    x_on_boundary = Or(lt(x[0], 1e-10), gt(x[0], width-1e-10))

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

class Mass(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        K = 1./gamma_al * inner(test,trial)*dx
        return (K, None)

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
        mesh = RectangleMesh(N, N, width, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)
        self.mh = MeshHierarchy(mesh, nref, reorder=True, callbacks=(before,after))
        mesh = self.mh[base_ref]
        return mesh

    def function_space(self, mesh):
        Ve = FiniteElement("BDM", mesh.ufl_cell(), 1, variant = "integral") # velocity
        Pe = FiniteElement("DG", mesh.ufl_cell(), 0) # pressure
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals
        Ze = MixedElement([Ce, Ve, Pe, Re])

        self.Ze = Ze
        Z  = FunctionSpace(mesh, Ze)
        info_blue("Number of degrees of freedom: ", Z.dim())
        # Take some data. First, BCs
        self.expr = InflowOutflow(mesh)
        Ge = MixedElement([Ve, Pe])
        self.G = FunctionSpace(mesh, Ge)
        self.Gbcs = [DirichletBC(self.G.sub(0), InflowOutflow(mesh), "on_boundary")]
        return Z

    def gradIP(self, z, w, index):
        u = split(z)[index]; v = split(w)[index]

        Z = z.function_space()
        mesh = Z.mesh()
        Re = 1
        g = self.expr

        sigma = Constant(10) * max(Z.sub(index).ufl_element().degree()**2, 1)
        n = FacetNormal(z.ufl_domain())
        h = CellDiameter(z.ufl_domain()) # CellVolume(mesh)/FacetArea(mesh)

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

        L = (
            + 0.5 * gamma_al*inner(div(u), div(u))*dx # augmented Lagrangian term
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            - inner(lmbda, gamma - rho)*dx
            )

        return L

    def residual(self, z, w, lb, ub, mu, params):
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
        L = (
              0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), self.expr, "on_boundary")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the Stokes equations for the values of (u, p, p0).
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


    def number_solutions(self, mu, params):
        if float(mu) == 105:
            return 1
        else:
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
                        "pc_type": "fieldsplit",
                        "pc_fieldsplit_type": "schur",
                        "pc_fieldsplit_schur_fact_type": "full",
                        "pc_fieldsplit_schur_precondition": "selfp",
                        "pc_fieldsplit_0_fields": "0",
                        "pc_fieldsplit_1_fields": "1,2",


                        # material distribution block is a diagonal matrix. Invert
                        # with MUMPS
                        "fieldsplit_0":{
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                            "mat_mumps_icntl_14": 500,},

                        # This approximation can be approximated by another Schur complement
                        # factorization since it looks like Stokes! The top left block is
                        # the momentum block
                        "fieldsplit_1":{
                            "ksp_type": "preonly",
                            #"ksp_norm_type": "unpreconditioned",
                            #"ksp_converged_reason": None,
                            "pc_type": "fieldsplit",
                            "pc_fieldsplit_type": "schur",
                            "pc_fieldsplit_schur_fact_type": "full",
                            "pc_fieldsplit_0_fields": "0",
                            "pc_fieldsplit_1_fields": "1",


                            # The Schur complement corrected-momentum block is solved with LU
                            # in this strategy
                            "fieldsplit_0":{
                                "ksp_type": "preonly",
                                "pc_type": "lu",
                                "pc_factor_mat_solver_type": "mumps",
                                "mat_mumps_icntl_14": 500,},
                            # Due to the augmented Lagrangian term, this Schur complement looks like
                            # the pressure mass matrix * 1/gamma_al. For now we invert the
                            # mass matrix with LU.
                            "fieldsplit_1":{
                                "ksp_type": "preonly",
                                "ksp_norm_type": "unpreconditioned",
                                "pc_type": "python",
                                "pc_python_type": __name__ +  ".Mass",
                                "aux_pc_type": "lu",
                                "aux_pc_factor_mat_solver_type": "mumps",
                                "aux_mat_mumps_icntl_14": 500,},
                        },
                  },
             },
         }
        return args

    def alpha(self, rho, params):
        (gamma, alphabar,q) = params
        return alphabar * ( 1. - rho*(1+q)/(rho+q))

    def stokes(self, u, p, rho, params):
        """The Stokes functional, without constraints"""
        J = (
              #inner(grad(u), grad(u))*dx
            + gamma_al * inner(div(u), div(u))*dx
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
    saving_folder = "output/BDM-N-%s-nref-%s-"%(N,base_ref)
    solutions = deflatedbarrier(problem, params, mu_start=105, mu_end = 1e-5, max_halfstep = 1, saving_folder = saving_folder)
