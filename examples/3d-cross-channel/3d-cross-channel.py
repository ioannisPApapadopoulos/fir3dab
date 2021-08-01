# -*- coding: utf-8 -*-
from firedrake import *
from fir3dab import *

"""
This script implements the discovery of solutions of the 3D cross-channel problem on a
"coarse" mesh. We use preconditioning for the linear systems to reduce the solve to

1. Solve the diagonal material distribution block
2. Solve the block-diagonal pressure mass matrix
3. Solve the augmented momentum block

aL1) All three are inverted with MUMPS LU in this script (feasible since we are on
a coarse mesh).

In total we find 3 solutions.
"""

width = 1.     # aspect ratio
N = 20         # mesh resolution
gamma_al = 1e6 # augmented Lagrangian parameter

def InflowOutflow(mesh):
    # Boundary conditions of 3D cross-channel example. This models 2 inlets
    # and 2 outlets arranged in a cross pattern.
    x = SpatialCoordinate(mesh)
    r = sqrt(1./(12.*pi))
    gbar = 1.0

    inlet = lt(x[0], 1e-10)
    outlet = gt(x[0], width-1e-10)
    first_pipe = And(inlet, lt(((x[1] - 1.0/2)/r)**2 + ((x[2] - 1.0/4)/r)**2, 1.0))
    second_pipe = And(inlet, lt(((x[1] - 1.0/2)/r)**2 + ((x[2] - 3.0/4)/r)**2, 1.0))

    third_pipe = And(outlet, lt(((x[1] - 1.0/4)/r)**2 + ((x[2] - 1.0/2)/r)**2, 1.0))
    fourth_pipe = And(outlet, lt(((x[1] - 3.0/4)/r)**2 + ((x[2] - 1.0/2)/r)**2, 1.0))
    x_on_boundary = Or(lt(x[0], 1e-10), gt(x[0], width-1e-10))

    val_in_first_pipe   = gbar*(1 - ( ((x[1] - 1.0/2)/r)**2 + ((x[2] - 1.0/4)/r)**2 ))
    val_in_second_pipe  = gbar*(1 - ( ((x[1] - 1.0/2)/r)**2 + ((x[2] - 3.0/4)/r)**2 ))
    val_in_third_pipe   = gbar*(1 - ( ((x[1] - 1.0/4)/r)**2 + ((x[2] - 1.0/2)/r)**2 ))
    val_in_fourth_pipe  = gbar*(1 - ( ((x[1] - 3.0/4)/r)**2 + ((x[2] - 1.0/2)/r)**2 ))

    x_component = conditional(x_on_boundary,
                              conditional(first_pipe, val_in_first_pipe,
                              conditional(second_pipe, val_in_second_pipe,
                              conditional(third_pipe, val_in_third_pipe,
                              conditional(fourth_pipe, val_in_fourth_pipe,
                              Constant(0))))),
                              Constant(0))

    y_component = Constant(0)
    z_component = Constant(0)
    return as_vector([x_component, y_component, z_component])

class Mass(AuxiliaryOperatorPC):
    # Class for pressure mass matrix approximation of the Schur complement
    def form(self, pc, test, trial):
        K = 1./gamma_al * inner(test,trial)*dx
        return (K, None)

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        # "Simple" box mesh
        mesh = BoxMesh(N, N, N, width, 1.0, 1.0, comm=comm)
        self.area = width
        return mesh

    def function_space(self, mesh):
        # (Pointwise-evaluated) first-order BDM discretization for the velocity
        # and pressure pair, with DG0 for the material distribution.
        Ve = FiniteElement("BDM", mesh.ufl_cell(), 1)# velocity
        Pe = FiniteElement("DG", mesh.ufl_cell(), 0) # pressure
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals

        Ze = MixedElement([Ce, Ve, Pe, Re])

        Z  = FunctionSpace(mesh, Ze)
        info_blue("Number of degrees of freedom: ", Z.dim())
        # Take some data. First, BCs
        self.expr = InflowOutflow(mesh)

        # Next, a function space we use to solve for our initial guess
        Ge = MixedElement([Ve, Pe])
        self.G = FunctionSpace(mesh, Ge)
        self.Gbcs = [DirichletBC(self.G.sub(0), self.expr, "on_boundary")]
        return Z

    def gradIP(self, z, w, index):
        # This method is an implementation of the broken gradient.
        u = split(z)[index]; v = split(w)[index]
        q = split(w)[index+1]

        Z = z.function_space()
        mesh = Z.mesh()
        Re = 1
        g = self.expr #InflowOutflow(mesh) # Constant((0,0,0))

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
        (gamma, alpha, q) = params
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
        Solve the linear Stokes equations for the values of (u, p).
        """

        info_green("Computing initial guess.")
        gamma = Constant(params[0])
        rho_guess = gamma

        g = Function(self.G)
        (u, p) = split(g)
        k = TestFunction(self.G)
        J = self.stokes(u, p, rho_guess, params)
        F = derivative(J, g, TestFunction(self.G)) + self.gradIP(g, k, 0)
        solver_params=({"ksp_type": "preonly",
                         "mat_type": "aij",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps",
                         "snes_monitor": None})
        nsp = MixedVectorSpaceBasis(self.G, [self.G.sub(0), VectorSpaceBasis(constant=True)])
        solve(F == 0, g, bcs = self.Gbcs, nullspace=nsp, solver_parameters = solver_params)
        info_green("Initial guess computed, projecting ...")

        lmbda_guess = Constant(10)
        rho_guess = variable(rho_guess)

        z = Function(Z)
        (u, p) = g.split()
        z.split()[0].interpolate(rho_guess)
        info_green("Interpolated rho.")
        z.split()[1].assign(u)
        info_green("Assigned u.")
        z.split()[2].assign(p)
        info_green("Assigned p.")
        z.split()[3].interpolate(lmbda_guess)
        info_green("Interpolated lmbda.")

        info_green("Initial guess projected.")
        return [z]

    def stokes(self, u, p, rho, params):
        """The augmented Stokes functional, without constraints"""

        J = (
            + 0.5 * gamma_al*inner(div(u), div(u))*dx
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )

        return J

    def number_solutions(self, mu, params):
        if float(mu) > 38.8:
            return 1
        elif 35 < float(mu) <= 38.8:
            return 2
        else: return 3

    def update_mu(self, z, mu, iters, k, k_mu_old, params):
        k_mu = 0.9
        theta_mu = 1.2
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu


    def solver_parameters(self, mu, branch, task, params):
        (gamma,alphabar, q) = params
        linesearch = "l2"
        max_it = 100
        if task == 'ContinuationTask':
            damping = 1.0
        elif task == 'DeflationTask':
            damping = 0.6
        else:
            damping = 1.0

        args = {
                "snes_max_it": max_it,
                "snes_atol": 1e-5,
                "snes_stol": 1e-5,
                "snes_converged_reason": None,
                "snes_divergence_tolerance": 1e100,
                "snes_monitor": None,
                "snes_linesearch_type": linesearch,
                "snes_linesearch_monitor": None,
                "snes_linesearch_damping": damping,
                "ksp_atol": 5.0e-7,
                "ksp_rtol": 5e-1,

                # fGMRES outermost solver
                "ksp_type": "fgmres",
                "ksp_monitor_true_residual": None,
                "ksp_converged_reason": None,

                # We want to do a fieldsplit to handle the R-block since firedrake
                # cannot assemble aij matrices with R-block
                "mat_type": "matfree",
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

                "mat_mumps_icntl_14": 500,

                # field 0 is the big guy, here we try and tackle it by more
                # Schur fieldsplits, first we take the material distribution
                # as the top left block

                "fieldsplit_0":{
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

                        # material distribution block is a dressed-up mass matrix and should
                        # not be difficult to invert, can use CG or CG preconditioned with ILU
                        "fieldsplit_0":{
                            "ksp_type": "preonly",
                            # "pc_type": "python",
                            # "pc_python_type": "firedrake.AssembledPC",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                            "mat_mumps_icntl_14": 500,},

                        # This approximation can be approximated by another Schur complement
                        # factorization since it looks like Stokes! The top left block is
                        # the momentum block
                        "fieldsplit_1":{
                            "ksp_type": "preonly",
                            "pc_type": "fieldsplit",
                            "pc_fieldsplit_type": "schur",
                            "pc_fieldsplit_schur_fact_type": "full",
                            "pc_fieldsplit_0_fields": "0",
                            "pc_fieldsplit_1_fields": "1",

                        # The momentum block should eventually be approximated by a multigrid
                        # scheme with special prolongation and restriction, for now LU
                            "fieldsplit_0_ksp_type": "preonly",
                            # "fieldsplit_0_pc_type": "python",
                            # "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                            "fieldsplit_0_pc_type": "lu",
                            "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
                            "fieldsplit_0_mat_mumps_icntl_14": 500,

                        # Due to the augmented Lagrangian term, this Schur complement looks like
                        # the pressure mass matrix * 1/gamma_al. For now we invert the
                        # mass matrix with LU, but can be done many others ways quite easily
                            "fieldsplit_1_ksp_type": "preonly",
                            "fieldsplit_1_ksp_norm_type": "unpreconditioned",
                            "fieldsplit_1_pc_type": "python",
                            "fieldsplit_1_pc_python_type": "__main__.Mass",
                            "fieldsplit_1_aux_pc_type": "lu",
                            "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
                            "fieldsplit_1_aux_mat_mumps_icntl_14": 500,},
                            }
                      }
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
    params = [0.1, 2.5e4, 0.1] #(gamma, alphabar, q) # parameters for the BorrvallProblem
    solutions = deflatedbarrier(problem, params, mu_start=100, mu_end = 1e-10, max_halfstep = 1)
