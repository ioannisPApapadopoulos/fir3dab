# -*- coding: utf-8 -*-
from firedrake import *
from fir3dab import *

"""
This script implements the double-pipe problem with a Taylor-Hood
discretization for the velocity-pressure pair.

This is simply a test of the effectiveness of the Cahouet-Chabard preconditioner for
the systems that arise in the deflated barrier method. Hence, the algorithm begins
and terminates at a barrier parameter value of mu=105 and we only converge to the
first branch.

The Cahouet-Chabard strategy reduces the solve to:

1. Solve the diagonal material distribution block
2. Solve the unaugmented momentum block
3. Approximate the inverse of the innermost Schur complement requiring:
    a) CG preconditioned with:
    b) Solve the block-diagonal pressure mass matrix
    c) Solve a pressure stiffness-like block

In this strategy, all the sparse blocks are inverted with MUMPS LU.

"""

width = 1.5 # aspect ratio

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


class BrinkmanPC(PCBase):
    r"""
    Preconditioner for the Stokes-Brinkman problem. Borrvall and Petersson
    use this PC in https://doi.org/10.1002/fld.426. Without using this terminology,
    they do a Schur complement of the Stokes-Brinkman system. The top left block
    is "easy" to invert and they approximate the Schur complement S with S_a where

    S_a^-1 q = \phi_q + \nu q where \phi_q solves

    -div(1/alpha grad \phi_q) = q, \int_\omega \phi_q = 0 with Neumann bcs

    This is a perfect preconditioner when alpha is constant and we have periodic bcs.

    This code has been adapted
    from https://lists.mcs.anl.gov/mailman/htdig/petsc-users/2019-October/039567.html
    originally written by Miguel Salazar De Troya, email salazardetro1@llnl.gov

    There was a primal-dual mismatch in the original code, in particular the Mass
    matrix is required to be inverted and not simply multiplied. This fixes Miguel's
    issues with high CG counts for non-constant alpha
    """
    def initialize(self, pc):
        info_red("Inside INITIALIZE")

        from firedrake.assemble import allocate_matrix, create_assembly_callable
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        prefix = pc.getOptionsPrefix() + "brink_"
         # we assume P has things stuffed inside of it
        _, P = pc.getOperators()

        if P.getType() == "python":
            context = P.getPythonContext()
            (a, bcs) = (context.a, context.row_bcs)
        else:
            context = dmhooks.get_appctx(pc.getDM())
            (a, bcs) = (context.Jp or context.J, context._problem.bcs)

        test, trial = a.arguments()
        if test.function_space() != trial.function_space():
            raise ValueError("Pressure space test and trial space differ")

        Q = test.function_space()

        p = TrialFunction(Q)
        q = TestFunction(Q)

        def alpha(rho):
            (gamma, alphabar, q) = [1.0/3, 2.5e4, 0.1]
            return alphabar + (1e-5-alphabar) * rho*(1+q)/(rho+q)

        state = context.appctx["state"]
        rho = split(state)[0]

        # Regularisation to avoid having to think about nullspaces.
        stiffness = inner(1./alpha(rho) * grad(p), grad(q))*dx + Constant(1e-6)*inner(p,q)*dx
        mass = inner(p,q)*dx

        opts = PETSc.Options()

        # We're inverting the stiffness Kp and the mass Mp, so default them to assembled.
        # These can be overridden.
        default = parameters["default_matrix_type"]
        Kp_mat_type = opts.getString(prefix+"Kp_mat_type", default)
        Mp_mat_type = opts.getString(prefix+"Mp_mat_type", default)
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        # Kp depends on rho which changes with each BM iteration, so setup as as
        # callable for the update method
        self.Kp = allocate_matrix(stiffness,
                                 form_compiler_parameters=fcp,
                                 mat_type=Kp_mat_type,
                                 options_prefix=prefix + "Kp_")
        self._assemble_Kp = create_assembly_callable(stiffness, tensor=self.Kp,
                                                    form_compiler_parameters=fcp,
                                                    mat_type=Kp_mat_type)
        self._assemble_Kp()

        # Mass matrix only needs to be assembled once.
        Mp = assemble(mass, form_compiler_parameters=fcp,
                  mat_type=Mp_mat_type,
                  options_prefix=prefix + "Mp_")

        # temporary vectors to be used in apply and applyTranspose
        self.tmp_a = Mp.petscmat.createVecLeft()
        L =  q * dx
        self.tmp_b = assemble(L)

        Kksp = PETSc.KSP().create(comm=pc.comm)
        Kksp.incrementTabLevel(1, parent=pc)
        Kksp.setOptionsPrefix(prefix + "Kp_")
        Kksp.setOperators(self.Kp.petscmat)
        Kksp.setFromOptions()
        Kksp.setUp()
        self.Kksp = Kksp

        Mksp = PETSc.KSP().create(comm=pc.comm)
        Mksp.incrementTabLevel(1, parent=pc)
        Mksp.setOptionsPrefix(prefix + "Mp_")
        Mksp.setOperators(Mp.petscmat)
        Mksp.setFromOptions()
        Mksp.setUp()
        self.Mksp = Mksp

    def update(self, pc):
        info_red("Inside UPDATE")
        self._assemble_Kp()

    def apply(self, pc, x, y):
        with self.tmp_b.dat.vec_wo as rhs_dat:
            x.copy(rhs_dat)

        # compute phi_q
        with self.tmp_b.dat.vec_wo as rhs_dat:
            self.Kksp.solve(rhs_dat, self.tmp_a) # Kp^-1 x
        # should invert the mass matrix rather than multiply, compute primal q
        self.Mksp.solve(x, y) # Mp^-1 x
        y.axpy(1.0, self.tmp_a) # y = Kp^-1 x + Mp^-1 x

    def applyTranspose(self, pc, x, y):
        a, b = self.workspace
        with self.tmp_b.dat.vec_wo as rhs_dat:
            x.copy(rhs_dat)
        with self.tmp_b.dat.vec_wo as rhs_dat:
            self.Kksp.solveTranspose(rhs_dat, self.tmp_a)
        self.Mksp.solveTranspose(x, y)
        y.axpy(1.0, self.tmp_a)

    def view(self, pc, viewer=None):
        super(BrinkmannPC, self).view(pc, viewer)
        viewer.printfASCII("KSP solver for K^-1:\n")
        self.Kksp.view(viewer)

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        mesh = RectangleMesh(N, N, width, 1.0, comm=comm)
        self.area = width
        return mesh

    def function_space(self, mesh):
        # Need a H^1-conforming space for the pressure to use the Cahouet-Chabard
        # preconditioner, so using Taylor-Hood.
        Ve = VectorElement("CG", mesh.ufl_cell(), 2, dim =2) # velocity
        Pe = FiniteElement("CG", mesh.ufl_cell(), 1) # pressure
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
        Re = FiniteElement("R", mesh.ufl_cell(), 0) # reals

        Ze = MixedElement([Ce, Ve, Pe, Re])

        self.W = []
        self.W.append(FunctionSpace(mesh, MixedElement([Ce,Ve])))

        Z  = FunctionSpace(mesh, Ze)
        print("Number of degrees of freedom: ", Z.dim())

        # Take some data. First, BCs
        self.expr = InflowOutflow(mesh)
        # Next, a function space we use to solve for our initial guess
        Ge = MixedElement([Ve, Pe])
        self.G = FunctionSpace(mesh, Ge)
        self.Gbcs = [DirichletBC(self.G.sub(0), self.expr, "on_boundary")]
        return Z

    def lagrangian(self, z, params):
        # Normal Lagrangian for the Borrvall-Petersson problem
        (rho, u, p, lmbda) = split(z)
        (gamma, alpha, q) = params

        L = (
              0.5 * inner(grad(u), grad(u))*dx
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
        return 1

    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the Stokes equations for the values of (u, p).
        """

        info_green("Computing initial guess.")
        gamma = Constant(params[0])
        rho_guess = gamma

        g = Function(self.G)
        (u, p) = split(g)
        J = self.stokes(u, p, rho_guess, params)
        F = derivative(J, g, TestFunction(self.G))
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
        z.split()[1].assign(u)
        z.split()[2].assign(p)
        z.split()[3].interpolate(lmbda_guess)

        info_green("Initial guess projected.")
        return [z]

    def number_solutions(self, mu, params):
        if float(mu) > 110:
            return 1
        else: return 1

    def update_mu(self, z, mu, iters, k, k_mu_old, params):
        etol = 1e-15
        k_mu = 0.8
        theta_mu = 1.5
        next_mu = max(etol/10., min(k_mu*mu, mu**theta_mu))
        return next_mu

    def solver_parameters(self, mu, branch, task, params):
        (gamma,alphabar, q) = params
        linesearch = "l2"
        max_it = 100
        damping = 1.0

        args = {
               "snes_max_it": max_it,
               "snes_atol": 7e-7,
               "snes_stol": 1e-5,
               "snes_converged_reason": None,
               "snes_divergence_tolerance": 1e10,
               "snes_monitor": None,
               "snes_linesearch_type": linesearch,
               "snes_linesearch_monitor": None,
               "snes_linesearch_damping": damping,

               "ksp_atol": 1.0e-8,

               # fGMRES outermost solver
               "ksp_type": "fgmres",
               "ksp_gmres_restart": 100,
               #"ksp_rtol": 1.0e-4,
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

               # field 0 is the material distribution-momentum-pressure block
               "fieldsplit_0":{
                   "ksp_type": "preonly",
                   "pc_type": "python",
                   "pc_python_type": "firedrake.AssembledPC",
                   "assembled":{
                       "ksp_type": "preonly",
                       "pc_type": "fieldsplit",
                       "pc_fieldsplit_type": "schur",
                       "pc_fieldsplit_schur_fact_type": "full",
                       #"pc_fieldsplit_schur_precondition": "a11",
                       "pc_fieldsplit_0_fields": "0",
                       "pc_fieldsplit_1_fields": "1,2",

                       # material distribution block is a diagonal matrix
                       "fieldsplit_0":{
                           "ksp_type": "preonly",
                           "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps",
                           "mat_mumps_icntl_14": 500
                           },

                       # We take a further fieldsplit for the Schur complement which looks like the
                       # Stokes-Brinkman equations
                       "fieldsplit_1":{
                           "ksp_type": "fgmres",
                           "ksp_converged_reason": None,
                           "ksp_rtol": 1e-5,
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "schur",
                           "pc_fieldsplit_0_fields": "0",
                           "pc_fieldsplit_1_fields": "1",

                           # Invert the unaugmented momentum block (i.e. just A) with LU
                           "fieldsplit_0":{
                               "ksp_type": "preonly",
                               "pc_type": "lu",
                               "pc_factor_mat_solver_type": "mumps",
                               "mat_mumps_icntl_14": 500,
                               },

                           # Cahouet--Chabard approximation to the Schur complement
                           "fieldsplit_1":{
                               # Since the innermost Schur complement is dense, we want
                               # to use a Krylov method so we do not need to assemble it.
                               # Here CG works.
                               "ksp_type": "cg",
                               "ksp_norm_type": "unpreconditioned",
                               "ksp_rtol": 1e-5,
                               "ksp_atol": 1e-8,
                               "ksp_max_it": 10,
                               "ksp_converged_reason": None,
                               "pc_type": "python",
                               # Use the Cahouet-Chabard PC (defined above) to
                               # precondition the CG approximation of the innermost
                               # Schur complement. Invert all blocks with LU.
                               "pc_python_type": "__main__.BrinkmanPC",
                               "brink_Kp_ksp_type": "preonly",
                               "brink_Kp_pc_type": "lu",
                               "brink_Kp_pc_factor_mat_solver_type": "mumps",
                               "brink_Kp_mat_mumps_icntl_14": 500,
                               "brink_Mp_ksp_type": "preonly",
                               "brink_Mp_pc_type": "lu",
                               "brink_Mp_pc_factor_mat_solver_type": "mumps",
                               "brink_Mp_mat_mumps_icntl_14": 500,
                               },
                           }
                       }
                   }
               }

        return args

    def alpha(self, rho, params):
        (gamma, alphabar,q) = params
        # Need a lower bound for the inverse permeability as its inverse is
        # taken in the Cahouet-Chabard preconditioner.
        return alphabar + (1e-5-alphabar) * rho*(1+q)/(rho+q)

    def stokes(self, u, p, rho, params):
        """The Stokes functional, without constraints"""

        J = (
              0.5 * inner(grad(u), grad(u))*dx
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )

        return J

    def bounds_vi(self, Z, mu, params):
        inf = 1e100
        lb = Function(Z); ub = Function(Z)
        lb.split()[0].interpolate(Constant(0))
        ub.split()[0].interpolate(Constant(1))
        for i in range(1,2):
            lb.split()[i].interpolate(Constant((-inf, -inf)))
            ub.split()[i].interpolate(Constant((+inf, +inf)))
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
    for N in [50, 100]:
        solutions = deflatedbarrier(problem, params, mu_start=105, mu_end = 106, premature_termination=True, max_halfstep = 0)
