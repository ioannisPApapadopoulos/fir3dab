from firedrake import *
from .deflation import ShiftedDeflation
from functools import partial
from .prediction import *
from .misc import *
import os
from firedrake.petsc import *


class PrimalInteriorPoint(object):
    def function_space(self, mesh):
        raise NotImplementedError

    def mesh(self, comm):
        raise NotImplementedError

    def coarse_meshes(self, comm):
        return None

    def expected_inertia(self):
        return None

    def initial_guesses(self, V, params):
        raise NotImplementedError

    def lagrangian(self, z, params):
        raise NotImplementedError

    def penalty(self, z, lb, ub, mu, params):
        rho = split(z)[0]
        return -mu*ln(rho - lb)*dx  - mu*ln(ub - rho)*dx

    def infeasibility(self, z, lb, ub, mu, params):
        u = split(z)[0]
        return plus(lb-u)**2*dx + plus(u - ub)**2*dx

    def residual(self, z, w, lb, ub, mu, params):
        J = self.lagrangian(z, params) + self.penalty(z, lb, ub, mu, params)
        F = derivative(J, z, w)
        return F

    def jacobian(self, F, state, params, test, trial):
        return derivative(F, state, trial)

    def boundary_conditions(self, V, params):
        raise NotImplementedError

    def bounds(self, mesh, mu, params):
        ep = 1e-5
        return (Constant(0.0-ep), Constant(1.0+ep))

    def bounds_vi(self, Z, mu, params):
        return None

    def squared_norm(self, a, b, params):
        arho = split(a)[0]
        brho = split(b)[0]
        return (inner(arho - brho, arho - brho)*dx)

    def solver_parameters(self, mu, branch, task, params):
        raise NotImplementedError

    def save_pvd(self, pvd, z, mu):
        if float(mu) == 0:
            rho = z.split()[0]
            pvd.write(rho)

    def save_solution(self, comm, z, branch, solutionpath_string):
        h5 = HDF5File(solutionpath_string + "/%s.xml.gz" %branch, "w", comm = comm)
        h5.write(z, "/guess")
        del h5
        return


    def update_mu(self, z, mu, iters, k, k_mu_old, params):
        # rules of IPOPT DOI: 10.1007/s10107-004-0559-y
        etol = 1e-15
        k_mu = 0.7
        theta_mu = 1.5
        next_mu = max(etol/10., min(k_mu*mu, mu**theta_mu))
        return next_mu

    def deflation_operator(self, params):
        return ShiftedDeflation(2, 1, self.squared_norm)

    def nullspace(self, V, params):
        return None

    def transfermanager(self):
        return None

    def context(self):
        return None

    def solver(self, u, lb, ub, mu, nref, params, task, branch):
        V = u.function_space()
        F = self.residual(u, TestFunction(V), lb, ub, mu, params)
        bcs = self.boundary_conditions(V, params)
        sp = self.solver_parameters(mu, branch, task, params)
        nsp = self.nullspace(V, params)

        nvp = NonlinearVariationalProblem(F, u, bcs=bcs)
        nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, nullspace=nsp, options_prefix="")

        nvs._ctx.appctx["mu"] = mu
        if hasattr(self, 'function_space_hierarchy'):
            nvs._ctx.appctx["Zs"] = self.function_space_hierarchy

        if hasattr(self, 'c2f_mapping'):
            nvs._ctx.appctx["c2f_mapping"] = self.c2f_mapping
        # add any user-defined contexts too
        if self.context():
            nvs._ctx.appctx.update(self.context())
        
        transfermanager = self.transfermanager()
        if transfermanager:
            info_blue("Setting Transfer Manager")
            nvs.set_transfer_manager(transfermanager)

        # add barrier parameter to context, useful if needed in solver
         
        # prolong = partial(self.prolong, params=params)
        # restrict = partial(self.restrict, params=params)
        # inject = partial(self.inject, params=params)
        # ctx = dmhooks.get_transfer_operators(V)
        # nvs.set_transfer_operators(ctx)
        #nvs._transfer_operators = None
        return nvs

    def compute_stability(self, mu, params, lb, ub, branch, z, v, w, Z, bcs, J):
        trial = w
        test  = v
        # test = TestFunction(Z)
        comm = Z.mesh().mpi_comm()
        # a dummy linear form, needed to construct the SystemAssembler
        b = inner(Function(Z), test)*dx  # a dummy linear form, needed to construct the SystemAssembler

        # Build the LHS matrix
        A = assemble(J, mat_type="aij")
        [bc.apply(A) for bc in bcs]

        pc = PETSc.PC().create(comm)
        pc.setOperators(A.petscmat)
        pc.setType("cholesky")
        # pc.setFactorSolverType("mumps")
        pc.setUp()
        Factor = pc.getFactorMatrix()
        (neg, zero, pos) = Factor.getInertia()
        inertia  = [neg, zero, pos]
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": is_stable}
        return inertia

    def number_solutions(self, mu, params):
        return 1

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return nothing(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)

    def cost(self, z, params):
        return assemble(self.energy(self, z, params))

    def volume_constraint(self, params):
        raise NotImplementedError
