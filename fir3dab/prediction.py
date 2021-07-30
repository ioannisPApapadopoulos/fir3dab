# Implement tangent predictor for continuation steps. These
# are intended for use in the BifurcationProblem.predict method.

from firedrake import *
from ufl import derivative
from .deflation import setSnesBounds
from .mlogging import *
import numpy as np
from copy import deepcopy
from contextlib import ExitStack
from itertools import chain

def feasibletangent(problem, solution, v, w, oldmu, newmu, k, params, task, vi, hint=None):

    if k == 1:
        return (hint, 0, 0)

    info_green("Attempting feasible tangent prediction")
    coldmu = Constant(float(oldmu))
    chgmu = Constant(float(newmu)-float(oldmu))
    Z = solution.function_space()

    (lb, ub)  = problem.bounds(Z, newmu, params)
    du = Function(Z)

    class FixTheBounds(object):
        def bounds_vi(self, Z, newmu, params):
            try:
                (lb_vi, ub_vi) = problem.bounds_vi(Z, newmu, params)
            except: return None
            lb_vi.assign(lb_vi - solution)
            ub_vi.assign(ub_vi - solution)
            return (lb_vi, ub_vi)

        def boundary_conditions(self, Z, newmu):
            dubcs = problem.boundary_conditions(Z, newmu)
            [dubc.homogenize() for dubc in dubcs]
            return dubcs

        def residual(self, u, v, lb, ub, mu, params):
            F = problem.residual(solution, v, lb, ub, coldmu, params)
            G = derivative(F, solution, du) + derivative(F, coldmu, chgmu)
            return G

        def __getattr__(self, attr):
            return getattr(problem, attr)

    newproblem = FixTheBounds()
    G = newproblem.residual(solution, v, lb, ub, coldmu, params)
    dubcs = newproblem.boundary_conditions(Z, newmu)

    sp = newproblem.solver_parameters(float(oldmu), 0, task, params)
    nsp = newproblem.nullspace(Z, params)

    nvp = NonlinearVariationalProblem(G, du, bcs=dubcs)
    nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, nullspace=nsp, options_prefix="")

    vi_du = newproblem.bounds_vi(Z, newmu, params)
    if vi_du:
        nvs.snes.setType("vinewtonrsls")
        setSnesBounds(nvs.snes, vi_du)

    nvs.solve()

    success = nvs.snes.getConvergedReason() > 0


    hint = [None, float(oldmu)]
    if not success:
        info_red("Feasible tangent prediction failed")
        return (hint, nvs.snes.its, 0)
    else:
        solution.assign(solution + du)

        info_green("Feasible tangent prediction success")
        if nvs.snes.its:
            its = nvs.snes.its
        else:
            its = 0
        if nvs.snes.getAttr("total_ksp_its"):
            lits = nvs.snes.getAttr("total_ksp_its")
        else:
            lits = 0
        its = nvs.snes.its
        nvs.snes.its
        return (hint, its, lits)

def tangent(problem, solution, v, w, oldmu, newmu, k, params, task, vi, hint=None):

    if k == 1:
        return (hint, 0, 0)

    info_green("Attempting tangent linearisation")
    coldmu = Constant(float(oldmu))
    chgmu = Constant(float(newmu)-float(oldmu))
    Z = solution.function_space()

    (lb, ub)  = problem.bounds(Z, newmu, params)
    du = Function(Z)
    #du.assign(solution)

    F = problem.residual(solution, v, lb, ub, coldmu, params)
    #G = derivative(F, solution, du) + derivative(F, coldmu, chgmu) 
    P = problem.penalty(solution, lb, ub, chgmu, params)
    G = derivative(F, solution, du) #+ derivative(P, solution, v)
    rho = split(solution)[0]
    eta = split(v)[0]
    #G += -chgmu/(rho-lb)*eta*dx + chgmu/(ub-rho)*eta*dx
    

    dubcs = problem.boundary_conditions(Z, newmu)
    [dubc.homogenize() for dubc in dubcs]

    sp = problem.solver_parameters(float(oldmu), 0, task, params)
    nsp = problem.nullspace(Z, params)

    #nvp = NonlinearVariationalProblem(G, du, bcs=dubcs)
    #nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, nullspace=nsp, options_prefix="")
    #nvs.solve()
    #success = nvs.snes.getConvergedReason() > 0
    #dm = nvs.snes.getDM()
    #work = nvs._work
    #with nvs._problem.u.dat.vec as u:
    #    u.copy(work)
    #    with ExitStack() as stack:
    #        for ctx in chain((nvs.inserted_options(),dmhooks.add_hooks(dm, nvs, appctx=nvs._ctx)),
    #                         nvs._transfer_operators):
    #            stack.enter_context(ctx)

     #       nvs.snes.solve(None, work)

     #   work.copy(u)
    
    #nvs._setup = True
    #success = nvs.snes.getConvergedReason() > 0
    solve(G == 0, du, bcs=dubcs, nullspace=nsp, solver_parameters=sp)
    success = True

    hint = [None, float(oldmu)]
    if not success:
        info_red("Tangent linearisation failed")
        return (hint, nvs.snes.its, 0)
    else:
        solution.assign(solution + du)
        # infeasibility = assemble(problem.infeasibility(solution, lb, ub, newmu, params))
        # infeasibility = 0.0
        # info_green("Tangent linearisation success, infeasibility of guess %.5e" % (infeasibility))
        info_green("Tangent linearisation success")
        if nvs.snes.its:
            its = nvs.snes.its
        else:
            its = 0
        if nvs.snes.getAttr("total_ksp_its"):
            lits = nvs.snes.getAttr("total_ksp_its")
        else:
            lits = 0
        its = nvs.snes.its
        nvs.snes.its
        return (hint, its, lits)

def secant(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint):


    newmu = float(newmu)
    oldmu = float(oldmu)
    #oldmu_c = deepcopy(oldmu)
    oldsolution = solution.copy()
    #dm = problem._dm
    if hint[0] == None:
        pass
    else:
        # Unpack previous solution
        (prevsolution, prevmu) = hint

        du = Function(solution.function_space())
        # Find secant predictor
        multiplier = (newmu-oldmu)/(oldmu-prevmu)
        du.assign((oldsolution - prevsolution)*multiplier)
        infeasibility = assemble(problem.infeasibility(solution, Constant(0), Constant(1), newmu, params))
        info_green("Found secant predictor, infeasibility of guess %.5e" % (infeasibility))
        solution.assign(solution + du)

    hint = [oldsolution, float(oldmu)]

    return (hint, 0, 0)


def nothing(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
    return (hint, 0, 0)


def projection(solution):
        rho = solution.split()[0]
        lb_ = rho.copy(deepcopy=True); lb_.vector()[:] = 0.0
        ub_ = rho.copy(deepcopy=True); ub_.vector()[:] = 1.0
        rho_array = rho.vector().get_local()
        lb_ = lb_.vector().get_local()
        ub_ = ub_.vector().get_local()
        ub_active = np.where(rho_array>ub_)
        lb_active = np.where(rho_array<lb_)
        rho_array[ub_active] = ub_[ub_active]-1e-8
        rho_array[lb_active] = lb_[lb_active]+1e-8
        rho.vector()[:] = rho_array
        assign(solution.sub(0),rho)
