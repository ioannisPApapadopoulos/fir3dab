import firedrake
from firedrake import *
from firedrake.petsc import PETSc
from contextlib import ExitStack
from itertools import chain

__all__ = ["SnesDeflator", "DeflationOperator", "ShiftedDeflation", "defsolve"]

def getEdy(deflation, y, dy, vi_inact):
    with deflation.derivative(y).dat.vec as deriv:
        # if vi_inact is not None:
        #     deriv_ = deriv.getSubVector(vi_inact)
        # else:
        #     deriv_ = deriv
        deriv_ = deriv
        out = -deriv_.dot(dy)

        # if vi_inact is not None:
        #     deriv.restoreSubVector(vi_inact, deriv_)

    return out

def setSnesBounds(snes, bounds):
    (lb, ub) = bounds
    with lb.dat.vec_ro as lb_, ub.dat.vec_ro as ub_:
        snes.setVariableBounds(lb_, ub_)

def compute_tau(deflation, state, update_p, vi_inact):
    if deflation is not None:
        Edy = getEdy(deflation, state, update_p, vi_inact)

        minv = 1.0 / deflation.evaluate(state)
        tau = (1 + minv*Edy/(1 - minv*Edy))
        return tau
    else:
        return 1

class DeflatedKSP(object):
    def __init__(self, deflation, y, ksp, snes):
        self.deflation = deflation
        self.y = y
        self.ksp = ksp
        self.snes = snes
        self.its = 0

    def solve(self, ksp, b, dy_pet):
        # Use the inner ksp to solve the original problem
        self.ksp.setOperators(*ksp.getOperators())
        self.ksp.solve(b, dy_pet)
        self.its += self.ksp.its
        deflation = self.deflation

        if self.snes.getType().startswith("vi"):
            vi_inact = self.snes.getVIInactiveSet()
        else:
            vi_inact = None

        tau = compute_tau(deflation, self.y, dy_pet, vi_inact)
        dy_pet.scale(tau)

        ksp.setConvergedReason(self.ksp.getConvergedReason())

    def reset(self, ksp):
        self.ksp.reset()

    def view(self, ksp, viewer):
        self.ksp.view(viewer)

class SnesDeflator(object):
    def __init__(self, snes, deflation, u):
        self.snes = snes
        self.deflation = deflation
        self.u = u

    def __enter__(self):
        snes = self.snes
        oldksp = snes.ksp
        defksp = DeflatedKSP(self.deflation, self.u, oldksp, snes)
        snes.ksp = PETSc.KSP().createPython(defksp, snes.comm)
        snes.ksp.pc.setType('none')

        self.oldksp = oldksp
        self.defksp = defksp

    def __exit__(self, *args):
        del self.defksp.snes # clean up circular references
        self.snes.ksp = self.oldksp # restore old KSP
        self.snes.setAttr("total_ksp_its", self.defksp.its)

class DeflationOperator(object):
    """
    Base class for deflation operators.
    """
    def set_parameters(self, params):
        self.parameters = params

    def deflate(self, roots):
        self.roots = roots

    def evaluate(self):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError

class ShiftedDeflation(DeflationOperator):
    """
    The shifted deflation operator presented in doi:10.1137/140984798.
    """
    def __init__(self, power, shift, normsq=None):
        self.power = power
        self.shift = shift
        self.roots = []
        if normsq is None:
            normsq = lambda u, v, params: inner(u-v, u-v)*dx
        self.normsqc = normsq
        self.parameters = None

    def normsq(self, y, root):
        out = self.normsqc(y, root, self.parameters)
        return out

    def evaluate(self, y):
        m = 1.0
        for root in self.roots:
            normsq = assemble(self.normsq(y, root))
            factor = normsq**(-self.power/2.0) + self.shift
            m *= factor

        return m

    def derivative(self, y):
        if len(self.roots) == 0:
            deta = Function(y.function_space()).vector()
            return deta

        p = self.power
        factors  = []
        dfactors = []
        dnormsqs = []
        normsqs  = []

        for root in self.roots:
            form = self.normsq(y, root)
            normsqs.append(assemble(form))
            dnormsqs.append(assemble(derivative(form, y)))

        for normsq in normsqs:
            factor = normsq**(-p/2.0) + self.shift
            dfactor = (-p/2.0) * normsq**((-p/2.0) - 1.0)

            factors.append(factor)
            dfactors.append(dfactor)

        eta = product(factors)

        deta = Function(y.function_space()).vector()

        for (solution, factor, dfactor, dnormsq) in zip(self.roots, factors, dfactors, dnormsqs):
            dnormsq = dnormsq.vector()
            deta.axpy(float((eta/factor)*dfactor), dnormsq)

        return deta

def defsolve(nvs, deflateop=None, vi = None):

    dm = nvs.snes.getDM()
    for dbc in nvs._problem.dirichlet_bcs():
        dbc.apply(nvs._problem.u)

    if vi != None:
        nvs.snes.setType("vinewtonrsls")
        setSnesBounds(nvs.snes, vi)
    
    work = nvs._work
    with nvs._problem.u.dat.vec as u:
        u.copy(work)
        with ExitStack() as stack:
            for ctx in chain((nvs.inserted_options(),dmhooks.add_hooks(dm, nvs, appctx=nvs._ctx)),
                             nvs._transfer_operators):
                stack.enter_context(ctx)

            with SnesDeflator(nvs.snes, deflateop, nvs._problem.u):
                nvs.snes.solve(None, work)

        work.copy(u)
    
    nvs._setup = True
    success = nvs.snes.getConvergedReason() > 0
    return (success, nvs.snes.its, nvs.snes.getAttr("total_ksp_its"))
