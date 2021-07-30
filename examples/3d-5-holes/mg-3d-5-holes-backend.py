# -*- coding: utf-8 -*-
from firedrake import *
from fir3dab import *
from alfi import *
from alfi.transfer import *

"""
This script grid-sequences the solutions found on the coarse-mesh by 3d-5-holes.

As before the linear systems are block preconditioned and reduced to the following:

1. Solve the diagonal material distribution block
2. Solve the block-diagonal pressure mass matrix
3. Solve the augmented momentum block

1. and 2. are inverted with MUMPS LU but 3. is now approximated my GMRES preconditioned
with a geometric MG cycle with star patch relaxation and a representation of the
active set on the coarse level.
"""
width = 1.5
nref = 1       # number of refinements of the base mesh
gamma_al = 1e5 # augmented Lagrangian parameter
branch = 0     # branch to gridsequence
dgtransfer = DGInjection() # some transfer operators from alfi

def InflowOutflow(mesh):
    # Boundary conditions of 3D 5-holes example. This models 4 inlets
    # and 4 outlets.
    x = SpatialCoordinate(mesh)
    r = sqrt(1./(12.*pi))
    gbar = 1.0

    first_pipe = lt(((x[1] - 1.0/4)/r)**2 + ((x[2] - 1.0/4)/r)**2, 1.0)
    second_pipe = lt(((x[1] - 3.0/4)/r)**2 + ((x[2] - 1.0/4)/r)**2, 1.0)
    third_pipe = lt(((x[1] - 3.0/4)/r)**2 + ((x[2] - 3.0/4)/r)**2, 1.0)
    fourth_pipe = lt(((x[1] - 1.0/4)/r)**2 + ((x[2] - 3.0/4)/r)**2, 1.0)
    x_on_boundary = Or(lt(x[0], 1e-10), gt(x[0], width-1e-10))

    val_in_first_pipe   = gbar*(1 - ( ((x[1] - 1.0/4)/r)**2 + ((x[2] - 1.0/4)/r)**2 ))
    val_in_second_pipe  = gbar*(1 - ( ((x[1] - 3.0/4)/r)**2 + ((x[2] - 1.0/4)/r)**2 ))
    val_in_third_pipe   = gbar*(1 - ( ((x[1] - 3.0/4)/r)**2 + ((x[2] - 3.0/4)/r)**2 ))
    val_in_fourth_pipe  = gbar*(1 - ( ((x[1] - 1.0/4)/r)**2 + ((x[2] - 3.0/4)/r)**2 ))

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

class BDMTransferManager(TransferManager):
    # Transfer operators for robust MG cycle
   def __init__(self, *, native_transfer=None, use_averaging=True):
        self.native_transfers = {VectorElement("DG", tetrahedron, 1, dim=3): (dgtransfer.prolong, restrict, dgtransfer.inject),
                                 FiniteElement("DG", tetrahedron, 0): (prolong, restrict, inject),
                                 FiniteElement("R", tetrahedron, 0): (prolong, restrict, inject),
                                 }

        self.use_averaging = use_averaging
        self.caches = {}

# before and after methods are for mesh hierarchy generation
def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        # arguments for mesh hierarchy
        distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
        # load mesh generated in Gmsh. Box domain (0,0,0)x(1.5,1,1) with 5
        # cuboid holes cut out at x=3/4.
        mesh = Mesh("mesh-5-holes.msh", comm=comm, distribution_parameters = distribution_parameters)
        # generate 2-level mesh hierarchy
        self.mh = MeshHierarchy(mesh, nref, reorder=True, callbacks=(before,after))
        mesh = self.mh[-1]
        # The coarse to fine element mappings are required in the MG cycle in order
        # to represent the active set on the coarser levels. We choose to save this
        # mapping here to self.
        self.c2f_mapping = self.mh.coarse_to_fine_cells
        return mesh

    def function_space(self, mesh):
        # (Integral-evaluated) first-order BDM discretization for the velocity
        # and pressure pair, with DG0 for the material distribution.
        Ve = FiniteElement("BDM", mesh.ufl_cell(), 1, variant = "integral") # velocity
        Pe = FiniteElement("DG", mesh.ufl_cell(), 0) # pressure
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # control
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals

        Ze = MixedElement([Ce, Ve, Pe, Re])
        self.Ze = Ze

        Z  = FunctionSpace(mesh, Ze)
        self.Z = Z

        # Here we save the function spaces at the different levels to self.
        # This is utilized later on during the MG cycle in order to assemble
        # the augmented momentum block.
        self.function_space_hierarchy = []
        for i in range(nref+1):
            self.function_space_hierarchy.append(FunctionSpace(self.mh[i], MixedElement([Ce,Ve])))

        info_green("Number of degrees of freedom: ", Z.dim())
        # Take some data. First, BCs
        self.expr = InflowOutflow(mesh)
        return Z

    def gradIP(self, z, w, index):
        # This method is an implementation of the broken gradient.
        u = split(z)[index]; v = split(w)[index]

        Z = z.function_space()
        mesh = Z.mesh()
        Re = 1
        g = self.expr

        sigma = Constant(1000) * max(Z.sub(index).ufl_element().degree()**2, 1)
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
            + 0.5 * gamma_al*inner(div(u), div(u))*dx
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

        sigma = Constant(1e3) * max(Z.sub(1).ufl_element().degree()**2, 1)
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
        Load saved initial guess.
        """
        comm = Z.comm
        z = []
        for branches in range(14):
            z.append(Function(Z))
            h5 = HDF5File("initial-guess/%s.xml.gz"%branches, "r", comm=comm)
            h5.read(z[branches], "/guess")
            del h5
        return [z[branch]]

    def number_solutions(self, mu, params):
        return 1

    def update_mu(self, z, mu, iters, k, k_mu_old, params):
        etol = 1e-15
        if float(mu) > 60:
            k_mu = 0.9
        else:
            k_mu = 0.9
        theta_mu = 1.5
        next_mu = max(etol/10., min(k_mu*mu, mu**theta_mu))
        return next_mu


    def solver_parameters(self, mu, branch, task, params):
        (gamma,alphabar, q) = params
        linesearch = "l2"
        max_it = 100
        damping = 1.0
        if float(mu)!=0.0:
            snes_atol = 1e-4
        else:
            snes_atol = 1e-6
        args = {
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
                "ksp_gmres_restart": 500,
                "ksp_gmres_modifiedgramschmidt": None,
                "ksp_monitor_true_residual": None,
                "ksp_converged_reason": None,
                "ksp_atol": 1e-6 if float(mu) > 0 else 1e-7,
                "ksp_rtol": 1e-5 if float(mu) > 0 else 1e-5,

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

                        # material distribution block is a dressed-up mass matrix and should
                        # not be difficult to invert, can use CG or CG preconditioned with ILU
                        "fieldsplit_0":{
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                            "mat_mumps_icntl_14": 500,},

                        # This approximation can be approximated by another Schur complement
                        # factorization since it looks like Stokes-Brinkman! The top left block is
                        # the momentum block
                        "fieldsplit_1":{
                            "ksp_type": "preonly",
                            "pc_type": "fieldsplit",
                            "pc_fieldsplit_type": "schur",
                            "pc_fieldsplit_schur_fact_type": "full",
                            "pc_fieldsplit_0_fields": "0",
                            "pc_fieldsplit_1_fields": "1",

                            # The momentum block is approximated by a specialized multigrid scheme
                            "fieldsplit_0":{
                                "ksp_type": "fgmres",
                                "ksp_max_it": 20,
                                "ksp_converged_reason": None,
                                #"ksp_monitor_true_residual": None,
                                "ksp_atol": 1e-8,
                                "ksp_rtol": 1e-9,
                                "ksp_gmres_restart": 500,
                                "ksp_gmres_modifiedgramschmidt": None,

                                "pc_type": "mg",
                                "pc_mg_type": "full",
                                # This is redundant (but we keep it around just in case), the
                                # transfer operator is picked up higher up
                                "mg_transfer_manager": __name__ + ".BDMTransferManager",

                                # DABCoarseGridPC is a class implemented in the
                                # fir3dab library. It correctly assembles the
                                # augmented momentum block on the coarsest level
                                # including the representation of the active set.
                                "mg_coarse":{
                                    "ksp_type": "preonly",
                                    "pc_type": "python",
                                    "pc_python_type": __name__ + ".DABCoarseGridPC",
                                    "dab": {
                                        "nref": nref,
                                        "params": params,
                                        "sigma": 1e3,
                                        "gamma_al": gamma_al,},
                                },
                                # DABFineGridPC is a class implemented in the
                                # fir3dab library. It correctly assembles the
                                # augmented momentum block on the finer levels
                                # including the representation of the active set.
                                # It can also be composed with exotic solver, e.g.
                                # asm patch solver.
                                "mg_levels": {
                                    "ksp_type": "gmres",
                                    "ksp_norm_type": "unpreconditioned",
                                    "ksp_convergence_test": "skip",
                                    #"ksp_monitor_true_residual": None,
                                    "ksp_max_it": 5,
                                    "pc_type": "python",
                                    "pc_python_type": __name__ + ".DABFineGridPC",
                                    "dab": {
                                       "nref": nref,
                                       "params": params,
                                       "gamma_al": gamma_al,
                                       "sigma": 1e3,
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
                                       "patch_sub_pc_factor_shift_type": "nonzero",
                                       #"patch_sub_pc_factor_mat_solver_type": "umfpack",
                                    },
                                },
                            },
                    # Due to the augmented Lagrangian term, this Schur complement looks like
                    # the pressure mass matrix * 1/gamma_al. For now we invert the
                    # mass matrix with LU, but can be done many others ways quite easily
                            "fieldsplit_1_ksp_type": "preonly",
                            "fieldsplit_1_ksp_norm_type": "unpreconditioned",
                            "fieldsplit_1_pc_type": "python",
                            "fieldsplit_1_pc_python_type": __name__ + ".Mass",
                            "fieldsplit_1_aux_pc_type": "lu",
                            "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
                            "fieldsplit_1_aux_mat_mumps_icntl_14": 500,
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
        for i in range(1,2):
            lb.split()[i].vector().set_local(-inf)
            ub.split()[i].vector().set_local(+inf)
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
    params = [0.2, 2.5e4, 0.1] #(gamma, alphabar, q)
    solutions = deflatedbarrier(problem, params, mu_start=1e-6, mu_end = 1e-5, max_halfstep = 0)
