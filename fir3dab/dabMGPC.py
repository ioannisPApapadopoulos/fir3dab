from firedrake import *
from .misc import *
import numpy

"""
Here are three classes for constructing the Schur complement corrected momentum
block that arises in our preconditioner for the deflated barrier method applied
to the topology optimization of the power dissipation of fluids in Stokes flow.

The first class is used in the geometric MG method on the fine levels. This is
not required if you are only using a 2-grid, but we are using an n-grid cycle
n >=3, then the smoother would be applying it to just the momentum block (without
the Schur complement correction).

The second class is for the coarsest grid in the geometric MG method.

The third class is a general Schur complement correction for the momentum block
"""

def alpha(rho, params):
    (gamma, alphabar,q) = params
    return alphabar * ( 1. - rho*(1+q)/(rho+q))

def alpha_cc(rho, params):
    (gamma, alphabar,q) = params
    return alphabar + (1e-5-alphabar) * rho*(1+q)/(rho+q)

class DABFineGridPC(PCBase):
    _prefix = "dab_"

    def initialize(self, opc):
        info_red("Inside FINE INITIALIZE")

        """
        1) Grab updated state and inject if necessary
        2) Formulate material distribution-velocity form
        3) Grab the active-set indices & velocity BCs
        4) Assemble material distribution-velocity 2x2 matrix
        5) Split it up
        6) Compute matrix triple-product
        7) Add correction to the momentum block
        8) Attach to a new PC which can handle MG & Star patch
        """
        _, P = opc.getOperators()
        if P.getType() == "python":
            context = P.getPythonContext()
            (a, bcs) = (context.a, context.row_bcs)
        else:
            context = dmhooks.get_appctx(opc.getDM())
            (a, bcs) = (context.Jp or context.J, context._problem.bcs)

        appctx = self.get_appctx(opc)
        fcp = appctx.get("form_compiler_parameters")
        self.fcp = fcp

        prefix = opc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "aij")
        self.mat_type = mat_type

        nref = int(PETSc.Options().getString(options_prefix + "nref", 1))
        self.nref = nref

        params = PETSc.Options().getString(options_prefix + "params")
        params = params[1:-1].split(',')
        params = [float(x) for x in params]

        gamma_al = float(PETSc.Options().getString(options_prefix + "gamma_al"))
        sigma = float(PETSc.Options().getString(options_prefix + "sigma"))

        mu = appctx["mu"]
        self.statec = appctx["state"]
        # This is passed inside the user-run script
        self.coarse_to_fine_mapping = appctx["c2f_mapping"]
        dm = opc.getDM()
        self.dm = dm
        self.mu = mu


        msh_level = dm.getRefineLevel()
        self.Zs = appctx["Zs"]
        Z = self.Zs[msh_level]
        self.Z = Z

        #print("Refinement Level %s"%dm.getRefineLevel())

        self.zc  = Function(Z)

        # Grab parent dm that has the full mixed fine-grid state and
        # active set bcs.
        dm_p = get_parent(dm)
        self.dm_p = dm_p

        # If at finest level, just assign the appropriate states.
        if msh_level == nref:
            # Assign fine-grid material distribution
            self.zc.split()[0].assign(self.statec.split()[0])
            # Assign velocity coarse-grid state
            self.zc.split()[1].assign(self.statec.split()[1])

        # If not at finest level, then need to inject the states because
        # BUG in firedrake means that the state on appctx is not the full
        # mixed state, but the state of the small block. Here we grab the
        # full mixed state from the finest-grid DM and inject it to
        # this grid.
        else:
            appctxf = get_appctx(dm_p)
            # Grab finest level state.
            self.statef = appctxf.appctx['state']
            # zc is a mixed space of material distribution x velocity
            self.zc  = Function(Z)
            # Grab inject operators
            self.dminject = dm.getAppCtx()[0].transfer_manager.inject
            # Inject fine-grid material distribution
            self.dminject(self.statef.split()[0], self.zc.split()[0])
            # Assign velocity state (automatically updated)
            self.zc.split()[1].assign(self.statec)

        # Now we need to set up the form
        (rho, u) = split(self.zc)
        trial  = TrialFunction(Z)
        test   = TestFunction(Z)
        # Extract BCs from appctx
        expr = get_appctx(dm).bcs_F[0].function_arg
        g = expr
        Re = 1

        sigma = Constant(sigma) * max(Z.sub(1).ufl_element().degree()**2, 1)
        n = FacetNormal(self.zc.ufl_domain())
        h = CellDiameter(self.zc.ufl_domain()) # CellVolume(mesh)/FacetArea(mesh)

        (gamma, alphabar,q) = params
        v = split(test)[1]
        Agrad = (
             1/Re * inner(grad(u), grad(v))*dx
           - 1/Re * inner(avg(grad(u)), 2*avg(outer(v, n))) * dS
           - 1/Re * inner(avg(grad(v)), 2*avg(outer(u, n))) * dS
           + 1/Re * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
           - 1/Re * inner(outer(v,n), grad(u))*ds
           - 1/Re * inner(outer(u-g,n), grad(v))*ds
           + 1/Re * (sigma/h)*inner(v,u-g)*ds
        )

        lb = Constant(0-1e-5)
        ub = Constant(1+1e-5)
        L = (
            + 0.5 * gamma_al*inner(div(u), div(u))*dx
            + 0.5 * alpha(rho, params) * inner(u, u)*dx
            - mu*ln(rho - lb)*dx  - mu*ln(ub - rho)*dx
            )
        F = derivative(L, self.zc, test) + Agrad
        J = derivative(F, self.zc, trial)
        self.J = J

        bcs = [DirichletBC(Z.sub(1), expr, "on_boundary")]
        self.bcs = bcs

        # Indices list for the active-set identification.
        ises = Z.dof_dset.field_ises
        self.ises = ises

        # Grab any active-set associated BCs.
        active_bcs = []
        for bc in self.dm_p.appctx[0].bcs_F:
            if type(bc) is firedrake.matrix_free.operators.ZeroRowsColumnsBC:
                active_bcs.append(bc)

        # If there are any active-set bcs, then time to do some work.
        if len(active_bcs) > 0:
            active_nodes = active_bcs[0].nodes
            bc_rho = DirichletBC(Z.sub(0), Constant(0), "on_boundary")

            if msh_level == nref:
                # If at finest level, then we can just assign the active-set nodes
                bc_rho.nodes = active_nodes
            else:
                # If not at finest level, we need to do some work...
                # Run through list of active-set fine-grid nodes and return which row they are
                # contained in inside the coarse_to_fine mapping. The row tells us which coarse
                # cell those active-set fine cells are contained in.

                # Function that keeps rows that appear for than 'occ' number of times.
                occurence_count = lambda nodes, occ: [x for x in set(nodes) if nodes.count(x) >= occ]

                # Keep coarse cells that contain 2 (if 2D) or 4 (if 3D) or more finer-grid active set cells.
                top_dim = Z.mesh().topological_dimension()
                if top_dim == 2:
                    occ_list = [2]*(nref - msh_level)
                elif top_dim == 3:
                    occ_list = [4]*(nref - msh_level)
                else:
                    raise("The dimension %s is not supported in dabMGPC"%top_dim)

                for i, occ in zip(reversed(range(nref)), occ_list):
                    coarse_active_nodes = []
                    # FIXME: The below for loop is very unhealthy. Looping over all active-set active_nodes
                    # is very inefficient. Definitely can speed this up.
                    for n in active_nodes:
                        row = numpy.where(numpy.any(self.coarse_to_fine_mapping[i] == n, axis=1))
                        row = row[0]
                        if len(row)> 0:
                            coarse_active_nodes.append(row[0])
                    # Only keep coarse cells that contain occ or more finer-grid active set cells
                    # as given in the occ_list.
                    coarse_active_nodes = occurence_count(coarse_active_nodes, occ)
                    active_nodes = coarse_active_nodes

                active_nodes = numpy.array(coarse_active_nodes)
                bc_rho.nodes = active_nodes

        # If active-set non-empty, add to bcs list
        if len(active_bcs) > 0:
            bcs.extend(bc_rho)

        # Assemble 2x2 material distribution-velocity block matrix
        Jz = assemble(J, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type)
        A  = Jz.petscmat.createSubMatrix(ises[1],ises[1]) # momentum block
        C  = Jz.petscmat.createSubMatrix(ises[0],ises[0]) # material distribution (diagonal) block
        D  = Jz.petscmat.createSubMatrix(ises[1],ises[0]) # bottom left block
        DT = Jz.petscmat.createSubMatrix(ises[0],ises[1]) # top right block
        # Now to compute Schur complement. Need to invert C but it's a diagonal matrix,
        # so we store the inverse of the diagonal in a vector and then set the diagonal of
        # C to be that vector
        vec = C.createVecRight()
        vec.setArray(C.invertBlockDiagonal())
        C.setDiagonal(vec)      # C_mu^-1
        CinvDT = C.matMult(DT)  # C_mu^-1 D^T
        S = D.matMult(CinvDT)   # D C_mu^-1 D^T
        A.axpy(-1, S)           # A - D C_mu^-1 D^T

        # A now holds Schur complement corrected momentum block
        self.A = A
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        # This PC set up allows for MG and PatchPC solvers :) Stolen from
        # firedrake.AssembledPC
        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        dm = opc.getDM()
        octx = get_appctx(dm)
        oproblem = octx._problem
        nproblem = NonlinearVariationalProblem(oproblem.F, oproblem.u, bcs, J=a, form_compiler_parameters=fcp)
        self._ctx_ref = _SNESContext(nproblem, mat_type, mat_type, octx.appctx, options_prefix=options_prefix)
        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(self.A, self.A)
        self.pc = pc
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            self.pc.setFromOptions()

    def update(self, pc):
        info_red("Inside FINE UPDATE")
        # FIXME: update method is not being called :(

        if self.nref == self.dm.getRefineLevel():
            # Assign updated velocity coarse-grid state if at finest level
            self.zc.split()[0].assign(self.statec.split()[0])
            self.zc.split()[1].assign(self.statec.split()[1])
        else:
            # Inject updated fine-grid material distribution
            self.dminject(self.statef.split()[0], self.zc.split()[0])
            # Assign updated velocity coarse-grid state
            self.zc.split()[1].assign(self.statec)

        # No need to recreate the form, J should be using self.zc which has been
        # updated so can just reassemble with new active-row bcs
        J = self.J
        ises = self.ises

        fcp = self.fcp
        mat_type = self.mat_type
        bcs = self.bcs

        # Compute Schur complement correction as in initialize()
        Jz = assemble(J, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type)
        A  = Jz.petscmat.createSubMatrix(ises[1],ises[1]) # momentum block
        C  = Jz.petscmat.createSubMatrix(ises[0],ises[0]) # material distribution (diagonal) block
        D  = Jz.petscmat.createSubMatrix(ises[1],ises[0]) # bottom left block
        DT = Jz.petscmat.createSubMatrix(ises[0],ises[1]) # top right block

        vec = C.createVecRight()
        vec.setArray(C.invertBlockDiagonal())
        C.setDiagonal(vec)
        CinvDT = C.matMult(DT)
        S = D.matMult(CinvDT)
        A.axpy(-1, S)

        # By copying A to self.A, self.pc should update its Operator and trigger a new
        # MUMPS LU factorization
        A.copy(self.A)

    def apply(self, pc, x, y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(x, y)

class DABCoarseGridPC(PCBase):

    _prefix = "dab_"

    def initialize(self, opc):
        info_red("Inside COARSE INITIALIZE")

        """
        1) Grab updated state and inject if necessary
        2) Formulate material distribution-velocity form
        3) Grab the active-set indices & velocity BCs
        4) Assemble material distribution-velocity 2x2 matrix
        5) Split it up
        6) Compute matrix triple-product
        7) Add correction to the momentum block
        8) Attach to a new PC
        """

        appctx = self.get_appctx(opc)
        fcp = appctx.get("form_compiler_parameters")
        self.fcp = fcp

        prefix = opc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "aij")
        self.mat_type = mat_type

        nref = int(PETSc.Options().getString(options_prefix + "nref", 1))
        self.nref = nref

        params = PETSc.Options().getString(options_prefix + "params")
        params = params[1:-1].split(',')
        params = [float(x) for x in params]

        gamma_al = float(PETSc.Options().getString(options_prefix + "gamma_al"))
        sigma = float(PETSc.Options().getString(options_prefix + "sigma"))

        mu = appctx["mu"]
        self.statec = appctx["state"]
        self.coarse_to_fine_mapping = appctx["c2f_mapping"]
        self.mu = mu

        # BUG in firedrake means that the state on appctx is not the full
        # mixed state, but the state of the small block. Here we grab the
        # full mixed state from the fine-grid DM and inject it to the
        # coarse grid.

        # Grab pre-prepared FunctionSpace of material distribution x velocity.
        self.Zs = appctx["Zs"]
        Z = self.Zs[0]
        self.Z = Z
        dm = opc.getDM()
        # Grab parent dm that has the full mixed fine-grid state
        dm_p = get_parent(dm)
        self.dm_p = dm_p
        appctxf = dm_p.getAppCtx()[0]
        # Grab full mixed fine-grid state
        self.statef = appctxf.appctx['state']
        # zc is a mixed space of material distribution x velocity
        self.zc  = Function(Z)
        # grab inject and restrict operators
        self.dminject = dm.getAppCtx()[0].transfer_manager.inject

        # Inject fine-grid material distribution
        self.dminject(self.statef.split()[0], self.zc.split()[0])
        # Assign velocity coarse-grid state
        self.zc.split()[1].assign(self.statec)

        # Now we need to set up the form
        (rho, u) = split(self.zc)
        trial  = TrialFunction(Z)
        test   = TestFunction(Z)
        # Extract BCs from appctx
        expr = get_appctx(dm).bcs_F[0].function_arg
        g = expr
        Re = 1

        sigma = Constant(sigma) * max(Z.sub(1).ufl_element().degree()**2, 1)
        n = FacetNormal(self.zc.ufl_domain())
        h = CellDiameter(self.zc.ufl_domain()) # CellVolume(mesh)/FacetArea(mesh)

        (gamma, alphabar,q) = params
        v = split(test)[1]
        Agrad = (
             1/Re * inner(grad(u), grad(v))*dx
           - 1/Re * inner(avg(grad(u)), 2*avg(outer(v, n))) * dS
           - 1/Re * inner(avg(grad(v)), 2*avg(outer(u, n))) * dS
           + 1/Re * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
           - 1/Re * inner(outer(v,n), grad(u))*ds
           - 1/Re * inner(outer(u,n), grad(v))*ds
           + 1/Re * (sigma/h)*inner(v,u)*ds
        )

        lb = Constant(0-1e-5)
        ub = Constant(1+1e-5)
        L = (
            + 0.5 * gamma_al*inner(div(u), div(u))*dx
            + 0.5 * alpha(rho, params) * inner(u, u)*dx
            - mu*ln(rho - lb)*dx  - mu*ln(ub - rho)*dx
            )
        F = derivative(L, self.zc, test) + Agrad
        J = derivative(F, self.zc, trial)
        self.J = J
        ises = Z.dof_dset.field_ises
        self.ises = ises

        bcs = [DirichletBC(Z.sub(1), expr, "on_boundary")]
        self.bcs = bcs

        # Coarse grid active-set will be determined via the fine-grid active-set.
        # If a coarse cell contains 'n' or more fine cells in the active-set
        # that coarse cell is also in the active set. Tests show that a good choice is
        # n = 3 in 2D.

        # First extract the BCs that contain the active-set nodes
        active_bcs = []
        for bc in self.dm_p.appctx[0].bcs_F:
            if type(bc) is firedrake.matrix_free.operators.ZeroRowsColumnsBC:
                active_bcs.append(bc)

        # Find coarse-grid active-set for the material distribution
        # Extract nodes from BCs
        if len(active_bcs) > 0:
            active_nodes = active_bcs[0].nodes
        bc_rho = DirichletBC(Z.sub(0), Constant(0), "on_boundary")
        # Run through list of active-set fine-grid nodes and return which row they are
        # contained in inside the coarse_to_fine mapping. The row tells us which coarse
        # cell those active-set fine cells are contained in.

        # Function that keeps rows that appear for than 'occ' number of times.
        occurence_count = lambda nodes, occ: [x for x in set(nodes) if nodes.count(x) >= occ]

        # If 2D keep coarse cells that contain 2 or more finer-grid active set cells until the
        # final coarsest grid where we only keep the cells with 3 or more finer grid cells.
        # Same if 3D expect 4 and then 5.
        top_dim = Z.mesh().topological_dimension()
        if top_dim == 2:
            occ_list = [2]*(nref - 1)
            occ_list.append(2)
        elif top_dim == 3:
            occ_list = [4]*(nref - 1)
            occ_list.append(4)
        else:
            raise("The dimension %s is not supported in dabMGPC"%top_dim)

        if len(active_bcs) > 0:
            for i, occ in zip(reversed(range(nref)), occ_list):
                coarse_active_nodes = []
                # FIXME: inefficient loop
                for n in active_nodes:
                    row = numpy.where(numpy.any(self.coarse_to_fine_mapping[i] == n, axis=1))
                    row = row[0]
                    if len(row)> 0:
                        coarse_active_nodes.append(row[0])
                # Only keep coarse cells that contain occ or more finer-grid active set cells
                # as given in the occ_list.
                coarse_active_nodes = occurence_count(coarse_active_nodes, occ)
                active_nodes = coarse_active_nodes

            active_nodes = numpy.array(coarse_active_nodes, dtype="int32")
            bc_rho.nodes = active_nodes

        self.bc_rho = bc_rho
        # If active-set non-empty, add to bcs list
        if len(active_bcs) > 0:
            bcs.append(bc_rho)
        # Assemble 2x2 block matrix
        Jz = assemble(J, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type)
        A  = Jz.petscmat.createSubMatrix(ises[1],ises[1]) # momentum block
        C  = Jz.petscmat.createSubMatrix(ises[0],ises[0]) # material distribution (diagonal) block
        D  = Jz.petscmat.createSubMatrix(ises[1],ises[0]) # bottom left block
        DT = Jz.petscmat.createSubMatrix(ises[0],ises[1]) # top right block
        # Now to compute Schur complement. Need to invert C but it's a diagonal matrix,
        # so we store the inverse of the diagonal in a vector and then set the diagonal of
        # C to be that vector
        vec = C.createVecRight()
        vec.setArray(C.invertBlockDiagonal())
        C.setDiagonal(vec)      # C_mu^-1
        CinvDT = C.matMult(DT)  # C_mu^-1 D^T
        S = D.matMult(CinvDT)   # D C_mu^-1 D^T
        A.axpy(-1, S)           # A - D C_mu^-1 D^T

        # Now create PETSc PC, use MUMPS LU to invert it
        self.A = A
        self.pc = PETSc.PC().create(comm=opc.comm)
        self.pc.incrementTabLevel(1, parent=opc)
        self.pc.setOperators(self.A, self.A)
        self.pc.setType("lu")
        self.pc.setFactorSolverType("mumps")
        self.pc.setUp()
        opc = self.pc


    def update(self, pc):
        info_red("Inside COARSE UPDATE")

        # Inject updated fine-grid material distribution
        self.dminject(self.statef.split()[0], self.zc.split()[0])
        # Assign updated velocity coarse-grid state
        self.zc.split()[1].assign(self.statec)

        # No need to recreate the form, J should be using self.zc which has been
        # updated so can just reassemble with new active-row bcs
        J = self.J
        ises = self.ises

        # If just updating, then the fine-grid active set has not changed, so no need
        # to recompute the coarse-grid active-set. Just keep the same from the previous
        # SNES iteration.
        bcs = self.bcs

        fcp = self.fcp
        mat_type = self.mat_type

        # Compute Schur complement correction as in initialize()
        Jz = assemble(J, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type)
        A  = Jz.petscmat.createSubMatrix(ises[1],ises[1]) # momentum block
        C  = Jz.petscmat.createSubMatrix(ises[0],ises[0]) # material distribution (diagonal) block
        D  = Jz.petscmat.createSubMatrix(ises[1],ises[0]) # bottom left block
        DT = Jz.petscmat.createSubMatrix(ises[0],ises[1]) # top right block

        vec = C.createVecRight()
        vec.setArray(C.invertBlockDiagonal())
        C.setDiagonal(vec)
        CinvDT = C.matMult(DT)
        S = D.matMult(CinvDT)
        A.axpy(-1, S)

        # By copying A to self.A, self.pc should update its Operator and trigger a new
        # MUMPS LU factorization
        A.copy(self.A)

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)

class DABSchurComplementCorrectionPC(PCBase):
    _prefix = "dab_"

    def initialize(self, opc):
        info_red("Inside Schur complement correction INITIALIZE")

        """
        1) Grab updated state
        2) Formulate material distribution-velocity form
        3) Grab the active-set indices & velocity BCs
        4) Assemble material distribution-velocity 2x2 matrix
        5) Split it up
        6) Compute matrix triple-product
        7) Add correction to the momentum block
        8) Attach to a new PC
        """
        _, P = opc.getOperators()
        if P.getType() == "python":
            context = P.getPythonContext()
            (a, bcs) = (context.a, context.row_bcs)
        else:
            context = dmhooks.get_appctx(opc.getDM())
            (a, bcs) = (context.Jp or context.J, context._problem.bcs)

        appctx = self.get_appctx(opc)
        fcp = appctx.get("form_compiler_parameters")
        self.fcp = fcp

        prefix = opc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "aij")
        self.mat_type = mat_type

        params = PETSc.Options().getString(options_prefix + "params")
        params = params[1:-1].split(',')
        params = [float(x) for x in params]

        mu = appctx["mu"]
        self.statec = appctx["state"]
        # This is passed inside the user-run script
        dm = opc.getDM()
        self.dm = dm
        self.mu = mu


        self.Zs = appctx["Zs"]
        Z = self.Zs[0]
        self.Z = Z

        self.zc  = Function(Z)

        # Grab parent dm that has the full mixed fine-grid state and
        # active set bcs.
        dm_p = get_parent(dm)
        self.dm_p = dm_p

        # Assign fine-grid material distribution
        self.zc.split()[0].assign(self.statec.split()[0])
        # Assign velocity coarse-grid state
        self.zc.split()[1].assign(self.statec.split()[1])

        # Now we need to set up the form
        (rho, u) = split(self.zc)
        trial  = TrialFunction(Z)
        test   = TestFunction(Z)
        # Extract BCs from appctx
        expr = get_appctx(dm).bcs_F[0].function_arg
        g = expr
        Re = 1

        lb = Constant(0-1e-5)
        ub = Constant(1+1e-5)
        L = (
            + 0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * alpha_cc(rho, params) * inner(u, u)*dx
            - mu*ln(rho - lb)*dx  - mu*ln(ub - rho)*dx
            )
        F = derivative(L, self.zc, test)
        J = derivative(F, self.zc, trial)
        self.J = J

        bcs = [DirichletBC(Z.sub(1), expr, "on_boundary")]
        self.bcs = bcs

        # Indices list for the active-set identification.
        ises = Z.dof_dset.field_ises
        self.ises = ises

        # Grab any active-set associated BCs.
        active_bcs = []
        for bc in self.dm_p.appctx[0].bcs_F:
            if type(bc) is firedrake.matrix_free.operators.ZeroRowsColumnsBC:
                active_bcs.append(bc)

        # If there any any active-set bcs, then time to do some work.
        if len(active_bcs) > 0:
            active_nodes = active_bcs[0].nodes
            bc_rho = DirichletBC(Z.sub(0), Constant(0), "on_boundary")
            bc_rho.nodes = active_nodes
            bcs.extend(bc_rho)


        # Assemble 2x2 material distribution-velocity block matrix
        Jz = assemble(J, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type)
        A  = Jz.petscmat.createSubMatrix(ises[1],ises[1]) # momentum block
        C  = Jz.petscmat.createSubMatrix(ises[0],ises[0]) # material distribution (diagonal) block
        D  = Jz.petscmat.createSubMatrix(ises[1],ises[0]) # bottom left block
        DT = Jz.petscmat.createSubMatrix(ises[0],ises[1]) # top right block
        # Now to compute Schur complement. Need to invert C but it's a diagonal matrix,
        # so we store the inverse of the diagonal in a vector and then set the diagonal of
        # C to be that vector
        vec = C.createVecRight()
        vec.setArray(C.invertBlockDiagonal())
        C.setDiagonal(vec)      # C_mu^-1
        CinvDT = C.matMult(DT)  # C_mu^-1 D^T
        S = D.matMult(CinvDT)   # D C_mu^-1 D^T
        A.axpy(-1, S)           # A = A - D C_mu^-1 D^T

        # A now holds Schur complement corrected momentum block
        self.A = A
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        # This PC set up allows for MG and PatchPC solvers :) Stolen from
        # firedrake.AssembledPC
        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        dm = opc.getDM()
        octx = get_appctx(dm)
        oproblem = octx._problem
        nproblem = NonlinearVariationalProblem(oproblem.F, oproblem.u, bcs, J=a, form_compiler_parameters=fcp)
        self._ctx_ref = _SNESContext(nproblem, mat_type, mat_type, octx.appctx, options_prefix=options_prefix)
        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(self.A, self.A)
        self.pc = pc
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            self.pc.setFromOptions()

    def update(self, pc):
        info_red("Inside Schur complement corrected UPDATE")

        # Assign updated velocity coarse-grid state if at finest level
        self.zc.split()[0].assign(self.statec.split()[0])
        self.zc.split()[1].assign(self.statec.split()[1])

        # No need to recreate the form, J should be using self.zc which has been
        # updated so can just reassemble with new active-row bcs
        J = self.J
        ises = self.Z.dof_dset.field_ises
        self.ises = ises

        fcp = self.fcp
        mat_type = self.mat_type
        bcs = self.bcs

        # Compute Schur complement correction as in initialize()
        Jz = assemble(J, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type)
        A  = Jz.petscmat.createSubMatrix(ises[1],ises[1]) # momentum block
        C  = Jz.petscmat.createSubMatrix(ises[0],ises[0]) # material distribution (diagonal) block
        D  = Jz.petscmat.createSubMatrix(ises[1],ises[0]) # bottom left block
        DT = Jz.petscmat.createSubMatrix(ises[0],ises[1]) # top right block

        vec = C.createVecRight()
        vec.setArray(C.invertBlockDiagonal())
        C.setDiagonal(vec)
        CinvDT = C.matMult(DT)
        S = D.matMult(CinvDT)
        A.axpy(-1, S)

        # By copying A to self.A, self.pc should update its Operator and trigger a new
        # MUMPS LU factorization
        A.copy(self.A)

    def apply(self, pc, x, y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(x, y)
