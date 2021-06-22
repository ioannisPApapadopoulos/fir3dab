# -*- coding: utf-8 -*-
from firedrake import *
from .mlogging import *
from .deflation import defsolve
from .misc import create_output_folder, inertia_switch, report_profile, MorYos
from copy import deepcopy
import os
import resource
import shutil
from firedrake.petsc import PETSc

def deflatedbarrier(problem, params=None, comm=COMM_WORLD, mu_start=1000,
                  mu_end = 1e-15, hint=None,max_halfstep = 1,
                  initialstring=None, premature_termination=False,
                  saving_folder = ""):

    # inelegant ways of trying to make the output folder...
    create_output_folder(saving_folder)

    iter_subproblem = 0 # start iteration count

    (mu, mesh, FcnSpace, dm) = FEMsetup(comm, problem, mu_start)

    guesses = extractguesses(problem, initialstring, FcnSpace, params, comm)
    number_initial_branches = len(guesses)

    Max_solutions = problem.number_solutions(0, params)
    found_solutions = [1]*number_initial_branches
    hmin = FcnSpace.dim()

    #FIXME smart way of knowing max number of solutions?
    [guesses, found_solutions] = initialise_guess(guesses, Max_solutions, found_solutions, FcnSpace, params)

    hint = [[None, 0.0]]*Max_solutions
    hint_guess = [[None, 0.0]]*Max_solutions

    oldguesses = [guess.copy(deepcopy=True) for guess in guesses]
    deflation = problem.deflation_operator(params)

    u = Function(FcnSpace)
    v = TestFunction(FcnSpace)
    w = TrialFunction(FcnSpace)
    bcs = problem.boundary_conditions(FcnSpace, params)
    nref = 0 # will be useful when doing refinements in script

    oldmu = Constant(0)
    halfmu = Constant(0)
    num_halfstep = 0
    half = 0
    branch_deflate_start = 0
    inertia = {}
    inertia_old = {}
    # the bounds that are passed to the solver
    vi = problem.bounds_vi(FcnSpace, mu, params)

    # Book-keeping
    (Log, runenv) = createLog(Max_solutions)
    solver = "BensonMunson"

    # start loop!
    while True:
        info_red("Memory used: %s" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sum(found_solutions) == 0:
            info_red("All branches have failed, algorithm stopping.")
            break

        info_blue("Considering mu = %s" % float(mu))
        deflation.deflate([])

        # the bounds used in the barrier method
        (lb, ub)  = problem.bounds(mesh, mu, params)
        # Remove issues with a strict relative interior
        # lb.assign(float(lb)-1e-9*max(1,abs(assemble(lb*dx(mesh)))))
        # ub.assign(float(ub)+1e-9*max(1,abs(assemble(ub*dx(mesh)))))

        # FIXME why does this have to be within the while loop?
        nvs_cont = problem.solver(u, lb, ub, mu, nref, params, "ContinuationTask")
        nvs_defl = problem.solver(u, lb, ub, mu, nref, params, "DeflationTask")
        solutions = []

        # increment iteration index
        iter_subproblem += 1

        # the branch to be deflated from in DeflationTask
        branch_deflate = branch_deflate_start

        solutionpath_string = (saving_folder + "output/mu-%.12e-dofs-%s-params-%s-solver-%s" %(float(mu),hmin,params,solver))
        pvd = File("%s/solutions.pvd" %solutionpath_string)

        max_solutions = problem.number_solutions(mu, params)
        # which branches to run through
        branch_iter = 0
        branches = initialise_branches(found_solutions, Max_solutions, max_solutions)
        max_solutions = max(max_solutions, sum(found_solutions))


        while branch_iter < len(branches):
             outputarg = PredictionCorrectionDeflation(iter_subproblem, problem, FcnSpace, mesh,
                                                      u, v, w, lb, ub, params, bcs, dm, deflation,
                                                      nvs_cont, nvs_defl, comm,
                                                      branches, branch_iter, solutionpath_string,
                                                      branch_deflate_start, branch_deflate,
                                                      hint_guess, hint, guesses,
                                                      mu, oldmu, oldguesses, found_solutions, solutions,
                                                      pvd, inertia, inertia_old,
                                                      max_solutions, Max_solutions, halfmu, num_halfstep, max_halfstep, hmin, Log,
                                                      runenv)

             (branch_iter, branch_deflate, mu, oldmu, oldguesses, Log, found_solutions,
             deflation, halfmu, num_halfstep, max_solutions, guesses, task) = outputarg

        for branch in range(Max_solutions):
            if found_solutions[branch] == 1:
                Log["solutions_cost"][branch].append(problem.cost(guesses[branch], params))
            else:
                Log["solutions_cost"][branch].append("NaN")


        # Live simplistic tracking of data
        Log["mus"].append(float(mu))
        save = open(saving_folder + "output/DABlog.txt","w")
        for i in range(Max_solutions):
            save.write( "%s,"%i+str(Log["solutions_cost"][i])[1:-1] +"\n" )
        save.write( "NaN," + str(Log["mus"])[1:-1] + "\n")
        save.write( "SNES-cont-%s" %str(Log["num_snes_its"])+"\n")
        save.write( "SNES-defl-%s" %str(Log["num_snes_its_defl"])+"\n" )
        save.write( "SNES-pred-%s" %str(Log["num_snes_its_pred"])+"\n" )
        save.write( "SNES-cont-failed-%s" %str(Log["num_snes_its_failed"])+"\n")
        save.write( "KSP-cont-%s" %str(Log["num_ksp_its"])+"\n" )
        save.write( "KSP-defl-%s" %str(Log["num_ksp_its_defl"])+"\n" )
        save.write( "KSP-pred-%s" %str(Log["num_ksp_its_pred"])+"\n" )
        save.write( "Time-taken-%s" %str(Log["run_time"])+"\n" )
        save.close()


        if float(mu) == 0.0:
            info_green("Terminating because we have reached target mu")
            break
        if float(mu) <= float(mu_end):
            if premature_termination:
                info_green("Terminating because we have reached target mu")
                break
            mu.assign(Constant(0.0))

        newparams = UpdateSubproblemParams(oldguesses, guesses, Max_solutions,
                                task, num_halfstep, max_halfstep,
                                hint_guess, hint, hmin,
                                FcnSpace, params, u, problem,
                                oldmu, mu, halfmu, iter_subproblem, Log)

        (oldguesses, hint, oldmu, mu, num_halfstep) = newparams

    out = report_profile(Log, Max_solutions)
    return (guesses, out)

def initialise_guess(guesses, Max_solutions,found_solutions, V, params):
    number_initial_branches = len(guesses)
    if (Max_solutions - number_initial_branches) > 0:
        for i in range(number_initial_branches,Max_solutions):
            guesses.append(Function(V))
            guesses[i].assign(guesses[0])
            found_solutions.append(0)
    return [guesses, found_solutions]

def FEMsetup(comm, problem, mu_start):
    mu = Constant(mu_start)
    mesh = problem.mesh(comm)
    FcnSpace = problem.function_space(mesh)
    dm = None
    return (mu, mesh, FcnSpace, dm)

def extractguesses(problem, initialstring, FcnSpace, params, comm):
    if initialstring != None:
        guesses = [Function(FcnSpace)]
        h5 = HDF5File(initialstring, "r", comm=comm)
        h5.read(guesses[0], "/guess")
        del h5
    else:
        guesses = problem.initial_guesses(FcnSpace, params)
    return guesses

def createLog(Max_solutions):
    
    Event = PETSc.Log.Event
    PETSc.Log.begin()
    runenv = Event("run") 
    
    Log = {}
    Log["num_ksp_its"]      = {}
    Log["num_ksp_its_pred"] = {}
    Log["num_ksp_its_defl"] = {}
    Log["mus"]              = []
    Log["solutions_cost"]   = {}
    Log["num_snes_its"]     = {}
    Log["num_snes_its_failed"] = {}
    Log["num_snes_its_defl"]= {}
    Log["num_snes_its_pred"]= {}
    Log["min"]= {}
    Log["run_time"] = {}
    Log["run_time"]["failed"] = [0.0]
    for i in range(Max_solutions):
        Log["solutions_cost"][i]      = []
        Log["num_snes_its"][i]        = []
        Log["num_snes_its_failed"][i] = []
        Log["num_snes_its_defl"][i]   = []
        Log["num_snes_its_pred"][i]   = []
        Log["num_ksp_its"][i]         = []
        Log["num_ksp_its_pred"][i]    = []
        Log["num_ksp_its_defl"][i]    = []
        Log["min"][i] = "No"
        Log["run_time"][i]            = [0.0]
    return (Log, runenv)

def initialise_branches(found_solutions, Max_solutions, max_solutions):
    branches = []
    iter1 = 0
    # Run through active solution branches
    for iter1 in range(Max_solutions):
        if found_solutions[iter1] == 1:
            branches.append(iter1)
    iter1 = 0
    iter2 = 0
    # Also initialise branches that the user has specified exist
    if max_solutions > sum(found_solutions):
        while iter2 < max_solutions - sum(found_solutions):
            if found_solutions[iter1] == 0:
                branches.append(iter1)
                iter2+=1
            iter1 +=1
    return branches

def PredictionCorrectionDeflation(iter_subproblem, problem, FcnSpace, mesh,
                                         u, v, w, lb, ub, params, bcs, dm, deflation,
                                         nvs_cont, nvs_defl, comm,
                                         branches, branch_iter, solutionpath_string,
                                         branch_deflate_start, branch_deflate,
                                         hint_guess, hint, guesses,
                                         mu, oldmu, oldguesses, found_solutions, solutions,
                                         pvd, inertia, inertia_old,
                                         max_solutions, Max_solutions, halfmu, num_halfstep, max_halfstep, hmin, Log,
                                         runenv):

    def outputargs():
        return (branch_iter, branch_deflate, mu, oldmu, oldguesses, Log, found_solutions,
                deflation, halfmu, num_halfstep, max_solutions, guesses, task)

    branch = branches[branch_iter]
    # the bounds that are passed to the solver
    vi = problem.bounds_vi(FcnSpace, mu, params)

    # If solution is already saved, no need to recalculate it
    exists = os.path.isfile("%s/%s.xml.gz" % (solutionpath_string, branch))
    if exists:
        hint_tmp = u.copy(deepcopy = True)
        h5 = HDF5File("%s/%s.xml.gz" % (solutionpath_string, branch), "r", comm=comm)
        h5.read(u, "/guess")
        del h5
        info_green("Solution already found")
        hint_guess[branch][0] = hint_tmp
        hint_guess[branch][1] = float(oldmu)
        task = "ContinuationTask"
    else:
        # If branch is active, continue the branch
        if found_solutions[branch] == 1:
            u.assign(oldguesses[branch])
            task = "PredictorTask"
            info_blue("Task: %s, Branch: %s" %(task, branch))
            # solver parameters that are passed to the solver
            sp = problem.solver_parameters(mu, branch, task, params)
            # Predictor-corrector scheme
            if float(mu) != 0.0:
                (hint_guess[branch], pred_snes_its, pred_ksp_its) = problem.predictor(problem, u, v, w,
                                                                      oldmu, mu, iter_subproblem,
                                                                      params, task, vi, hint[branch])

                Log["num_ksp_its_pred"][branch].append(pred_ksp_its)
                Log["num_snes_its_pred"][branch].append(pred_snes_its)
            task = "ContinuationTask"

        # If branch is inactive, perform deflation
        elif found_solutions[branch] == 0:
            task = "DeflationTask"

            if branch == 0 and branch_deflate == 0: branch_deflate = branches[0]
            u.assign(oldguesses[branch_deflate])

            # elif task == "InfeasibleRhoTask":
            #     u.assign(oldguesses[branch])

    # solver parameters that are passed to the solver
    sp = problem.solver_parameters(mu, branch, task, params)

    # Solve!!!
    if task == "ContinuationTask":
        info_blue("Task: %s, Branch: %s, mu: %s" %(task, branch, float(mu)))
        #FIXME For reason it needs to be reset for each solve when switching branches
        # without this with the same initial guess, the initial residual at iteration 0 would be incorrect
        nvs_cont = problem.solver(u, lb, ub, mu, 0, params, "ContinuationTask")
        runenv.begin()
        (success, snes_its, ksp_its) = defsolve(nvs_cont, deflateop = deflation, vi=vi)
        runenv.end()
    elif task == "DeflationTask":
        info_blue("Task: %s, Branch: %s, Initial guess: branch %s, mu: %s" %(task, branch, branch_deflate, float(mu)))
        #FIXME For reason it needs to be reset for each solve when switching branches
        # without this with the same initial guess, the initial residual at iteration 0 would be incorrect
        nvs_defl = problem.solver(u, lb, ub, mu, 0, params, "DeflationTask")
        runenv.begin()
        (success, snes_its, ksp_its) = defsolve(nvs_defl, deflateop = deflation, vi=vi)
        runenv.end()
    

    total_time = runenv.getPerfInfo()["time"]
    cumulative_time = sum(sum(Log["run_time"][b]) for b in branches)
    cumulative_time += sum(Log["run_time"]["failed"])
    # Hopefully, scheme has converged
    if success:
        Log["run_time"][branch].append(total_time-cumulative_time)
        
        if task == "ContinuationTask":
            # count iterations
            Log["num_snes_its"][branch].append(snes_its)
            Log["num_ksp_its"][branch].append(ksp_its)
        elif task == "DeflationTask":
            Log["num_snes_its_defl"][branch].append(snes_its)
            Log["num_ksp_its_defl"][branch].append(ksp_its)

        # sometimes found solution violates rho volume constraint and we should
        # not continue these solutions.
        rho = split(u)[0]
        #infeasibility_rho = assemble(rho*dx)/assemble(Constant(1.0)*dx(mesh))
        infeasibility_rho = 0 # for some reason above line breaks Firedrake as of 11 Nov 2020

        if infeasibility_rho > problem.volume_constraint(params)+1e-4:
            found_solutions[branch] = 0

            lmbda = split(u)[-1]
            infeasibility_lmbda = assemble(lmbda*dx)/assemble(Constant(1.0)*dx(mesh))
            info_red(r"Found solution violates volume constraint on rho, rho*dx/$|\Omega|$: %s,lmbda %s\nDeflating non-feasible solution and trying again" %(infeasibility_rho, infeasibility_lmbda))
            task = "InfeasibleRhoTask"

        else:
            found_solutions[branch] = 1 # keep track of successful branch continuation

            guesses[branch].assign(u)

            problem.save_pvd(pvd, u, mu)
            problem.save_solution(comm, u, branch, solutionpath_string)

            if float(mu) == 0.0:
                Log["min"][branch] = "Unknown"

                # rho = split(u)[0]
                # lb_inertia = Constant(0)
                # ub_inertia = Constant(1)
                # L = ( problem.lagrangian(u, params)
                #       + 1e10*MorYos(rho - lb_inertia)*dx
                #       + 1e10*MorYos(ub_inertia - rho)*dx
                #     )
                # F = derivative(L, u, v)
                # J = problem.jacobian(F,u,params,v,w)
                # inertia[branch] = problem.compute_stability(mu, params, lb, ub, branch, u, v, w, FcnSpace, bcs, J)
                # if problem.expected_inertia() == None:
                #     Log["min"][branch] = "Unknown"
                # elif inertia[branch][0] == problem.expected_inertia():
                #     Log["min"][branch] = "Yes"

            info_green(r"Found solution in branch %d for mu = %s, time = %.4f seconds" % (branch, float(mu), Log["run_time"][branch][-1]))
            branch_iter +=1


        solutions.append(u.copy(deepcopy=True))
        deflation.deflate(solutions)
        branch_deflate = branch_deflate_start



        if len(solutions) >= max(max_solutions, sum(found_solutions)) and task != 'InfeasibleRhoTask':
            info_green("Not deflating as we have found the maximum number of solutions for given mu")
            package = outputargs()
            return package
    else:
        # Keep track of time spent in failed solves
        Log["run_time"]["failed"].append(total_time-cumulative_time)
        
        if task == "ContinuationTask":
            # keep track of failed iterations too
            Log["num_snes_its_failed"][branch].append(snes_its)
            # if max_halfstep == True and num_halfstep == 0:
            if num_halfstep < max_halfstep:
                # if continuation has failed, half mu stepsize
                halfmu.assign(mu)
                mu.assign(0.5*(float(mu)+float(oldmu)))
                info_red("%s for branch %s has failed, halfing stepsize in mu, considering mu = %s" %(task,branch, float(mu)))
                branch_deflate = 0
                num_halfstep += 1
                branch_iter = Max_solutions # to break the while loop
                task = "HalfstepsizeTask"
                package = outputargs()
                return package
            else:
                # already attempted to half stepsize in mu, time to move on...
                found_solutions[branch] = 0
                branch_iter +=1
                num_halfstep = 0


        elif task == 'DeflationTask':
            # if deflation failed, use a different branch as an initial guess

            branch_deflate +=1
            found_solutions[branch] = 0
            try:
                if branch_deflate == branch: branch_deflate +=1 # should not use one's own branch as an initial guess in deflation
                while found_solutions[branch_deflate] == 0: branch_deflate +=1 # should not use branch with no current solution
            except:
                pass
            if branch_deflate > max_solutions-1: # if deflation has failed from all branches, move on...
                info_red("No solution found for branch %s, moving onto next branch" %branch)
                while found_solutions[branch] == 0:
                    branch_iter += 1
                    if branch_iter==max_solutions:
                        package = outputargs()
                        return package
                branch_deflate = branch_deflate_start
        if sum(found_solutions) == 0:
            package = outputargs()
            return package

    return outputargs()

def UpdateSubproblemParams(oldguesses, guesses, Max_solutions,
                        task, num_halfstep, max_halfstep,
                        hint_guess, hint, hmin,
                        FcnSpace, params, u, problem,
                        oldmu, mu, halfmu, iter_subproblem, Log):

    def outputargs():
        return (oldguesses, hint, oldmu, mu, num_halfstep)

    # This is complicated by the half step procedure. Need to ensure to correct
    # hints and oldmu is passed on if half steps are being used

    # update the oldguesses
    for i in range(Max_solutions):
        oldguesses[i].assign(guesses[i])
    if task == "HalfstepsizeTask":
        pass # if in halfstep mode, do nothing

    # if half step has suceeded, then update hints for predictor task and let the
    # new mu be the previously failed mu.
    elif num_halfstep <= max_halfstep and num_halfstep !=0:
        # If hints are empty, can skip this
        for branch in range(Max_solutions):
            if hint_guess[branch][0] is not None:
                # If hint is empty, then it needs to be initialised
                if hint[branch][0] == None:
                    hint[branch] = [Function(FcnSpace), 0.0]
                hint[branch][0].assign(hint_guess[branch][0])
                hint[branch][1] = deepcopy(hint_guess[branch][1])
        oldmu.assign(mu)
        mu.assign(halfmu)
        num_halfstep = 0
    # If there was no half step happening, then proceed as normal. Update hints
    # and update mu accordingly
    else:
        for branch in range(Max_solutions):
            if hint_guess[branch][0] is not None:
                if hint[branch][0] == None:
                    hint[branch] = [Function(FcnSpace), 0.0]
                hint[branch][0].assign(hint_guess[branch][0])
                hint[branch][1] = deepcopy(hint_guess[branch][1])
        k_mu_old = float(mu)/float(oldmu) if iter_subproblem>1 else "NaN"
        oldmu.assign(mu)
        if float(mu) > 0.0:
            mu.assign(problem.update_mu(u, float(mu), min(Log["num_snes_its"]), iter_subproblem, k_mu_old, params))
    return outputargs()
