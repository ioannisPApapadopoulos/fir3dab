from firedrake import *
import sys
import os
from .mlogging import *
import numpy as np

def soln(array):
    if len(array) == 1:
        return "solution"
    else:
        return "solutions"

def plus(x):
    return conditional(gt(x, 0), x, 0)

def MorYos(x):
    return conditional(ge(x, 0), 0, x**2)

def create_output_folder(saving_folder):
    if sys.version_info[0] < 3:
        try:
            os.removedirs(saving_folder + "output")
        except:
            pass
        try:
            os.makedirs(saving_folder + "output")
        except:
            pass
    else:
        os.makedirs(saving_folder + "output", exist_ok=True)
    return None

def initialise_guess(guesses,Max_solutions,found_solutions, V, params):
    number_initial_branches = len(guesses)
    if (Max_solutions - number_initial_branches) > 0:
        for i in range(number_initial_branches,Max_solutions):
            guesses.append(Function(V))
            guesses[i].assign(guesses[0])
            found_solutions.append(0)
    return [guesses, found_solutions]

def initialise_branches(found_solutions, Max_solutions, max_solutions):
    branches = []
    iter = 0
    # Run through active solution branches
    for iter in range(Max_solutions):
        if found_solutions[iter] == 1:
            branches.append(iter)
    iter = 0
    iter2 = 0
    # Also initialise branches that the user has specified exist
    if max_solutions > sum(found_solutions):
        while iter2 < max_solutions - sum(found_solutions):
            if found_solutions[iter] == 0:
                branches.append(iter)
                iter2+=1
            iter +=1
    return branches

def inertia_switch(inertia, inertia_old, max_solutions, Max_solutions, branches, branch):
    try: # interia_old[branch] may not be defined yet
        if inertia[branch] != inertia_old[branch]:
            info_green("Inertia change detected")
            if max_solutions < Max_solutions: # do not want to exceed known max solution count
                max_solutions +=1
                for iter in range(Max_solutions):
                    if found_solutions[iter] == 0:
                        branches.append(iter)
                        break

    except: pass
    return [max_solutions, branches]

def density_filter(u, rho_filter, test_filter):
    rho = split(u)[0]
    F = 1e-5*inner(grad(rho_filter),grad(test_filter))*dx + inner(rho_filter,test_filter)*dx - inner(rho, test_filter)*dx
    solve(F == 0, rho_filter)
    assign(u.sub(0), rho_filter)
    return u

def report_profile(Log, Max_solutions):

    info_blue("-" * 80)
    info_blue("| Profiling statistics collected" + " "*35 + "|")
    info_blue("-" * 80)

    info_blue(" " + "*"*21)

    cont  = sum(sum(Log["num_snes_its"][b]) for b in range(Max_solutions))
    cont  = int(cont)
    defl  = sum(sum(Log["num_snes_its_defl"][b]) for b in range(Max_solutions))
    defl  = int(defl)
    pred  = sum(sum(Log["num_snes_its_pred"][b]) for b in range(Max_solutions))
    pred  = int(pred)
    total = cont+defl+pred
    ksp_cont = sum(sum(Log["num_ksp_its"][b]) for b in range(Max_solutions))
    ksp_cont = int(ksp_cont)
    ksp_defl = sum(sum(Log["num_ksp_its_defl"][b]) for b in range(Max_solutions))
    ksp_defl = int(ksp_defl)
    ksp_pred = sum(sum(Log["num_ksp_its_pred"][b]) for b in range(Max_solutions))
    ksp_pred = int(ksp_pred)
    ksp_total = ksp_cont + ksp_defl + ksp_pred
    if total:
        ksp_total_avg = float(ksp_total)/float(total)
    else: ksp_total_avg = 0
    if cont:
        ksp_cont_avg = float(ksp_cont)/float(cont)
    else: ksp_cont_avg = 0
    if defl:
        ksp_defl_avg = float(ksp_defl)/float(defl)
    else: ksp_defl_avg = 0
    if pred:
        ksp_pred_avg = float(ksp_pred)/float(pred)
    else: ksp_pred_avg = 0

    time_taken = sum(sum(Log["run_time"][b]) for b in range(Max_solutions))

    info_green(" * Totals over all branches *")
    info_blue(" " + "*"*21)
    info_green("     Time taken (s):                    %s" %time_taken)
    info_green("     Total SNES iterations:             %s" %total)
    info_green("     Continuation SNES iterations:      %s" %cont)
    info_green("     Deflation SNES iterations:         %s" %defl)
    info_green("     Prediction SNES iterations:        %s" %pred)
    info_green("                                        Total (Avg per SNES iteration)")
    info_green("     Total KSP iterations:              %s (%s)" %(ksp_total, ksp_total_avg))
    info_green("     Continuation KSP iterations:       %s (%s)" %(ksp_cont, ksp_cont_avg))
    info_green("     Deflation KSP iterations:          %s (%s)" %(ksp_defl, ksp_defl_avg))
    info_green("     Prediction KSP iterations:         %s (%s)" %(ksp_pred, ksp_pred_avg))
    info_green(" ")

    for branch in range(Max_solutions):
        cont  = np.asarray(Log["num_snes_its"][branch])
        cont  = int(np.sum(cont))
        defl  = np.asarray(Log["num_snes_its_defl"][branch])
        defl  = int(np.sum(defl))
        pred  = np.asarray(Log["num_snes_its_pred"][branch])
        pred  = int(np.sum(pred))
        ksp_cont = np.asarray(Log["num_ksp_its"][branch])
        ksp_cont = int(np.sum(ksp_cont))
        ksp_defl = np.asarray(Log["num_ksp_its_defl"][branch])
        ksp_defl = int(np.sum(ksp_defl))
        ksp_pred = np.asarray(Log["num_ksp_its_pred"][branch])
        ksp_pred = int(np.sum(ksp_pred))
        if cont:
            ksp_cont_avg = float(ksp_cont)/float(cont)
        else: ksp_cont_avg = 0
        if defl:
            ksp_defl_avg = float(ksp_defl)/float(defl)
        else: ksp_defl_avg = 0
        if pred:
            ksp_pred_avg = float(ksp_pred)/float(pred)
        else: ksp_pred_avg = 0


        cost = Log["solutions_cost"][branch][-1]
        minimum  = Log["min"][branch]
        time_taken = np.sum(np.asarray(Log["run_time"][branch]))

        info_green(" * Branch %s *"%branch)
        info_blue(" " + "*"*21)
        info_green("     Cost:                              %s" %cost)
        info_green("     Local minimum:                     %s" %minimum)
        info_green("     Time taken (s):                    %s" %time_taken)
        info_green("     Continuation SNES iterations:      %s" %cont)
        info_green("     Deflation SNES iterations:         %s" %defl)
        info_green("     Prediction SNES iterations:        %s" %pred)
        info_green("                                        Total (Avg per SNES iteration)")
        info_green("     Continuation KSP iterations:       %s (%s)" %(ksp_cont, ksp_cont_avg))
        info_green("     Deflation KSP iterations:          %s (%s)" %(ksp_defl, ksp_defl_avg))
        info_green("     Prediction KSP iterations:         %s (%s)" %(ksp_pred, ksp_pred_avg))
        info_green(" ")
