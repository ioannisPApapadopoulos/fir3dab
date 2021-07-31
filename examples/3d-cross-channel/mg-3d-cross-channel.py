# -*- coding: utf-8 -*-
from firedrake import *
from fir3dab import *
from alfi import *
from alfi.transfer import *

"""
This script cycles through all the initial guesses for grid-sequencing the 3D
cross-channel solutions. The main script is in mg-3d-cross-channel-backend.py
"""
cross_channel = __import__("mg-3d-cross-channel-backend")


def gridsequence():
    problem = cross_channel.BorrvallProblem()
    width = 1. # aspect ratio
    N = 20     # mesh resolution
    nref = 1   # number of refinements of the base mesh
    gamma_al = 1e5 # augmented Lagrangian parameter
    dgtransfer = DGInjection()

    for branch in range(3):

        class CrossChannel(object):

            def initial_guesses(self, Z, params):
                """
                 Use the saved solutions from save_h5 as initial guesses
                """
                comm = Z.comm
                z = Function(Z)
                h5 = HDF5File("initial-guess/%s.xml.gz"%branch, "r", comm=comm)
                h5.read(z, "/guess")
                del h5
                return [z]

            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = CrossChannel()
        saving_folder = "mg-output/mg-branch-%s-"%(branch)
        params = [0.1, 2.5e4, 0.1] #(gamma, alphabar, q)
        deflatedbarrier(newproblem, params, mu_start= 1e-6, mu_end = 1e-5, max_halfstep = 0, saving_folder = saving_folder)

gridsequence()
