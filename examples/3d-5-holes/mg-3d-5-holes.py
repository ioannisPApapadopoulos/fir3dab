# -*- coding: utf-8 -*-
from firedrake import *
from fir3dab import *
from alfi import *
from alfi.transfer import *

"""
This script cycles through all the initial guesses for grid-sequencing the 3D
five-holes solutions. The main script is in mg-3d-5-holes-backend.py
"""
five_holes = __import__("mg-3d-5-holes-backend")


def gridsequence():
    problem = five_holes.BorrvallProblem()
    width = 1.5 # aspect ratio
    nref = 1
    gamma_al = 1e5
    dgtransfer = DGInjection()

    for branch in range(14):

        class FiveHoles(object):

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

        newproblem = FiveHoles()
        saving_folder = "mg-output/mg-branch-%s-"%(branch)
        params = [0.2, 2.5e4, 0.1] #(gamma, alphabar, q)
        deflatedbarrier(newproblem, params, mu_start= 1e-6, mu_end = 1e-5, max_halfstep = 0, saving_folder = saving_folder)

gridsequence()
