# README #

### Synopsis ###

Firedrake implementation of the deflated barrier method for 3D topology optimization problems (fir-3d-ab or fir3dab)

This deflated-barrier library implements the deflated barrier method of Papadopoulos, Farrell and Surowiec in Firedrake. The original implementation based on FEniCS can be found at https://bitbucket.org/papadopoulos/deflatedbarrier/.   

The objective is to compute multiple minima of topology optimization problems which are non-convex, PDE & box-constrained optimization problems. This particular implementation focuses on developing preconditioners for the deflated barrier method linear systems that arise.

### Dependencies and installation ###


The code is written in Python using Firedrake: a finite element solver platform. Firedrake is well documented here: firedrakeproject.org.

At this time, this library depends on changes to Firedrake and Firedrake's branch of PETSc that have not yet been merged to master. These changes are mainly related to the current implementation of vinewtonrsls (Benson and Munson's reduced-space active-set strategy) in PETSc which is not compatable with matrix-free operations (and often fails even if using aij matrices). These changes reframe vinewtonrsls as zeroing rows and columns associated with the active set. On the Firedrake side, these are interpreted as DirichletBCs. These changes allows us to compose complicated solvers (including geometric multigrid) with matrices that have an active set.

This package also requires the alfi package for the correct prolongation and injection of functions in the multigrid scheme (https://github.com/florianwechsung/alfi).

In order to install this library and the version of Firedrake required for this library, use the following commands:

    git clone https://github.com/ioannisPApapadopoulos/fir3dab.git
    mkdir firedrake
    cp fir3dab/firedrake-install/install.sh firedrake/
    cd firedrake
    ./install.sh install
    cd firedrake/src/firedrake
    git apply ../../../../fir3dab/firedrake-install/firedrake.diff
    cd ../petsc
    git apply ../../../../fir3dab/firedrake-install/petsc.diff
    make
    cd ../../../../

Then make sure to activate the Firedrake venv and run a firedrake-clean!

    source firedrake/install.sh
    firedrake-clean

A pip install of the fir3dab library:

    cd fir3dab/
    pip3 install .
    cd ../

Finally, an installation of the alfi library and a switch to the correct branch:

    git clone https://github.com/florianwechsung/alfi.git
    cd alfi/
    git checkout fw/hdiv
    pip3 install .
    cd ../


### Examples ###

Checkout out examples/double-pipe-tables. To generate the convergence plots and tables found in "Numerical Analysis of a discontinuous Galerkin method for the Borrvall-Petersson topology optimization problem" - I.P.A. Papadopoulos, then run the following command in the parent directory of fir3dab:


    make double-pipe-tables
    
In examples/3d-5-holes.py we compute and grid-sequence 11 solutions to a 3D quadruple pipe problem with 5 cuboid holes in the domain. This examples utilizes preconditioning techniques including the robust MG cycle that with star patch relaxation and a representation of the active set on coarser levels. The command:

    make 3d-5-holes-coarse

finds the solution on a (coarse) mesh. The command

    make 3d-5-holes-prolong-solutions

then prolongs the solutions to a uniform refinement of the coarse mesh. The command

    make 3d-5-holes-fine	
    
then grid-sequences the solutions utilizing the robust MG cycle. 

s
### Contributors ###

Ioannis P. A. Papadopoulos (ioannis.papadopoulos@maths.ox.ac.uk)

Patrick E. Farrell (patrick.farrell@maths.ox.ac.uk)


### License ###

GNU LGPL, version 3.
