# README #

### Synopsis ###

Firedrake 3D implementation of the Deflated Barrier method (fir-3d-ab or fir3dab)

This deflated-barrier library implements the deflated barrier method of Papadopoulos, Farrell and Surowiec in Firedrake. The original implementation based on FEniCS can be found at https://bitbucket.org/papadopoulos/deflatedbarrier/.   

The objective is to compute multiple minima of topology optimization problems which are non-convex, PDE & box-constrained optimization problems. This particular implementation focuses on developing preconditioners for the deflated barrier method linear systems that arise. 

### Dependencies and installation ###


The code is written in Python using Firedrake: a finite element solver platform. Firedrake is well documented here: firedrakeproject.org. 

At this time this library depends on changes to Firedrake and Firedrake's branch of PETSc that have not yet been merged to master. These changes are mainly related to the current implementation of vinewtonrsls (Benson and Munson's reduced-space active-set strategy) in PETSc which is not compatable with matrix-free operations (and often fails even if using aij matrices). These reframe vinewtonrsls as zeroing rows and columns associated with the active set. On the Firedrake side, these are interpreted as DirichletBCs. These changes allows us to compose complicated solvers (including geometric multigrid) with matrices that have an active set. 

In order to install this library and the version of Firedrake required for this library, use the following commands:

	git clone https://papadopoulos@bitbucket.org/papadopoulos/fir3dab.git
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

To activate the Firedrake venv use:

    source firedrake/install.sh

To install the fir3dab library:
    
    cd fir3dab/
    pip3 install .


### Examples ###

Checkout out examples/double-pipe-tables. To generate the convergence plots and tables found in "Numerical Analysis of a discontinuous Galerkin method for the Borrvall-Petersson topology optimization problem" - I.P.A. Papadopoulos, then run the following command in the parent directory:


    make double-pipe-tables

### Contributors ###

Ioannis P. A. Papadopoulos (ioannis.papadopoulos@maths.ox.ac.uk)

Patrick E. Farrell (patrick.farrell@maths.ox.ac.uk)


### License ###

GNU LGPL, version 3.
