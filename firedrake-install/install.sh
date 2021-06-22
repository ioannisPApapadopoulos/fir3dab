#!/bin/bash
SCRIPTPATH=$(cd `dirname "${BASH_SOURCE[0]}"` && pwd)
VENVNAME=firedrake

if [ ! -z "$FIREDRAKE_DIR" ]
then
    echo "FIREDRAKE_DIR is already set to $FIREDRAKE_DIR"
    return
fi

export FIREDRAKE_DIR=$SCRIPTPATH/$VENVNAME
export FD_DIR=$SCRIPTPATH/$VENVNAME/src/firedrake
export PETSC_DIR=$FIREDRAKE_DIR/src/petsc
export SLEPC_DIR=$FIREDRAKE_DIR/src/slepc
export PARA_DIR=$FIREDRAKE_DIR/../../ParaView-5.8.0-MPI-Linux-Python3.7-64bit/bin
export PETSC_ARCH=linux-gnu-c-opt
export OPENBLAS_NUM_THREADS=1

if [ -d "/scratch" ]
then
    export PYOP2_CACHE_DIR=/scratch/$USER/.cache/pyop2/$VENVNAME
    export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/scratch/$USER/.cache/firedrake/$VENVNAME
fi

#set -e

if [ "${BASH_SOURCE[0]}" != "${0}" ]
then
    echo "Setting paths for $FIREDRAKE_DIR"
    source $FIREDRAKE_DIR/bin/activate
    export FIREDRAKE_DIR=$FIREDRAKE_DIR
    export LD_LIBRARY_PATH=$FIREDRAKE_DIR/lib/python3.6/site-packages/vtk:$LD_LIBRARY_PATH
    export XDG_CACHE_HOME=$VIRTUAL_ENV/.cache
else
    echo "Installing into $FIREDRAKE_DIR"

    if [ ! -f $FIREDRAKE_DIR/src/petsc/$PETSC_ARCH/lib/petsc/conf/petscvariables ]
    then
        mkdir -p $FIREDRAKE_DIR/src
        cd $FIREDRAKE_DIR/src
        git clone https://github.com/firedrakeproject/petsc.git || true
        cd $FIREDRAKE_DIR/src/petsc
        #git clean -ffdx .
        #./configure     --CC_LINKER_FLAGS=[-L${MKL_DIR}/lib/intel64,-lmkl_intel_lp64,-lmkl_sequential,-lmkl_core,-lpthread,-lm,-ldl] \
        #                --with-blas-lapack-dir=$MKL_DIR \
        #                --with-mkl_pardiso-dir=$MKL_DIR \
                        #--with-mpi-dir=/usr/lib/x86_64-linux-gnu/openmpi \
        ./configure     --with-blas-lib=-lblas \
                        --with-lapack-lib=-llapack \
                        --CFLAGS=$CFLAGS \
                        --CXXFLAGS=$CFLAGS \
                        --FFLAGS=$CFLAGS \
                        --with-cc=mpicc.mpich --with-cxx=mpicxx.mpich --with-fc=mpif90.mpich \
                        --with-mpiexec=mpiexec.mpich \
                        --download-spai=yes \
                        --download-suitesparse=1 \
                        --with-cxx-dialect=C++11 \
                        --with-scalapack=1 \
                        --download-scalapack=1 \
                        --with-blacs=1 \
                        --download-blacs=1 \
                        --with-c-support \
                        --with-cxx-dialect=C++11 \
                        --with-debugging=0 \
                        --with-etags=1 \
                        --with-fortran-interfaces=0 \
                        --with-fortran-bindings=0 \
                        --download-hypre=1 \
                        --with-hypre=1 \
                        --download-superlu=1 \
                        --with-superlu=1 \
                        --with-ml=1 \
                        --download-ml=1 \
                        --with-eigen=1 \
                        --download-eigen=1 \
                        --with-mumps=1 \
                        --download-mumps=1 \
                        --with-hdf5=1 \
                        --download-hdf5=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.bz2 \
                        --with-metis=1 \
                        --download-metis=1 \
                        --with-parmetis=1 \
                        --download-parmetis=1 \
                        --with-chaco=1 \
                        --download-chaco=1 \
                        --with-netcdf=1 \
                        --download-netcdf=1 \
                        --with-pnetcdf=1 \
                        --download-pnetcdf=1 \
                        --with-exodusii=1 \
                        --download-exodusii=1 \
                        --with-ptscotch=1 \
                        --download-ptscotch=1 \
                        --with-shared-libraries \
                        --with-spai=1 \
                        --with-suitesparse=1 \
                        --download-suitesparse=1 \
                        --with-threadcomm=0 \
                        --download-triangle=1 \
                        --with-vtk=1 \
                        --download-vtk=1 \
                        --download-zlib=1
        make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
        make alletags
    fi

    if [ ! -d $FIREDRAKE_DIR/src/slepc/$PETSC_ARCH ]
    then
        cd $FIREDRAKE_DIR/src
        git clone https://github.com/firedrakeproject/slepc.git || true
        cd $FIREDRAKE_DIR/src/slepc
        ./configure
        make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH
        cd ..
    fi

    cd $FIREDRAKE_DIR/src
    rm -rf fiat/  FInAT/ firedrake/ PyOP2/ tsfc/ ufl/ COFFEE/ h5py/ petsc4py/

    cd $FIREDRAKE_DIR/..
    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    unset PYTHONPATH
    sed -i 's@quit("Can@#quit("Can@' firedrake-install
    sed -i 's@os.mkdir("src")@#os.mkdir("src")@' firedrake-install
    rm -f $VENVNAME/bin/{mpicc,mpicxx,mpif90,mpiexec}
    python3 firedrake-install --honour-petsc-dir --slepc --venv-name $VENVNAME --mpicc mpicc.mpich --mpicxx mpicxx.mpich --mpif90 mpif90.mpich --mpiexec mpiexec.mpich

    source $FIREDRAKE_DIR/bin/activate
    pip install ipdb
    pip install vtk
    pip install meshio
    pip install sip
    pip install lxml
    pip install snakeviz
    pip install PyQt5==5.10.1
fi

set +e
