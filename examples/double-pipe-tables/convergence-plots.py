# -*- coding: utf-8 -*-

from firedrake import *
from firedrake.petsc import PETSc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
import numpy as np
import os
comm = COMM_WORLD

def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)


def broken_h1_norm(u_f, u_c, zc):    
    n_fun = FacetNormal(zc.ufl_domain())
    h_fun = CellDiameter(zc.ufl_domain())
    uc = zc.split()[1]

    Ff = inner(u_f - u_c, u_f - u_c)*dx + inner(grad(u_f - u_c), grad(u_f - u_c))*dx
    Fc = 1./avg(h_fun) * inner(2*avg(outer(uc,n_fun)), 2*avg(outer(uc,n_fun))) * dS

    F = assemble(Ff) + assemble(Fc)
    return sqrt(F)

N = 20
delta = 1.5
mesh = RectangleMesh(N,N,delta,1.0,comm=comm)
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = RectangleMesh(N, N, delta, 1.0, distribution_parameters = distribution_parameters, comm=COMM_WORLD)

Ve = FiniteElement("BDM", mesh.ufl_cell(), 1, variant = "integral") # velocity
Pe = FiniteElement("DG", mesh.ufl_cell(), 0) # pressure
Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # material distribution
Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals
Ze = MixedElement([Ce, Ve, Pe, Re])
hierarchy = MeshHierarchy(mesh, 6)

mesh0 = hierarchy[-1]
mesh1 = mesh0

Ze = MixedElement([Ce, Ve, Pe, Re])
Z_fine_0  = FunctionSpace(mesh0, Ze)
Z_fine_1  = FunctionSpace(mesh1, Ze)
z_fine_0= Function(Z_fine_0)
z_fine_1= Function(Z_fine_1)
z_prolong = Function(Z_fine_0)
dofs_fine = Z_fine_0.dim()


parent_folder = "output/"
h5 = HDF5File(parent_folder + "BDM-N-20-nref-6-output/mu-0.000000000000e+00-dofs-16389121-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/0.xml.gz", "r", comm=comm)
h5.read(z_fine_0, "/guess")
del h5
h5 = HDF5File(parent_folder + "BDM-N-20-nref-6-output/mu-0.000000000000e+00-dofs-16389121-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/1.xml.gz", "r", comm=comm)
h5.read(z_fine_1, "/guess")
del h5
(rho_0, u_0, p_0, _) = z_fine_0.split()
(rho_1, u_1, p_1, _) = z_fine_1.split()
p_0.vector().set_local(p_0.vector().get_local()-assemble(p_0*dx)/1.5)
p_1.vector().set_local(p_1.vector().get_local()-assemble(p_1*dx)/1.5)


list_rho = [[],[]]
list_u = [[],[]]
list_u_l2 = [[],[]]
list_p = [[],[]]
h = []


for levels in [1,2,3,4,5]:
    mesh = hierarchy[levels]
    Z  = FunctionSpace(mesh, Ze)
    z = Function(Z)

    dofs = Z.dim()
    hmin = sqrt((1.5/(N*2**levels))**2 + (1.0/(N*2**levels))**2)
    h.append(hmin)

    for branch in range(0,2):
        
        h5 = HDF5File(parent_folder + "BDM-N-%s-nref-%s-output/mu-0.000000000000e+00-dofs-%s-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/%s.xml.gz"%(N, levels, dofs, branch), "r", comm=comm)
        h5.read(z, "/guess")
        del h5
        
        prolong(z, z_prolong)

        (rho, u, p, _) = z_prolong.split()
        p.vector().set_local(p.vector().get_local()-assemble(p*dx)/1.5)
        if branch == 0:
            rho_ = rho_0; u_ = u_0; p_ = p_0
        else:
            rho_ = rho_1; u_ = u_1; p_ = p_1
        
        # Compute norm
        list_rho[branch].append(errornorm(rho_, rho, norm_type='L2'))
        list_u[branch].append(broken_h1_norm(u_,u, z))
        list_u_l2[branch].append(errornorm(u_,u, norm_type='L2'))
        list_p[branch].append(errornorm(p_,p, norm_type='L2'))
        print("Finished level = %s"%levels)

rho_0 = list_rho[0]
rho_1 = list_rho[1]
u_0   = list_u[0]
u_1   = list_u[1]
p_0   = list_p[0]
p_1   = list_p[1]
u_0_l2 = list_u_l2[0]
u_1_l2 = list_u_l2[1]

hnorm = np.asarray(h)/h[0]
print("h = %s" %h)
print("rho_0 = %s" %rho_0)
print("rho_1 = %s" %rho_1)
print("u_0 = %s" %u_0)
print("u_1 = %s" %u_1)
print("u_0_l2 = %s" %u_0_l2)
print("u_1_l2 = %s" %u_1_l2)
print("p_0 = %s" %p_0)
print("p_1 = %s" %p_1)

try:
    os.makedirs('figures')
except:
    pass

#h1 = [0.180, 0.090, 0.045, 0.023, 0.011]

plt.loglog(h,u_0, marker = 'x', label = r"Straight channels")
plt.loglog(h,u_1, marker = 'o', label = r"Double-ended wrench")
#plt.loglog(h,hnorm*u_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
#plt.loglog(h,hnorm**1.5*u_0[0], color = 'r', linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*u_0[0], color = 'm', linestyle = '--',label = r"$\mathcal{O}(h^2)$")
plt.legend(loc = 0)
plt.title(r"$H^1(\mathcal{T}_h)$-norm error of the velocity", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|u - u_h\|_{H^1(\mathcal{T}_h)}$", fontsize = 20)
#plt.minorticks_off()
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_u.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h,u_0_l2, marker = 'x', label = r"Straight channels")
plt.loglog(h,u_1_l2, marker = 'o', label = r"Double-ended wrench")
#plt.loglog(h,hnorm*u_0_l2[0],linestyle = '--', label = r"$\mathcal{O}(h)$")
#plt.loglog(h,hnorm**1.5*u_0_l2[0], linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*u_0_l2[0], color = 'm', linestyle = '--',label = r"$\mathcal{O}(h^2)$")
#plt.loglog(h,hnorm**3*u_0_l2[0], color = 'y', linestyle = '--',label = r"$\mathcal{O}(h^3)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the velocity", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|u - u_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks(fontsize = 15)
#plt.minorticks_off()
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_u_l2.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h,rho_0, marker = 'x', label = r"Straight channels")
plt.loglog(h,rho_1, marker = 'o', label = r"Double-ended wrench")
#plt.loglog(h,hnorm*rho_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
#plt.loglog(h,hnorm**1.5*rho_0[0], color = 'r', linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*rho_0[0], color = 'm', linestyle = '--', label = r"$\mathcal{O}(h^2)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the material distribution", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|\rho - \rho_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks(fontsize = 15)
#plt.minorticks_off()
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_rho.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h,p_0, marker = 'x', label = r"Straight channels")
plt.loglog(h,p_1, marker = 'o', label = r"Double-ended wrench")
#plt.loglog(h,hnorm*p_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
#plt.loglog(h,hnorm**1.5*p_0[0], color = 'r', linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*p_0[0], color = 'm', linestyle = '--', label = r"$\mathcal{O}(h^2)$")
#plt.loglog(h,hnorm**3*p_0[0], color = 'y', linestyle = '--', label = r"$\mathcal{O}(h^3)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the pressure", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|p - p_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks(fontsize = 15)
#plt.minorticks_off()
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_p.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()
