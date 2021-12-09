# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:18:59 2021

@author: luboy
"""
import numpy as np
import copy as cp
# Navier Stokes Solvers
#%% Pysical setting
# physical domain
Lx = 30
Ly = 30
nx = 100 
ny = 100 
x0 = Lx/2
y0 = Ly/2

# physical parameter
gamma = 1
sigma = 2.5

# Index extents (with out boundary domain)
imin = 1
imax = imin + nx - 1

jmin = 1
jmax = jmin + ny - 1

# Create mesh
x = np.zeros([1,nx+2])
x[0,imin:imax+1+1] = np.linspace(0,Lx,nx+1)
y = np.zeros([1,ny+2])
x[0,jmin:jmax+1+1] = np.linspace(0,Ly,ny+1)
xm = np.zeros([1,nx+1])
xm[0,imin:imax] = 0.5*(x[0,imin:imax]+x[0,imin+1:imax+1])
ym = np.zeros([1,ny+1])
ym[0,jmin:jmax] = 0.5*(y[0,jmin:jmax]+y[0,jmin+1:jmax+1])

# Create mesh sizes
dx = x[0,imin+1] - x[0,imin]
dy = y[0,jmin+1] - y[0,jmin]

dxi=1/dx 
dyi=1/dy 

#%% initialization
# u velocity
fu = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*(-(x-x0)/sigma**2)
# v velocity
fv = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*(-(y-y0)/sigma**2)





#%% momentum discretization

# u direction
def u_star(u,v,dt,dxi,dyi,nu):
    u_s = cp.deepcopy(u)
    for i in range(imin,imax+1):
        for j in range(jmin, jmax+1):
            v_cur = (v[i-1,j] + v[i+1,j] + v[i,j-1] + v[i,j+1])/4
            lap_op = (u[i+1,j] - 2*u[i,j] + u[i-1,j])*dxi**2 + (u[i,j+1]-2*u[i,j]+u[i,j-1])**dyi**2
            u_s[i,j] = u[i,j] + dt*(nu*lap_op-u[i,j]*(u[i+1,j]-u[i-1,j])*0.5*dxi
                                    -v_cur*(u[i,j+1]-u[i,j-1])*0.5*dyi)
    return u_s       

# v direction
def v_star(u,v,dt,dxi,dyi,nu):
    v_s = cp.deepcopy(v)
    for i in range(imin,imax+1):
        for j in range(jmin,jmax+1):
            u_cur = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])/4
            lap_op = (v[i+1,j] - 2*v[i,j] + v[i-1,j])*dxi**2 + (v[i,j+1]-2*v[i,j]+v[i,j-1])**dyi**2
            v_s[i,j] = v[i,j] + dt*(nu*lap_op-v[i,j]*(v[i+1,j]-v[i-1,j])*0.5*dxi
                                    -u_cur*(v[i,j+1]-v[i,j-1])*0.5*dyi)
    return v_s

#%% pressure solver

def poisson_solver():
    pass
