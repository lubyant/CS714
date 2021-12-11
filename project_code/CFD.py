# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:18:59 2021

@author: luboy
"""
import numpy as np
import copy as cp
from numpy import linalg as LA
import matplotlib.pyplot as plt
# Navier Stokes Solvers
#%% Pysical setting
# physical domain
Lx = 30
Ly = 30
nx = 32
ny = 32
x0 = Lx/2
y0 = Ly/2

# physical parameter
gamma = 1
sigma = 2.5
nu = 0.001
rho = 1

# Index extents (without boundary domain)
imin = 1
imax = imin + nx - 1

jmin = 1
jmax = jmin + ny - 1

# Create mesh
x = np.zeros([1,nx+2])
x[0,imin:imax+1+1] = np.linspace(0,Lx,nx+1)
y = np.zeros([1,ny+2])
y[0,jmin:jmax+1+1] = np.linspace(0,Ly,ny+1)
xm = np.zeros([1,nx+1])
xm[0,imin:imax] = 0.5*(x[0,imin:imax]+x[0,imin+1:imax+1])
ym = np.zeros([1,ny+1])
ym[0,jmin:jmax] = 0.5*(y[0,jmin:jmax]+y[0,jmin+1:jmax+1])

# Create mesh sizes
dx = x[0,imin+1] - x[0,imin]
dy = y[0,jmin+1] - y[0,jmin]

dxi=1/dx 
dyi=1/dy 




#%% momentum discretization

# u direction
def u_star(u,v,dt,dxi,dyi,nu):
    u_s = cp.deepcopy(u)
    for j in range(jmin,jmax+1):
        for i in range(imin+1, jmax+1):
            v_cur = (v[i-1,j] + v[i-1,j+1] + v[i,j] + v[i,j+1])/4
            lap_op = (u[i+1,j] - 2*u[i,j] + u[i-1,j])*dxi**2 + (u[i,j+1]-2*u[i,j]+u[i,j-1])*dyi**2
            u_s[i,j] = u[i,j] + dt*(nu*lap_op-u[i,j]*(u[i+1,j]-u[i-1,j])*0.5*dxi
                                    -v_cur*(u[i,j+1]-u[i,j-1])*0.5*dyi)
    return u_s       

# v direction
def v_star(u,v,dt,dxi,dyi,nu):
    v_s = cp.deepcopy(v)
    for j in range(jmin+1,jmax+1):
        for i in range(imin,imax+1):
            u_cur = (u[i,j-1] + u[i,j] + u[i+1,j-1] + u[i+1,j])/4
            lap_op = (v[i+1,j] - 2*v[i,j] + v[i-1,j])*dxi**2 + (v[i,j+1]-2*v[i,j]+v[i,j-1])*dyi**2
            v_s[i,j] = v[i,j] + dt*(nu*lap_op-u_cur*(v[i+1,j]-v[i-1,j])*0.5*dxi
                                    -v[i,j]*(v[i,j+1]-v[i,j-1])*0.5*dyi)
    return v_s

#%% pressure poisson solver
# d2p/dx2 + d2p/dy2 = - rho/dt du_s/dx + dv_s/dy
# def pressure_poisson_matrix():
#     L = np.zeros([nx*ny,nx*ny])
#     for j in range(ny):
#         for i in range(nx):
#             L[i-1+(j-1)*nx, i + (j)*nx] == 2*dxi**2+2*dyi**2
#             for ii in range(i-1,i+2,2):
#                 if ii>-1 and ii<=nx:
#                     L[i+(j)*nx,ii+(j)*nx] = -dxi**2
#                 else:
#                     L[i+(j)*nx,i+(j)*nx] -= dxi**2
            
#             for jj in range(j-1,j+2,2):
#                 if jj>-1 and jj<= ny:
#                     L[i+(j)*nx,i+(jj)*nx] = -dyi**2
#                 else:
#                     L[i+(j)*nx,i+(j)*nx] -= dyi**2
#     L[0,:] = 0
#     L[0,0] = 1
#     return L  
# def pressure_poisson_matrix():
#     diag1 = -2*np.ones([1,nx]).flatten()
#     diag2 = np.ones([1,nx-1]).flatten()
#     A = np.diag(diag1) + np.diag(diag2,k=-1) + np.diag(diag2,k=1)
#     A[0,0] = -1
#     A[-1,-1] = -1
    
    
#     M = np.kron(A,A)/dx**2
# L =         pressure_poisson_matrix()  
def pressure_poisson(P, u_s, v_s, rho, dt, dx, dy, imin, imax, jmin, jmax):
    P0 = cp.deepcopy(P)
    error = np.inf
    while error > 1/(P.shape[0]*P.shape[1]):
        for i in range(imin,imax+1):
            for j in range(imin,imax+1):
                rhs = rho/dt*((u_s[i+1,j] - u_s[i,j])/dx + (v_s[i,j+1] - v_s[i,j])/dy)
                if i == imin and j == jmin: # (0,0)
                    P0[i,j] = 1/(dx**2+dy**2)*(dy**2*P[i+1,j]+dx**2*P[i,j+1]-dx**2*dy**2*rhs)
                elif i == imax-1 and j == jmin: #right bc
                    P0[i,j] = 1/(dx**2+dy**2)*(dy**2*P[i+1,j]+dx**2*P[i,j+1]-dx**2*dy**2*rhs)
                elif i == imin and j == jmax-1: # bottom bc
                    P0[i,j] = 1/(dx**2+dy**2)*(dy**2*P[i+1,j]+dx**2*P[i,j-1]-dx**2*dy**2*rhs)
                elif j == jmax-1 and j == jmax-1: # top bc
                    P0[i,j] = 1/(dx**2+dy**2)*(dy**2*P[i-1,j]+dx**2*P[i,j-1]-dx**2*dy**2*rhs)
                elif i == imin:
                    P0[i,j] = 1/(2*dx**2+dy**2)*(dy**2*P[i+1,j]+dx**2*P[i,j-1]+dx**2*P[i,j+1]-dx**2*dy**2*rhs)
                elif i == imax-1:
                    P0[i,j] = 1/(2*dx**2+dy**2)*(dy**2*P[i-1,j]+dx**2*P[i,j-1]+dx**2*P[i,j+1]-dx**2*dy**2*rhs)
                elif j == jmin:
                    P0[i,j] = 1/(dx**2+2*dy**2)*(dy**2*P[i+1,j]+dy**2*P[i-1,j]+dx**2*P[i,j+1]-dx**2*dy**2*rhs)
                elif j == jmax-1:
                    P0[i,j] = 1/(dx**2+2*dy**2)*(dy**2*P[i+1,j]+dy**2*P[i-1,j]+dx**2*P[i,j-1]-dx**2*dy**2*rhs)
                else:
                    
                    P0[i,j] = 1/(2*dx**2+2*dy**2)*(dy**2*P[i+1,j]+dy**2*P[i-1,j]+dx**2*P[i,j-1]+dx**2*P[i,j+1]-dx**2*dy**2*rhs)
        error = LA.norm(P0.flatten() - P.flatten(), np.inf)
        P = P0
    return P

#%% correction
def correct_step(u_s,v_s, P, rho, dt, imax, imin, jmax, jmin, dx, dy):
    u_new = cp.deepcopy(u_s)
    v_new = cp.deepcopy(v_s)
    # only updates interior nodes, correct boundary at the end
    for j in range(jmin,jmax+1):
        for i in range(imin+1, imax+1):
            u_new[i,j] = u_s[i,j] - dt/rho*(P[i,j]-P[i-1,j])/dx # update v
            
    for j in range(jmin+1, jmax):
        for i in range(imin,imax):
            v_new[i,j] = v_s[i,j] - dt/rho*(P[i,j]-P[i,j-1])/dx # update u
    
    return u_new, v_new

#%% boundary condition
def boundary_corr(u,v, imin, imax, jmin, jmax):
    u_b = cp.deepcopy(u)
    v_b = cp.deepcopy(v)
    
    # u velocity boundary condition
    # bottom
    u_b[:,jmin-1] = 0 - u_b[:,jmin]
    # top
    u_b[:,jmax+1] = 2*1 - u_b[:,jmax]
    # # left 
    # u_b[imin,:] = 0
    # # right
    # u_b[imax,:] = 0

    # u velocity boundary condition
    # # bottom
    # v_b[:,jmin] = 0
    # # top
    # v_b[:,jmax] = 0
    # left 
    v_b[imin-1,:] = 0 - v_b[imin,:]
    # right
    v_b[imax+1,:] = 0 - v_b[imax,:]
    
    return u_b,v_b


#%% initialization
# # u velocity
# fu = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*(-(x-x0)/sigma**2)
# # v velocity
# fv = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*(-(y-y0)/sigma**2)

# u_0 = fu(x,y)
# v_0 = fu(x,y)

u = np.zeros([nx+2,nx+2])
v = np.zeros([nx+2,nx+2])
P = np.zeros([nx+2,nx+2])

#%% runing the iteration
epoch = 100
T = 1
dt = 0.001
nt = T/dt

for t in range(int(nt)):
    u,v =  boundary_corr(u,v, imin, imax, jmin, jmax)
    
    u_s = u_star(u,v,dt,dxi,dyi,nu)
    v_s = v_star(u,v,dt,dxi,dyi,nu)
    
    P_updated = pressure_poisson(P, u_s, v_s, rho, dt, dx, dy,  imin, imax, jmin, jmax)
    
    u, v = correct_step(u_s,v_s, P_updated, rho, dt, imax, imin, jmax, jmin, dx, dy)
    
    
#%% plotting
plt.figure()
plt.imshow(u[imin:imax-1,jmin:jmax-1])