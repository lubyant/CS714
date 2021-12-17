# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:18:59 2021

@author: luboy
"""
import numpy as np
import copy as cp
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib import cm
# Navier Stokes Solvers
#%% Pysical setting
# physical domain
Lx = 30
Ly = 30
nx = 16
ny = 16
x0 = Lx/2
y0 = Ly/2

# physical parameter
Re = 100
u_top = 1
gamma = 1
sigma = 2.5
nu = u_top*Lx/Re
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

def pressure_poisson_matrix(u_s,v_s):
    L = np.zeros([nx*ny,nx*ny])
    for j in range(ny):
        for i in range(nx):
            L[i+(j)*nx, i + (j)*nx] = 2*dxi**2+2*dyi**2
            for ii in range(i-1,i+2,2):
                if ii>=0 and ii<nx:
                    L[i+(j)*nx,ii+(j)*nx] = -dxi**2
                else:
                    L[i+(j)*nx,i+(j)*nx] -= dxi**2
            
            for jj in range(j-1,j+2,2):
                if jj>=0 and jj< ny:
                    L[i+(j)*nx,i+(jj)*nx] = -dyi**2
                else:
                    L[i+(j)*nx,i+(j)*nx] -= dyi**2
    L[0,:] = 0
    L[0,0] = 1
    
    R = RHS(u_s,v_s)
    
    P_v = spsolve(L,R)
    
    n = 0
    p = np.zeros([imax+1,jmax+1])
    for j in range(jmin,jmax+1):
        for i in range(imin,imax+1):
            p[i,j] = P_v[n]
            n = n+1
    
    return p
def RHS(u_s,v_s):
    n = 0
    R = []
    for j in range(jmin,jmax+1):
        for i in range(imin,imax+1):
            n += 1 
            cur_r = (-rho/dt)*((u_s[i+1,j]-u_s[i,j])*dxi+(v_s[i,j+1]-v_s[i,j])*dyi)
            R.append(cur_r)
    R = np.array(R)
    return R.reshape(-1,1)   
 
def pressure_poisson(P, u_s, v_s, rho, dt, dx, dy, imin, imax, jmin, jmax):
    P0 = cp.deepcopy(P)
    error = np.inf
    while error > 1/(P.shape[0]*P.shape[1]):
        for i in range(imin,imax+1):
            for j in range(imin,imax+1):
                rhs = rho/dt*((u_s[i+1,j] - u_s[i,j])/dx + (v_s[i,j+1] - v_s[i,j])/dy)
                if i == imin and j == jmin: # (0,0)
                    P0[i,j] = 1/(dx**2+dy**2)*(dy**2*P[i+1,j]+dx**2*P[i,j+1]-dx**2*dy**2*rhs)
                    # P0[i,j] = 0
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

def plot_uvp():
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#%% correction
def correct_step(u_s,v_s, P, rho, dt, imax, imin, jmax, jmin, dx, dy):
    u_new = cp.deepcopy(u_s)
    v_new = cp.deepcopy(v_s)
    # only updates interior nodes, correct boundary at the end
    for j in range(jmin,jmax+1):
        for i in range(imin+1, imax+1):
            u_new[i,j] = u_s[i,j] - dt/rho*(P[i,j]-P[i-1,j])/dx # update v
            
    for j in range(jmin+1, jmax+1):
        for i in range(imin,imax+1):
            v_new[i,j] = v_s[i,j] - dt/rho*(P[i,j]-P[i,j-1])/dx # update u
    
    return u_new, v_new

#%% boundary condition
def boundary_corr(u,v, imin, imax, jmin, jmax):
    u_b = cp.deepcopy(u)
    v_b = cp.deepcopy(v)
    
    # u velocity boundary condition
    # bottom
    u_b[:,jmin-1] = u_b[:,jmin]
    # top
    u_b[:,jmax+1] = u_b[:,jmax]
    # left 
    u_b[imin,:] = 1
    # right
    u_b[imax,:] = 0

    # u velocity boundary condition
    # bottom
    v_b[:,jmin] = 0
    # # top
    v_b[:,jmax] = 0
    # left 
    v_b[imin-1,:] = 0 - v_b[imin,:]
    # right
    v_b[imax+1,:] = 0 - v_b[imax,:]
    
    return u_b,v_b


#%% initialization
# u velocity
fu = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*(-(y-y0)/sigma**2)
# v velocity
fv = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*((x-x0)/sigma**2)



u = np.zeros([nx+2,nx+2])
v = np.zeros([nx+2,nx+2])

for j in range(len(v)):
    for i in range(len(u)):
        u[i,j] = fu(x[0,i],y[0,j])
        v[i,j] = fv(x[0,i],y[0,j])
        
P_updated = np.zeros([nx+1,nx+1])

#%% runing the iteration
epoch = 100
T = 0.01
dt = 0.001
nt = T/dt

for t in range(int(nt)):
    u,v =  boundary_corr(u,v, imin, imax, jmin, jmax)
    
    u_s = u_star(u,v,dt,dxi,dyi,nu)
    v_s = v_star(u,v,dt,dxi,dyi,nu)
    
    P_updated += pressure_poisson_matrix(u_s,v_s)
    # P_updated  = pressure_poisson(P, u_s, v_s, rho, dt, dx, dy, imin, imax, jmin, jmax)
    
    u, v = correct_step(u_s,v_s, P_updated, rho, dt, imax, imin, jmax, jmin, dx, dy)
    
    if t % 10 ==0:
        plot_uvp()
#%% plotting
plt.figure()

X,Y = np.meshgrid(x,y)
plt.contourf(u[imin:imax,jmin:jmax].T,levels=20)

plt.figure()
u_c = u[15,:]

plt.plot(y.flatten(),u_c.flatten(),label='Our results')

ref = np.array([1,0.84123,0.78871,0.73722,0.68717,0.23151,0.00332,-0.13641,-0.20581,-0.21090,-0.15662,-0.1015,-0.06434,-0.04775,-0.04192,-0.03717,0])
grid = np.array([129,126,125,124,123,110,95,80,65,59,37,23,14,10,9,8,1])/129
plt.scatter(grid,ref,label='Ghia(1982) results',color='red')
plt.xlabel('y/L')
plt.ylabel('u')
plt.legend()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X,Y, np.array(u,dtype='float'),cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.1, aspect=2)
plt.title('Pressure at 1s')
plt.show()