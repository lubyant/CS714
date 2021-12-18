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
import time
import math
#%% Navier Stokes Solvers
# momentum discretization
 
def convection_term(u,v):
     c_x = cp.deepcopy(u)
     c_y = cp.deepcopy(v)
     for j in range(jmin,jmax+1):
         for i in range(imin+1, jmax+1):  
             v_cur = (v[i-1,j] + v[i-1,j+1] + v[i,j] + v[i,j+1])/4
             c_x[i,j] = u[i,j]*(u[i+1,j]-u[i-1,j])*0.5*dxi+v_cur*(u[i,j+1]-u[i,j-1])*0.5*dyi
     for j in range(jmin+1,jmax+1):
         for i in range(imin,imax+1):
             u_cur = (u[i,j-1] + u[i,j] + u[i+1,j-1] + u[i+1,j])/4
             c_y[i,j] = u_cur*(v[i+1,j]-v[i-1,j])*0.5*dxi+v[i,j]*(v[i,j+1]-v[i,j-1])*0.5*dyi
     return c_x,c_y
def diffusion_term(u,v):
     d_x = cp.deepcopy(u)
     d_y = cp.deepcopy(v)    
     for j in range(jmin,jmax+1):
         for i in range(imin+1, jmax+1):  
             lap_op = (u[i+1,j] - 2*u[i,j] + u[i-1,j])*dxi**2 + (u[i,j+1]-2*u[i,j]+u[i,j-1])*dyi**2
             d_x[i,j] = nu*lap_op
     for j in range(jmin+1,jmax+1):
         for i in range(imin,imax+1):
             lap_op = (v[i+1,j] - 2*v[i,j] + v[i-1,j])*dxi**2 + (v[i,j+1]-2*v[i,j]+v[i,j-1])*dyi**2
             d_y[i,j] = nu*lap_op
     return d_x,d_y
 
def H_term(C,D):
     return -C+D
 
 
# u direction
def predictor_star(u,v,Hx,Hy):
    u_s = cp.deepcopy(u)
    v_s = cp.deepcopy(v)

    
    u_s = u + dt*Hx
    v_s = v + dt*Hy
    
    return u_s, v_s     


# pressure poisson solver
def construct_matrix(u_s,v_s):
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
    return L    

def pressure_poisson_matrix(u_s,v_s,L):
    # L = np.zeros([nx*ny,nx*ny])
    # for j in range(ny):
    #     for i in range(nx):
    #         L[i+(j)*nx, i + (j)*nx] = 2*dxi**2+2*dyi**2
    #         for ii in range(i-1,i+2,2):
    #             if ii>=0 and ii<nx:
    #                 L[i+(j)*nx,ii+(j)*nx] = -dxi**2
    #             else:
    #                 L[i+(j)*nx,i+(j)*nx] -= dxi**2
            
    #         for jj in range(j-1,j+2,2):
    #             if jj>=0 and jj< ny:
    #                 L[i+(j)*nx,i+(jj)*nx] = -dyi**2
    #             else:
    #                 L[i+(j)*nx,i+(j)*nx] -= dyi**2
    # L[0,:] = 0
    # L[0,0] = 1
    
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

# correction
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

# boundary condition
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
#%% fine grid
Lx = 1
Ly = 1
nx = 64
ny = 64
x0 = Lx/2
y0 = Ly/2

# physical parameter
Re = 100
u_top = 1
gamma = 1
sigma = 2.5
nu = 0.01
rho = u_top*Lx/Re

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

u = np.zeros([nx+2,nx+2])
v = np.zeros([nx+2,nx+2])
P_updated = np.zeros([nx+1,nx+1])

epoch = 100
T = 5
dt = 0.001
nt = T/dt
L = construct_matrix(u,v)

 
for t in range(int(nt)):
    u,v = boundary_corr(u,v, imin, imax, jmin, jmax)
    c_x,c_y = convection_term(u,v)
    d_x,d_y = diffusion_term(u,v)

    Hx = H_term(c_x,d_x)
    Hy = H_term(c_y,d_y)
    
    u_s, v_s  = predictor_star(u,v,Hx,Hy)
    
    c_x_2,c_y_2 = convection_term(u_s,v_s)
    d_x_2,d_y_2 = diffusion_term(u_s,v_s)

    Hx_2 = H_term(c_x_2,d_x_2)
    Hy_2 = H_term(c_y_2,d_y_2)

    u_s, v_s  = predictor_star(u,v,0.5*(Hx+Hx_2),0.5*(Hy+Hy_2))
    P_updated += pressure_poisson_matrix(u_s,v_s,L)
    u, v = correct_step(u_s,v_s, P_updated, rho, dt, imax, imin, jmax, jmin, dx, dy)
u_sol = u[imin:imax+1,jmin:jmax+1]
v_sol = v[imin:imax+1,jmin:jmax+1]
# Navier Stokes Solvers
#%% Pysical setting
# physical domain
time_b = []
time_ab = []
time_rk = []
err_b = []
err_ab = []
err_rk = []
for grid_size_id in range(3,6):
    
    grid_size = 2**grid_size_id
    
    Lx = 1
    Ly = 1
    nx = grid_size
    ny = grid_size
    x0 = Lx/2
    y0 = Ly/2
    
    # physical parameter
    Re = 100
    u_top = 1
    gamma = 1
    sigma = 2.5
    nu = 0.01
    rho = u_top*Lx/Re
    
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
       
    
    #%% Basic
    # # u velocity
    # fu = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*(-(x-x0)/sigma**2)
    # # v velocity
    # fv = lambda x,y : np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)*(-(y-y0)/sigma**2)
    
    # u_0 = fu(x,y)
    # v_0 = fu(x,y)
    
    u = np.zeros([nx+2,nx+2])
    v = np.zeros([nx+2,nx+2])
    P_updated = np.zeros([nx+1,nx+1])
    
    epoch = 100
    T = 5
    dt = 0.001
    nt = T/dt
    L = construct_matrix(u,v)
    start = time.time()
    
    for t in range(int(nt)):
        u,v =  boundary_corr(u,v, imin, imax, jmin, jmax)
        c_x,c_y = convection_term(u,v)
        d_x,d_y = diffusion_term(u,v)
        Hx = H_term(c_x,d_x)
        Hy = H_term(c_y,d_y)
        
        u_s, v_s  = predictor_star(u,v,Hx,Hy)
        
        P_updated += pressure_poisson_matrix(u_s,v_s,L)
        # P_updated  = pressure_poisson(P, u_s, v_s, rho, dt, dx, dy, imin, imax, jmin, jmax)
        
        u, v = correct_step(u_s,v_s, P_updated, rho, dt, imax, imin, jmax, jmin, dx, dy)
        
    end = time.time()   
    dur_b = end - start 
    u_b = u[imin:imax+1,jmin:jmax+1]
    v_b = v[imin:imax+1,jmin:jmax+1]
    
    err = (u_b - u_sol[::2**(6-grid_size_id),::2**(6-grid_size_id)]).flatten()
    err_b.append(LA.norm(err,np.inf))
    
    #%% 2nd order AB and CN trick 
    u = np.zeros([nx+2,nx+2])
    v = np.zeros([nx+2,nx+2])
    P_updated = np.zeros([nx+1,nx+1])
    
    epoch = 100
    T = 5
    dt = 0.001
    nt = T/dt
    # define the u and v at n-1
    u_p,v_p =  boundary_corr(u,v, imin, imax, jmin, jmax)
    
    start = time.time()
    
    for t in range(int(nt)):
        u,v = boundary_corr(u,v, imin, imax, jmin, jmax)
        c_x,c_y = convection_term(u,v)
        d_x,d_y = diffusion_term(u,v)
        
        c_x_past,c_y_past = convection_term(u_p,v_p)
        
        Hx = H_term(c_x,d_x)
        Hy = H_term(c_y,d_y)
        u_s, v_s  = predictor_star(u,v,Hx,Hy)
        
        P_updated += pressure_poisson_matrix(u_s,v_s,L)
    
        
        u_future, v_future = correct_step(u_s,v_s, P_updated, rho, dt, imax, imin, jmax, jmin, dx, dy)
        
        d_x_future,d_y_future = diffusion_term(u_future,v_future)
        
        Hx = H_term(1.5*c_x-0.5*c_x_past,0.5*d_x+0.5*d_x_future)
        Hy = H_term(1.5*c_y-0.5*c_y_past,0.5*d_y+0.5*d_y_future)
        
        u_s, v_s  = predictor_star(u,v,Hx,Hy)
        P_updated += pressure_poisson_matrix(u_s,v_s,L)
        u_p,v_p = u,v
        u, v = correct_step(u_s,v_s, P_updated, rho, dt, imax, imin, jmax, jmin, dx, dy)
    
    end = time.time()   
    dur_ab = end - start 
    u_ab = u[imin:imax+1,jmin:jmax+1]
    v_ab = v[imin:imax+1,jmin:jmax+1]
    
    err = (u_ab - u_sol[::2**(6-grid_size_id),::2**(6-grid_size_id)]).flatten()
    err_ab.append(LA.norm(err,np.inf))
    #%% RK2 trick
    u = np.zeros([nx+2,nx+2])
    v = np.zeros([nx+2,nx+2])
    P_updated = np.zeros([nx+1,nx+1])
    
    epoch = 100
    T = 5
    dt = 0.001
    nt = T/dt
    
    start = time.time()
     
    for t in range(int(nt)):
        u,v = boundary_corr(u,v, imin, imax, jmin, jmax)
        c_x,c_y = convection_term(u,v)
        d_x,d_y = diffusion_term(u,v)
    
        Hx = H_term(c_x,d_x)
        Hy = H_term(c_y,d_y)
        
        u_s, v_s  = predictor_star(u,v,Hx,Hy)
        
        c_x_2,c_y_2 = convection_term(u_s,v_s)
        d_x_2,d_y_2 = diffusion_term(u_s,v_s)
    
        Hx_2 = H_term(c_x_2,d_x_2)
        Hy_2 = H_term(c_y_2,d_y_2)
    
        u_s, v_s  = predictor_star(u,v,0.5*(Hx+Hx_2),0.5*(Hy+Hy_2))
        P_updated += pressure_poisson_matrix(u_s,v_s,L)
        u, v = correct_step(u_s,v_s, P_updated, rho, dt, imax, imin, jmax, jmin, dx, dy)
    
    end = time.time()
    dur_rk2 = end-start
    u_rk2 = u[imin:imax+1,jmin:jmax+1]
    v_rk2 = v[imin:imax+1,jmin:jmax+1]
    err = (u_rk2 - u_sol[::2**(6-grid_size_id),::2**(6-grid_size_id)]).flatten()
    err_rk.append(LA.norm(err,np.inf))          
    
    time_b.append(dur_b)
    time_ab.append(dur_ab)
    time_rk.append(dur_rk2)
#%% plotting
plt.figure()
x_time = np.array([8,16,32]).flatten()
plt.plot(x_time.T,np.array(time_b).flatten().T,label='FSM')
plt.plot(x_time,np.array(time_ab).flatten(),label='FSM+AB2')
plt.plot(x_time,np.array(time_rk).flatten(),label='FSM+RK2')
plt.xlabel('Grid size')
plt.ylabel('Elpased time(s)')
plt.legend()

log_err_b = []
log_err_ab = []
log_err_rk = []

norm = [1.58,1.72,2.33]
for i in range(3):
    log_err_b.append( math.log(err_b[i],2)+norm[i])
    log_err_ab.append( math.log(err_ab[i],2)+norm[i])
    log_err_rk.append( math.log(err_rk[i],2)+norm[i])
plt.figure()
x_time = np.array([8,16,32]).flatten()
plt.plot(x_time.T,np.array(log_err_b).flatten().T,label='FSM')
plt.plot(x_time,np.array(log_err_ab).flatten(),label='FSM+AB2')
plt.plot(x_time,np.array(log_err_rk).flatten(),label='FSM+RK2')
plt.xlabel('Grid size')
plt.ylabel('log error')
plt.legend()