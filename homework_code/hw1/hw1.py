# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:16:22 2021

@author: luboy
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nplng
import copy as cp
from scipy import interpolate
# %%
# generate a really fining grid for the analytical solution
h0 = 0.01;
hx = h0; # spacing at x direction
hy = h0; # spacing at y direction
pi = np.pi

x0 = np.linspace(0, 1,num=int(1/hx)+1)
y0 = np.linspace(0, 1,num=int(1/hx)+1)

u_a = np.ones([int(1/hx)+1,int(1/hy)+1]) # analystical sol
u_0 = np.ones([int(1/hx)+1,int(1/hy)+1]) # initial


eps = 1e-2
error = 1
count0 = 0
while error>1/(1/hx+1)**2:# Jacobi

    for j in range(len(y0)):
        for i in range(len(x0)):
            if i == 0: # Dirichlet BC
                u_a[i,j] = np.cos(2*pi*y0[j])
            elif i == len(x0)-1: # Dirichlet BC
                u_a[i,j] = 0
            elif j == 0: # Neumann BC
                u_a[i,j] = u_0[i,j+1]
            elif j == len(y0)-1: # Neumann BC
                u_a[i,j] = u_0[i,j-1]  
            else:
                # internal grid
                u_a[i,j] = (u_0[i+1,j] + u_0[i-1,j] + u_0[i,j+1] + u_0[i,j-1])/4
    error = nplng.norm(u_a - u_0) 
    u_0 = cp.deepcopy(u_a)
    count0 = count0 + 1
    if count0 % 1000 == 0:
        print(count0,error)


# plotting
# I don't know if there is a bug in this plotting code. It seems that the image
# is output in blank. I just directly save this plot to jpg.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x,y = np.meshgrid(x0,y0)
ax.plot_surface(x, y, u_a, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.title('Profile of FD solution')

plt.savefig('fig1')
plt.show()
# %%

# define a function for calculating the error between numerical and analytical
def error_cal(u_a,u_n,h0,h):
    u_c = cp.deepcopy(u_n) # copy of nest matrix
    for i in range(u_n.shape[0]):
        for j in range(u_n.shape[1]):
            u_c[i][j] = u_a[int(i*h/h0)][int(j*h/h0)]
    # error = np.linalg.norm(u_c-u_n,np.inf)
    error = np.max(np.abs((u_c-u_n)).reshape(1,-1))
    return error

# generate the FD for different spacing
h = [0.02, 0.04, 0.05, 0.1, 0.2]
E_list = []
c_list = []
for item in h:
    # square spacing
    hx = item 
    hy = item
    
    x = np.linspace(0, 1,num=int(1/hx)+1)
    y = np.linspace(0, 1,num=int(1/hx)+1)

    u_n = np.ones([int(1/hx)+1,int(1/hy)+1]) # numerical sol
    u_0 = np.ones([int(1/hx)+1,int(1/hy)+1]) # initial
    
    eps = 1e-2
    error = 1
    count = 0
    while error> 1/(1/item+1)**2:# Jacobi
        init = cp.deepcopy(u_a)
        for j in range(len(y)):
            for i in range(len(x)):
                if i == 0: # Dirichlet BC
                    u_n[i,j] = np.cos(2*pi*y[j])
                elif i == len(x)-1: # Dirichlet BC
                    u_n[i,j] = 0
                elif j == 0: # Neumann BC
                    u_n[i,j] = u_0[i,j+1]
                elif j == len(y)-1: # Neumann BC
                    u_n[i,j] = u_0[i,j-1]  
                else:
                    # internal grid
                    u_n[i,j] = (u_0[i+1,j] + u_0[i-1,j] + u_0[i,j+1] + u_0[i,j-1])/4
        error = nplng.norm(u_n - u_0) 
        u_0 = cp.deepcopy(u_n) 
        count = count + 1
        if count % 500 == 0:
            print(count,error)    
    E = error_cal(u_a,u_n,h0,item)    
    E_list.append(E)
    c_list.append(count)

fig2 = plt.figure()
plt.plot(-np.log2(h),-np.log2(E_list),label='Error pattern')
plt.plot(np.linspace(0,6,10),2*np.linspace(0,6,10),label='slope 2 line')
plt.xlabel('-log(h)')
plt.ylabel('-log(error)')
plt.title('Error distribution for different spacing')
plt.legend()
plt.savefig('fig2')

fig3 = plt.figure()
he = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
ce = np.zeros(6)
for i in range(6):
    if i == 0:
        ce[i] = count0
    else: ce[i] = c_list[i-1]
plt.plot(1/np.array(he),ce,label='The grid-iteration curve')
plt.plot(np.linspace(0,100,100),np.linspace(0,100,100)**2,label='reference quadratic curve')
plt.plot(np.linspace(0,100,100),np.linspace(0,100,100)**2*np.log(np.linspace(0,100,100)),label='reference n2logn curve')

plt.xlabel('Number of grid')
plt.ylabel('Number of iteration')
plt.title('Grid size vs iteration steps')
plt.legend()
plt.savefig('fig3')

# %%
# fhat
h = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
E_list_hat = []
c_list_hat = []
for item in h:
    # square spacing
    hx = item 
    hy = item
    
    x = np.linspace(0, 1,num=int(1/hx)+1)
    y = np.linspace(0, 1,num=int(1/hx)+1)

    u_n = np.ones([int(1/hx)+1,int(1/hy)+1]) # numerical sol
    u_0 = np.ones([int(1/hx)+1,int(1/hy)+1]) # initial
    
    eps = 1e-2
    error = 1
    count = 0
    while error> 1/(1/item+1)**2:# Jacobi
        init = cp.deepcopy(u_a)
        for j in range(len(y)):
            for i in range(len(x)):
                if i == 0 and j == 0: # Dirichlet BC
                    u_n[i,j] = 0
                elif i == 0 and j != 0:
                    u_n[i,j] = 1
                elif i == len(x)-1: # Dirichlet BC
                    u_n[i,j] = 0
                elif j == 0: # Neumann BC
                    u_n[i,j] = u_0[i,j+1]
                elif j == len(y)-1: # Neumann BC
                    u_n[i,j] = u_0[i,j-1]  
                else:
                    # internal grid
                    u_n[i,j] = (u_0[i+1,j] + u_0[i-1,j] + u_0[i,j+1] + u_0[i,j-1])/4
        error = nplng.norm(u_n - u_0) 
        u_0 = cp.deepcopy(u_n) 
        count = count + 1
        if count % 500 == 0:
            print(count,error)
    if item == 0.01:
        u_a = u_n
    E = error_cal(u_a,u_n,h0,item)    
    E_list_hat.append(E)
    c_list_hat.append(count)

fig4 = plt.figure()
plt.plot(1/np.array(he),ce,label='f(y)')
plt.plot(1/np.array(h),c_list_hat,label='f(y)_hat')
plt.legend()
plt.xlabel('Number of grid')
plt.ylabel('Number of iteration')
plt.title('f(y) vs f(y)_hat convergence speed')
plt.savefig('fig4')
# %%
# multigrid method

# define a coarsen method
def coarsen(matrix):
    coarsen = matrix[::2,::2]
    c = cp.deepcopy(coarsen)
    return c

def residual(u):
    # rv = f - Au
    r = cp.deepcopy(u)
    y0 = np.linspace(0, 1,num=u.shape[1])
    for j in range(u.shape[1]):
        for i in range(u.shape[0]):
            if i == 0: # Dirichlet BC
                r[i,j] = -u[i,j] + np.cos(2*pi*y0[j])
            elif i == u.shape[0]-1: # Dirichlet BC
                r[i,j] = -u[i,j] + 0
            elif j == 0: # Neumann BC
                r[i,j] = - u[i,j] + u[i,j+1]
            elif j == u.shape[1]-1: # Neumann BC
                r[i,j] = - u[i,j] + u[i,j-1]  
            else:
                # internal grid
                r[i,j] = - u[i,j] + (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])/4
    
    return r

def error_approximate(r):
    error = 1
    count_e = 0
    u = cp.deepcopy(r)
    u_0 = cp.deepcopy(r)
    #while error > 1/(1/r.shape[0])**2:
    while count < 100:
        for j in range(r.shape[1]):
            for i in range(r.shape[0]):
                if i == 0:
                    u[i,j] = -r[i,j]
                elif i == r.shape[0] -1:
                    u[i,j] = -r[i,j]
                elif j == 0:
                    u[i,j] = -r[i,j] + u_0[i,j+1]
                elif j == r.shape[1] - 1:
                    u[i,j] = -r[i,j] + u_0[i,j-1]
                else:
                    u[i,j] = -r[i,j] + (u_0[i+1,j] + u_0[i-1,j] + u_0[i,j+1] + u_0[i,j-1])/4
        
        error = nplng.norm(u - u_0) 
        count_e = count_e + 1
    
    return count_e,u


def interpolater(x,y,u):
    x2 = np.linspace(0, 1,num=u_e.shape[0])
    y2 = np.linspace(0, 1,num=u_e.shape[1])
    f = interpolate.interp2d(x2, y2, u, kind='linear')
    
    u_new = f(x,y)
    
    return u_new

h = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
E_list_MG = []
c_list_MG = []
#number of steps to iterate at the finest level
inner_its = 100
 
#number of steps to iterate at the coarse level        
inner2_its = 50
for item in h:
    # square spacing
    hx = item 
    hy = item
    
    x = np.linspace(0, 1,num=int(1/hx)+1)
    y = np.linspace(0, 1,num=int(1/hx)+1)

    u_n = np.ones([int(1/hx)+1,int(1/hy)+1]) # numerical sol
    u_0 = np.ones([int(1/hx)+1,int(1/hy)+1]) # initial
    
    eps = 1e-2
    error = 1
    count = 0
    iteration = 0 # iteration tracer for MG
    while error> 1/(1/item+1)**2:# Jacobi
        
        for j in range(len(y)):
            for i in range(len(x)):
                if i == 0: # Dirichlet BC
                    u_n[i,j] = np.cos(2*pi*y[j])
                elif i == len(x)-1: # Dirichlet BC
                    u_n[i,j] = 0
                elif j == 0: # Neumann BC
                    u_n[i,j] = u_0[i,j+1]
                elif j == len(y)-1: # Neumann BC
                    u_n[i,j] = u_0[i,j-1]  
                else:
                    # internal grid
                    u_n[i,j] = (u_0[i+1,j] + u_0[i-1,j] + u_0[i,j+1] + u_0[i,j-1])/4

        count = count + 1
        
        # start the MG every inner_its iteration
        if count % inner_its == 0:
            # print("coarse")
            u_c = coarsen(u_n)
            r = residual(u_n)
            r_c = coarsen(r)
            
            # solve Ae = -r
            count_e, u_e = error_approximate(r_c)
            
            # intepolate the u_e
            u_interp = interpolater(x,y,u_e)
            
            # subtract the error
            u_n = u_n - u_interp
            
            # adding counts
            count = count + count_e
        
        error = nplng.norm(u_n - u_0) 
        u_0 = cp.deepcopy(u_n) 
        if count % 500 == 0:
            print(count,error)
    if item == 0.01:
        u_a = u_n
    E = error_cal(u_a,u_n,h0,item)    
    E_list_MG.append(E)
    c_list_MG.append(count)

fig5= plt.figure()
plt.plot(1/np.array(he),ce,label='Jacobi')
plt.plot(1/np.array(h),c_list_MG,label='multigrid')
plt.legend()
plt.xlabel('Number of grid')
plt.ylabel('Number of iteration')
plt.title('jacobi vs multigrid convergence speed')
plt.savefig('fig5')