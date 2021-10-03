# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:16:22 2021

@author: luboy
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nplng
import copy as cp


# generate a really fining grid for the analytical solution
hx = 0.001; # spacing at x direction
hy = 0.001; # spacing at y direction
pi = np.pi

x = np.linspace(-1, 1,num=int(2/hx)+1)
y = np.linspace(-1, 1,num=int(2/hx)+1)

u_a = np.ones([int(2/hx)+1,int(2/hy)+1]) # analystical sol
u_0 = np.ones([int(2/hx)+1,int(2/hy)+1]) # initial


eps = 1e-2
error = 1
count = 0
while error>eps:# Jacobi
    init = cp.deepcopy(u_a)
    for j in range(len(y)):
        for i in range(len(x)):
            if i == 0: # Dirichlet BC
                u_a[i,j] = np.cos(2*pi*y[j])
            elif i == len(x)-1: # Dirichlet BC
                u_a[i,j] = 0
            elif j == 0: # Neumann BC
                u_a[i,j] = u_0[i,j+1]
            elif j == len(y)-1: # Neumann BC
                u_a[i,j] = u_0[i,j-1]  
            else:
                # internal grid
                u_a[i,j] = (u_0[i+1,j] + u_0[i-1,j] + u_0[i,j+1] + u_0[i,j-1])/4
    error = nplng.norm(u_a - init) 
    u_0 = u_a
    count = count + 1


# plotting
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x,y = np.meshgrid(x,y)
ax.plot_surface(x, y, u_a, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

plt.show()

# define a function for calculating the error between numerical and analytical
def error_cal(u_a,u_n):
    return 0

# generate the FD for different spacing
h = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
E_list = []
for i in h:
    # square spacing
    hx = h 
    hy = h
    
    x = np.linspace(-1, 1,num=int(2/hx)+1)
    y = np.linspace(-1, 1,num=int(2/hx)+1)

    u_n = np.ones([int(2/hx)+1,int(2/hy)+1]) # numerical sol
    u_0 = np.ones([int(2/hx)+1,int(2/hy)+1]) # initial
    
    eps = 1e-2
    error = 1
    count = 0
    while error>eps:# Jacobi
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
        error = nplng.norm(u_n - init) 
        u_0 = u_n
        count = count + 1
    
        E = error_cal(u_a,u_n)    
        E_list.append(E)
