# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:16:22 2021

@author: luboy
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nplng
import copy as cp

hx = 0.01; # spacing at x direction
hy = 0.01; # spacing at y direction
pi = np.pi

x = np.linspace(-1, 1,num=int(2/hx)+1)
y = np.linspace(-1, 1,num=int(2/hx)+1)

init = np.zeros([int(2/hx)+1,int(2/hy)+1])
u = np.ones([int(2/hx)+1,int(2/hy)+1])

eps = 1e-2
error = 1
count = 0
while error>eps:
    init = cp.deepcopy(u)
    for j in range(len(y)):
        for i in range(len(x)):
            if i == 0: # Dirichlet BC
                u[i,j] = np.cos(2*pi*y[j])
            elif i == len(x)-1: # Dirichlet BC
                u[i,j] = 0
            elif j == 0: # Neumann BC
                u[i,j] = u[i,j+1]
            elif j == len(y)-1: # Neumann BC
                u[i,j] = u[i,j-1]  
            else:
                # internal grid
                u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])/4
    error = nplng.norm(u - init) 
    
    count = count + 1

# plotting
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x,y = np.meshgrid(x,y)
ax.plot_surface(x, y, u, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

plt.show()