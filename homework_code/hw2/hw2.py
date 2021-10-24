# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 10:00:29 2021

@author: lby
"""

import numpy as np
import numpy.linalg as ling
import copy as cp
from scipy import interpolate
import matplotlib.pyplot as plt

# implement the steepest gradient descent
def sgd(A, x0, b, tol=1e-4, maxiter=100):
    # initial residual
    r0 = b - np.matmul(A,x0)
    
    # interation
    for i in range(maxiter):
        w = np.matmul(A,r0)
        alpha = np.dot(r0.transpose(),r0)/np.dot(r0.transpose(),w)
        x = x0 + alpha*r0
        r = r0 - alpha*w
        if ling.norm(r) < tol:
            return x
        else:
            r0 = cp.deepcopy(r)
            x0 = cp.deepcopy(x)

def scgd(A, x0, b, tol=1e-4, maxiter=100):
    # initial residual
    r0 = b - np.matmul(A,x0)
    p0 = cp.deepcopy(r0)

    for i in range(maxiter):
        w = np.matmul(A,p0)
        alpha = np.dot(r0.transpose(),r0)/np.dot(p0.transpose(),w)
        x = x0 + alpha*p0
        r = r0 - alpha*w        
        if ling.norm(r) < tol:
            return x
        else:
            beta = np.dot(r.transpose(),r)/np.dot(r0.transpose(),r0)
            p = r + beta*p0    
            r0 = cp.deepcopy(r)
            x0 = cp.deepcopy(x)    
            p0 = cp.deepcopy(p)
    return x       

def probB():
    
    norm_list = []
    for N in range(95,105):
        x = np.linspace(0,1,N)
        y = np.exp(-400*(x-0.5)**2)
        
        xi = np.linspace(0,1,N+1)
        yi = np.exp(-400*(xi-0.5)**2)   
        
        f = interpolate.interp1d(xi, yi)
        y_new = f(x)
        norm = np.linalg.norm((y_new-y),np.inf)
        norm_list.append((N,norm))
    
    plt.figure()
    plt.plot(np.array(norm_list)[:,0]. astype(int),np.array(norm_list)[:,1])
    plt.plot(np.linspace(95,105,20),0.01*np.ones([20,1]))
    plt.xlabel('Sampling size N')
    plt.ylabel('Inf norm for sampling error')
    plt.savefig('problemB.jpg') 

def wave_solution():
    f = lambda x : np.exp(-400*(x-0.5)**2) 
    
    N = 1024 # use the solution from problem b
    
    dt = 1/(2*N) 
    dx = 1/N
    
    u = np.zeros([N+1,N+1,N+1])
    
    # initial condition for first derivative of time
    for i in range(N+1):
        for j in range(N+1):
            u[i,j,1] = dt*f((i-1)*dx)*f((i-1)*dx)
    
    # pointwise linear function:
    for t in range(2,N+1):
        for i in range(1,N):
            for j in range(1,N):
                u[i,j,t] = (dt**2/dx**2)*(u[i+1,j,t-1]+u[i,j+1,t-1]+u[i-1,j,t-1]+u[i,j-1,t-1]-4*u[i,j,t-1])-u[i,j,t-2]+2*u[i,j,t-1]
 
def wave_solver():
    pass
    
if __name__ == '__main__':
    x0 = np.zeros([10,1])
    
    x = np.linspace(3,10,10).reshape(-1,1)
    A = 2*np.eye(10) + np.eye(10,k=1) + np.eye(10,k=-1)

    b = np.matmul(A,x).reshape(-1,1)
    
    test = scgd(A,x0,b,maxiter=1000)
    
    # problem B
    probB()
    
    # problem C
    wave_solver()