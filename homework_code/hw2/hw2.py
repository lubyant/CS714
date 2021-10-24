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

#%%
# Here is the secion for question A
# question A (a)
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
            return (x,i)
        else:
            r0 = cp.deepcopy(r)
            x0 = cp.deepcopy(x)
# question A (b)
# implement the conjugate gradient descent
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
            return (x,i)
        else:
            beta = np.dot(r.transpose(),r)/np.dot(r0.transpose(),r0)
            p = r + beta*p0    
            r0 = cp.deepcopy(r)
            x0 = cp.deepcopy(x)    
            p0 = cp.deepcopy(p)
       
    
# question A(c)
def convergence_rate():
    # rate vs size
    iter_list = []
    
    for n in range(10,100):
        arr = np.zeros([1,n]).flatten()
        arr[0] = 0.1
        arr[-1] = 10
        
        # generate a diagnol matrix
        A = np.diag(arr)
        
        # call the scgd
        x0 = np.zeros([n,1])
        x = np.linspace(3,10,n).reshape(-1,1)
        b = np.matmul(A,x).reshape(-1,1)
        d, iterations = scgd(A, x0, b, tol=1e-4, maxiter=100)
        
        iter_list.append([n,iterations])
        
    plt.figure()
    plt.plot(np.array(iter_list)[:,0], np.array(iter_list)[:,1]+1)
    plt.xlabel('Size of the diagnol matrix')
    plt.ylabel('Number of iterations to converge')
    plt.savefig('SCGD_rate.jpg')
    
    # rate vs conditioning number
    iter_list = []
    for n in range(5,12):
        arr = np.zeros([100,1]).flatten()
        arr[0] = np.exp(-n)
        arr[-1] = np.exp(n)
        
        # generate a matrix
        A = np.diag(arr)
        
        # call the scgd
        x0 = np.zeros([100,1])
        x = np.linspace(3,10,100).reshape(-1,1)
        b = np.matmul(A,x).reshape(-1,1)
        d, iterations = scgd(A, x0, b, tol=1e-4, maxiter=100)
        
        iter_list.append([n,iterations])   
        
    plt.figure()
    plt.plot(np.array(iter_list)[:,0], np.array(iter_list)[:,1]+1)
    plt.xlabel('Differnce between the conditioning number')
    plt.ylabel('Number of iterations to converge')
    plt.savefig('SCGD_rate_2.jpg')
    
#%%
# Here is the secession for question B
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

#%%
# Here is the sesesion for the question C
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
    
    return u

# define a function for calculate the rror
def error_cal(u_a,u_n,h0,h):
    u_c = cp.deepcopy(u_n) # copy of nest matrix
    for i in range(u_n.shape[0]):
        for j in range(u_n.shape[1]):
            u_c[i][j] = u_a[int(i*h/h0)][int(j*h/h0)]
    # error = np.linalg.norm(u_c-u_n,np.inf)
    error = np.max(np.abs((u_c-u_n)).reshape(1,-1))
    return error

def wave_solver(u):
    f = lambda x : np.exp(-400*(x-0.5)**2) 
    error_list = []
    N = 100
    for k in range(5,8):
        N = 2**k
        dx = 1/(2*N)
        dt = 1/N
        u_new = np.zeros([N+1,N+1,N+1])
        
        for i in range(1,N+1):
            for j in range(1,N+1):
                u_new[i,j,2] = dt*f((i-1)*dx)*f((j-1)*dx);

    
    
        for n in range(3,N+1):
            for i in range(2,N):
                for j in range(2,N):
                    u_new[i,j,n] = (dt**2/dx**2)*(u_new[i+1,j,n-1]+u_new[i,j+1,n-1]+u_new[i-1,j,n-1]+u_new[i,j-1,n-1]-4*u_new[i,j,n-1])-u_new[i,j,n-2]+2*u_new[i,j,n-1]

        error_list.append(np.linalg.norm((u-u_new[::2**(10-k),::2**(10-k),
                                            ::2**(10-k)]).reshape(1,-1,-1),
                                         np.inf))
        return error_list
#%%
if __name__ == '__main__':
    x0 = np.zeros([10,1])
    
    x = np.linspace(3,10,10).reshape(-1,1)
    A = 2*np.eye(10) + np.eye(10,k=1) + np.eye(10,k=-1)

    b = np.matmul(A,x).reshape(-1,1)
    # Problem A(a)
    test,a = sgd(A,x0,b,maxiter=1000)
    print('Testing the Steepest GD')
    print('Testing case: ', np.round(x.flatten(),3))
    print('Compute by the code: ', np.round(test.flatten(),3))
    print('Error : ', np.round((test - x).flatten(),3))
    
    
    # Problem A(b)
    print('Testing Conjugate GD')
    test,a = scgd(A,x0,b,maxiter=1000)
    print('Testing case: ', np.round(x.flatten(),3))
    print('Compute by the code: ', np.round(test.flatten(),3))
    print('Error : ', np.round((test - x).flatten(),3))    
    
    # Problem A(c)
    convergence_rate()
    
    #%%
    # problem B
    # probB()
    
    # problem C
    # u = wave_solution()
    # wave_solver(u)