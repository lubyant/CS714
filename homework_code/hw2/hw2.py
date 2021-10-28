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
from scipy.sparse.linalg import spsolve
import numpy.linalg as nplng
#%%             Here is the secion for question A
# 
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

# question A(d)
def condition_number():
    # condtion number = 100
    arr1 = np.linspace(0.1,10,100).flatten() # uniform distribution
    arr2 = np.zeros([100,1]).flatten() # cluster 
    arr2[:10] = 0.1
    arr2[30:60] = 1
    arr2[90:99] = 10
    
    # generate two matrices for two conditions sereis
    A1 = np.diag(arr1)
    A2 = np.diag(arr2)
    
    # call the scgd
    x0 = np.zeros([100,1])
    x = np.linspace(3,10,100).reshape(-1,1)
    b1 = np.matmul(A1,x).reshape(-1,1)
    d1, iterations1 = scgd(A1, x0, b1, tol=1e-4, maxiter=100)
    
    b2 = np.matmul(A2,x).reshape(-1,1)
    d2, iterations2 = scgd(A2, x0, b2, tol=1e-4, maxiter=100)
    
    print(iterations1)
    print(iterations2)
    
# question A(e)
def sp_solve():
    # function for x
    f = lambda x: -np.exp(-(x-0.5)**2/(2*0.04**2))
    
    N = 100
    h = 1/(N-1)
    # construct the matrix D
    dig1 = -2 * np.ones([1,N]).flatten()
    dig2 = np.ones([1,N-1]).flatten()
    matrix_D = np.diag(dig1) + np.diag(dig2, k = 1) + np.diag(dig2, k=-1)
    matrix_D[0,0] = h**2
    matrix_D[N-1,N-1] = h**2
    matrix_D[0,1] = 0
    matrix_D[N-1,N-2] = 0
    
    
    # construct the vector b
    b = np.zeros([N,1])
    for i in range(len(b)):
        b[i] = f(i*h)
    b[0] = 0
    b[-1] = 0
    
    # construct 3d matrix by kronecker product
    I = np.eye(N)
    A = np.kron(np.kron(matrix_D,I),I)
    + np.kron(np.kron(I,matrix_D),I)
    + np.kron(np.kron(I,I),matrix_D)
    A = A/h**2
    
    B = np.kron(np.kron(b,b),b)
    
    x = spsolve(A, B)
    sol = x.reshape(N,N,N)
    
    # plotting 3d
    # x0 = np.linspace(0,1,N)
    # y0 = np.linspace(0,1,N)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # x,y = np.meshgrid(x0,y0)
    # ax.plot_surface(x, y, sol[0,:,:], color='green')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('u')
    # plt.title('Profile of FD solution')
    
    # plot the projection
    plt.figure()
    plt.imshow(sol[0,:,:])
    plt.xlabel("x")
    plt.ylabel("y",rotation="horizontal",labelpad=20)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.imshow(sol[9,:,:])
    plt.xlabel("x")
    plt.ylabel("y",rotation="horizontal",labelpad=20)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.imshow(sol[18,:,:])
    plt.xlabel("x")
    plt.ylabel("y",rotation="horizontal",labelpad=20)
    plt.colorbar()
    plt.show()
    return x

# implement a matrix free method by Gauss-Sedel    
def mf_solve():
    f = lambda x: -np.exp(-(x-0.5)**2/(2*0.04**2))
    
    N = 100
    h = 1/(N-1) 
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    zz = np.linspace(0,1,N)
    
    u = np.zeros([N,N,N])
    u_0 = np.ones([N,N,N])
    
    error = 1
    while error>1/(1/h+1)**2:
        u_0 = cp.deepcopy(u)
        for z in range(1,u.shape[2]-1):
            for j in range(1,u.shape[1]-1):
                for i in range(1,u.shape[0]-1):
                    u[i,j,z] = (u[i+1,j,z] + u[i-1,j,z] +u[i,j+1,z] + u[i,j-1,z] + u[i,j,z+1] + u[i,j,z-1] - h**2*f(x[i])*f(y[j])*f(zz[z]))/8
        error = nplng.norm(u - u_0)
        print(error)
    
    sol = u
    plt.figure()
    plt.imshow(sol[:,:,0])
    plt.xlabel("x")
    plt.ylabel("y",rotation="horizontal",labelpad=20)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.imshow(sol[:,:,50])
    plt.xlabel("x")
    plt.ylabel("y",rotation="horizontal",labelpad=20)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.imshow(sol[:,:,N-1])
    plt.xlabel("x")
    plt.ylabel("y",rotation="horizontal",labelpad=20)
    plt.colorbar()
    plt.show()
    
# implement a method for SCGD
def cg_solve():
    # function for x
    f = lambda x: -np.exp(-(x-0.5)**2/(2*0.04**2))
    
    N = 30
    h = 1/(N-1)
    # construct the matrix D
    dig1 = -2 * np.ones([1,N]).flatten()
    dig2 = np.ones([1,N-1]).flatten()
    matrix_D = np.diag(dig1) + np.diag(dig2, k = 1) + np.diag(dig2, k=-1)
    matrix_D[0,0] = h**2
    matrix_D[N-1,N-1] = h**2
    matrix_D[0,1] = 0
    matrix_D[N-1,N-2] = 0
    
    
    # construct the vector b
    b = np.zeros([N,1])
    for i in range(len(b)):
        b[i] = f(i*h)
    b[0] = 0
    b[-1] = 0
    
    # construct 3d matrix by kronecker product
    I = np.eye(N)
    A = np.kron(np.kron(matrix_D,I),I)
    + np.kron(np.kron(I,matrix_D),I)
    + np.kron(np.kron(I,I),matrix_D)
    A = A/h**2
    
    B = np.kron(np.kron(b,b),b)
    
    # call the SCGD I define
    x0 = np.zeros([N**3,1])
    test,a = scgd(A,x0,B,maxiter=1000)
    print(test)
    print(a)
    
#%%            Here is the secession for question B 
# 
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

#%%         Here is the sesesion for the question C
# 
def wave_solution():
    f = lambda x : np.exp(-400*(x-0.5)**2) 
    
    N = 256 # use the solution from problem b
    
    dt = 1/(2*N) 
    dx = 1/N
    
    u = np.zeros([N+1,N+1,N+1])
    
    # initial condition for first derivative of time
    for i in range(N+1):
        for j in range(N+1):
            u[i,j,1] = dt*f((i)*dx)*f((j)*dx)
    
    # pointwise linear function:
    for t in range(2,N+1):
        for i in range(1,N):
            for j in range(1,N):
                u[i,j,t] = (dt**2/dx**2)*(u[i+1,j,t-1]+u[i,j+1,t-1]+u[i-1,j,t-1]+u[i,j-1,t-1]-4*u[i,j,t-1])-u[i,j,t-2]+2*u[i,j,t-1]
    
    return u

# compute the wave equaiton in really fine grid
def wave_solver(u):
    f = lambda x : np.exp(-400*(x-0.5)**2) 
    error_list = []
    for k in range(4,7):
        N = 2**k
        dt = 1/(2*N)
        dx = 1/N
        u_new = np.zeros([N+1,N+1,N+1])
        
        for i in range(N+1):
            for j in range(N+1):
                u_new[i,j,1] = dt*f((i)*dx)*f((j)*dx);

    
    
        for n in range(2,N+1):
            for i in range(1,N):
                for j in range(1,N):
                    u_new[i,j,n] = (dt**2/dx**2)*(u_new[i+1,j,n-1]+u_new[i,j+1,n-1]+u_new[i-1,j,n-1]+u_new[i,j-1,n-1]-4*u_new[i,j,n-1])-u_new[i,j,n-2]+2*u_new[i,j,n-1]
                    
        m = u_new - u[::2**(8-k),::2**(8-k),::2**(8-k)]
        m = m.reshape([1,1,-1])
        error = np.max(m)
        error_list.append(error)
    plt.figure()
    plt.plot(np.array([4,5,6]),-np.log2(error_list),label='error plot')
    plt.plot(np.array([4,5,6]),np.array([8,10,12]),label='slope 2 reference')
    plt.xlabel('number of grid (2^x)')
    plt.ylabel('-log(norm)')
    plt.savefig('wave_sol.jpg')
    plt.legend()
#%%   main function for grader :)
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
    
    # Problem A(d)
    condition_number()
    
    # problem A(e)
    x = sp_solve() # part a
    mf_solve() # part b
    x = cg_solve() # part c
    
    #%%
    # problem B
    probB()
    
    # problem C
    u = wave_solution()
    error = wave_solver(u)
