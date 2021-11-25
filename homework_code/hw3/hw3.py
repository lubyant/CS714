# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:37:53 2021

@author: luboy
"""
import numpy as np
import numpy.matlib
from numpy import linalg as LA
import matplotlib.pyplot as plt

# define the Cheb
def cheb(N):
    if N == 0:
        D = 0
        x = 1
        return D,x
    else:
        x = np.cos(np.pi*np.linspace(0,N,N+1)/N).reshape(-1,1)
        c = np.ones([N+1,1])
        c[0] = 2
        c[-1] = 2
        for i in range(len(c)):
            c[i] = c[i]* (-1)**i
        X = np.matlib.repmat(x,1,N+1)
        DX = X-X.T
        D = (np.dot(c,(1/c).T))/(DX+np.eye(N+1))
        D = D- np.diag(np.sum(D,axis=1))
        return D,x

# define a cheb solver
def cheb_solver(k):
    
    N = 2**k
    delta_t = 4/N**2
    timepoint = int(N**2/128)
    # print(timepoint)
    U = np.zeros([N+1,N+1,timepoint+1])
    
    D,x = cheb(N)
    I = np.eye(N-1)
    D2 = D**2
    D2 = D2[1:N,1:N]
    L = np.kron(I,D2) + np.kron(D2,I)
    L_square = L**2
    
    f = lambda x : np.exp(-400*x**2) 
    lap_f = lambda x : 640000*x**2*np.exp(-400*x**2) - 800*np.exp(-400*x**2)
    
    Ut = np.zeros([N+1,N+1])
    lap_Ut = np.zeros([N+1,N+1])
    
    for i in range(N+1):
        for j in range(N+1):
            Ut[i,j] = f(x[i])*f(x[j])
            lap_Ut[i,j] = lap_f(x[i])*f(x[j])+f(x[i])*lap_f(x[j])
            
    U[:,:,1] = delta_t*(Ut+delta_t**2/6*lap_Ut)
    
    n_l = np.linspace(2,timepoint,timepoint-1,dtype=int)
    for n in n_l:
        U[1:N,1:N,n] = 2*U[1:N,1:N,n-1]-U[1:N,1:N,n-2]
        +delta_t**2*(L@U[1:N,1:N,n-1].flatten().reshape(-1,1)).reshape(-1,N-1)
        +1/12*delta_t**4*(L_square@U[1:N,1:N,n-1].flatten().reshape(-1,1)).reshape(-1,N-1)
    
    return U

def sol(B):
    k = 7
    N = 2**k
    delta_t = 4/N**2
    deltax = 1/N
    timepoint = int(N**2/256)
    U= np.zeros([N+1,N+1,timepoint+1]) 

    u_x = lambda x,B : np.sin(B*np.pi*x)
   
    for i in range(N+1):
        for j in range(N+1):
            U[i,j,1] = delta_t*u_x((i)*deltax,B)*u_x((j)*deltax,B)
    n_l = np.linspace(2,timepoint,timepoint-1,dtype=int)
    for n in n_l:
        for i in range(1,N):
            for j in range(1,N):
                U[i,j,n] = (delta_t**2/deltax**2)*(U[i+1,j,n-1]+U[i,j+1,n-1]+U[i-1,j,n-1]+U[i,j-1,n-1]-4*U[i,j,n-1])-U[i,j,n-2]+2*U[i,j,n-1]
    return U

# define a new solver for problem e initial condition
def cheb_solver_e(k=6):
    N = 2**k
    delta_t = 3/N**2
    deltax = 1/N
    timepoint = int(N**2/4)
    # print(timepoint)
  

    D,x = cheb(N)
    I = np.eye(N-1)
    D2 = D**2
    D2 = D2[1:N,1:N]
    L = np.kron(I,D2) + np.kron(D2,I)
    L_square = L**2    
    
    B_l = [2,4,8,16,32]
    u_x = lambda x,B : np.sin(B*np.pi*x)
    u_t = lambda x,B : np.sin(np.sqrt(2)*B*np.pi*x)
    
    error_FD = []
    error_Spec = []
    
    for B in B_l:
        U = np.zeros([N+1,N+1,timepoint+1])    
        U_a = np.zeros([N+1,N+1,timepoint+1]) 
        
        # Analytical solution
        for i in range(N+1):
            for j in range(N+1):
                for k in range(timepoint+1):
                    U_a[i,j,k] = u_x((i)*deltax,B)*u_x((j)*deltax,B)*u_t((k)*delta_t,B)/(np.sqrt(2)*B*np.pi)
        U_a_2 = sol(B) 
           
        # FD method
        for i in range(N+1):
            for j in range(N+1):
                U[i,j,1] = delta_t*u_x((i)*deltax,B)*u_x((j)*deltax,B)
        n_l = np.linspace(2,timepoint,timepoint-1,dtype=int)
        for n in n_l:
            for i in range(1,N):
                for j in range(1,N):
                    U[i,j,n] = (delta_t**2/deltax**2)*(U[i+1,j,n-1]+U[i,j+1,n-1]+U[i-1,j,n-1]+U[i,j-1,n-1]-4*U[i,j,n-1])-U[i,j,n-2]+2*U[i,j,n-1]
        error_FD.append(LA.norm((U-U_a).flatten(),np.inf))
        
        # Spectrum method
        Us = np.zeros([N+1,N+1,timepoint+1])
        Ut = np.zeros([N+1,N+1])
        lap_Ut = np.zeros([N+1,N+1])
        for i in range(N+1):
            for j in range(N+1):
                Ut[i,j] = u_x(x[i],B)*u_x(x[j],B)
                lap_Ut[i,j] = -2*B**2*np.pi**2*Ut[i,j] 
        Us[:,:,1] = delta_t*(Ut+delta_t**2/6*lap_Ut)
        for n in n_l:
            Us[1:N,1:N,n] = 2*Us[1:N,1:N,n-1]-Us[1:N,1:N,n-2]
            +delta_t**2*(L@Us[1:N,1:N,n-1].flatten().reshape(-1,1)).reshape(-1,N-1)
            +1/12*delta_t**4*(L_square@Us[1:N,1:N,n-1].flatten().reshape(-1,1)).reshape(-1,N-1)
        
        error_Spec.append(LA.norm((Us-U_a).flatten(),np.inf))
        
       
    return error_FD,error_Spec
    
if __name__ == '__main__':
    # #%% Problem c
    
    # # compute an solution with fine grid
    # k = 7 # fine grid == 2^7
    # U_sol = cheb_solver(k)

    # # compute the cheb sol from different grid size
    # grid_size = np.linspace(4,6,3,dtype=int)
    # error_list = []
    # for k in grid_size:
    #     U_cheb = cheb_solver(k)
    #     diff = U_cheb - U_sol[0::2**(7-k),0::2**(7-k),0::4**(7-k)]
    #     error_list.append(LA.norm(diff.flatten(),np.inf))
        
    # # plot the error 
    # plt.figure()
    # plt.plot(grid_size,-np.log2(error_list),label='Error plot')
    # plt.plot(grid_size,4*grid_size,label='Slope 4 reference')
    # plt.xlabel('Log grid size')
    # plt.ylabel('Log error for infinity norm')
    # plt.legend()
    
    #%% Problem e
        
    error_Spec,error_FD = cheb_solver_e()     
        
    # plot the error
    plt.figure()
    B_l = [2,4,8,16,32]
    plt.plot(B_l,np.log10(error_FD),label='FD')
    plt.plot(B_l,np.log10(error_Spec),label='Spectrum')
    plt.plot(B_l,-3*np.ones([len(B_l),1]))
    plt.legend()
    plt.xlabel('B value')
    plt.ylabel('Log10 error')