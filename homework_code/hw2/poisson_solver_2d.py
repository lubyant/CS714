"""Solver for 	Lap u(x) = f(x) for x in Omega = [0,1]^2
			   	    u(x) = 0  on partial Omega
	Using Finite difference discretization
			   	"""

# importing the main libraries
import numpy as np
import scipy.sparse as spsp

# importing sparse solver
from scipy.sparse.linalg import spsolve

# importing the ploting libray
import matplotlib.pyplot as plt

# number of discretization points
N = 100
# grid spacing
h = 1./(N+1)

# right-hand side
def f(x, y):
	# this is the constant function, we write it 
	# this way to use vectorization
	return -2. + 0*x + 0*y

# interior grid
x_i = np.linspace(h, 1-h, N)
y_i = np.linspace(h, 1-h, N)

X_i, Y_i = np.meshgrid(x_i, y_i)

##  creating laplace stencil 1/h^2 [1 -2 1] csr format
# first we prepare the diagonals and the indices
ones = np.ones(N)
data = 1/(h**2)*np.array([ones,-2*ones,ones])
diags = np.array([-1, 0, 1])

# here we create the Laplacian as a sparse matrix
A_h = spsp.spdiags(data, diags, N,N, format="csr")

# we build the 2D laplacian using the kronecker product
Lap_h = spsp.kron(A_h, spsp.eye(N)) + spsp.kron(spsp.eye(N), A_h)

# checking that the Laplacian has the correct sparsity pattern
plt.figure()
plt.spy(Lap_h)
plt.show()
# if you want to print the matrix you can fo 
# A_h.toarray() to obtain a dense versio of the matrix

# creating the right-hand side
f_h = f(X_i, Y_i)
# we are considering homogeneous Dirichlet boundary conditions

# solve the system (it will use an sparse LU factorizatio)
U_h = spsolve(Lap_h,f_h.reshape((-1,)))

# plot the solution
plt.figure()
plt.imshow(U_h.reshape((N,N)))
plt.xlabel("x")
plt.ylabel("y",rotation="horizontal",labelpad=20)
plt.colorbar()
plt.show()


