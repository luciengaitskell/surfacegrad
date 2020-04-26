import numpy as np
import scipy.interpolate as interp

# auxiliary function for mesh generation
def gimme_mesh(n):
    minval = -1
    maxval =  1
    # produce an asymmetric shape in order to catch issues with transpositions
    return np.meshgrid(np.linspace(minval,maxval,n), np.linspace(minval,maxval,n+1))

# set up underlying test functions, vectorized
def fun_smooth(x, y):
    return np.cos(np.pi*x)*np.sin(np.pi*y)

def fun_evil(x, y):
    # watch out for singular origin; function has no unique limit there
    return np.where(x**2+y**2>1e-10, x*y/(x**2+y**2), 0.5)

# sparse input mesh, 6x7 in shape
N_sparse = 6
x_sparse,y_sparse = gimme_mesh(N_sparse)
z_sparse_smooth = fun_smooth(x_sparse, y_sparse)
z_sparse_evil = fun_evil(x_sparse, y_sparse)

# scattered input points, 10^2 altogether (shape (100,))
N_scattered = 10
x_scattered,y_scattered = np.random.rand(2,N_scattered**2)*2 - 1
z_scattered_smooth = fun_smooth(x_scattered, y_scattered)
z_scattered_evil = fun_evil(x_scattered, y_scattered)

# dense output mesh, 20x21 in shape
N_dense = 20
x_dense,y_dense = gimme_mesh(N_dense)

z_dense_smooth_griddata = interp.griddata(np.array([x_sparse.ravel(),y_sparse.ravel()]).T,
                                          z_sparse_smooth.ravel(),
                                          (x_dense,y_dense), method='cubic')   # default method is linear

print("done")
