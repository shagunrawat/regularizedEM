import numpy as np

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x):
    theta = np.full((x.shape[0], ddd.dof), 1.)
    index_map = index_mapping()

    for index in index_map:
        for d in range(ddd.dim):
            theta[:, index_map[index]] *= np.power(x[:, d], index[d])

    return theta

def H(degree, x):
    switcher = {
        0: 0.63161877774606470129,
        1: 0.63161877774606470129 * x,
        2: 0.44662192086900116570 * (np.power(x, 2) - 1),
        3: 0.25785728623970555997 * (np.power(x, 3) - 3 * x),
        4: 0.12892864311985277998 * (np.power(x, 4) - 6 * np.power(x, 2) + 3),
    }
    return switcher.get(degree, "Polynomial degree exceeded")

# this function defines our hermite basis functions
# x must be a numpy array, a column vector of points
# (x = vector of points at which we seek to evaluate the basis functions)
# dof is the number of degrees of freedom, i.e., the number of basis functions.
def hermite_basis(x):
    theta = np.full((x.shape[0], ddd.dof), 1.)
    index_map = index_mapping()

    for index in index_map:
        for d in range(ddd.dim):
            theta[:, index_map[index]] *= H(index[d], x[:, d])

    return theta

def h2o_simple_transformation():
    mat = np.zeros((dof.degree, dof.degree))
    mat[0, 0] = 0.63161877774606470129
    mat[1, 1] = 0.63161877774606470129
    mat[2, 2] = 0.44662192086900116570
    mat[0, 2] = -mat[2, 2]
    mat[3, 3] = 0.25785728623970555997
    mat[1, 3] = -3 * mat[3, 3]

    return mat

def h2o_transformation_matrix():
    transformation = np.zeros((ddd.dof, ddd.dof))
    index_map = index_mapping()
    index = 0

    mat = h2o_simple_transformation()

    for index in index_map:


    return transformation

def hermite_to_ordinary(theta):
    transformation = h2o_transformation_matrix() 
    ordinary_theta = np.matmul(transformation, theta)
    return ordinary_theta

def ordinary_to_hermite(theta):
    transformation = np.linalg.inv(h2o_transformation_matrix())
    hermite_theta = np.matmul(transformation, theta)
    return hermite_theta

def compute_error(estimated, true, type = 'L2norm'):
    if (type = 'L2norm'):
        return (np.sqrt(np.sum(np.power(np.abs(true - estimated), 2))))

    if (type = 'precision'):

    if (type = 'recall'):
        