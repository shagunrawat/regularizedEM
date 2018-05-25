import numpy as np

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x):
    theta = np.full((x.shape[0], dof.dof), 1.)
    index_map = index_mapping()

    for index in index_map:
        for d in range(dof.dim):
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

# TODO : currently this loop has to be written separately for varying dimensions.
def hermite_basis(x):
    theta = np.full((x.shape[0], dof.dof), 1.)
    index_map = index_mapping()

    for index in index_map:
        for d in range(dof.dim):
            theta[:, index_map[index]] *= H(index[d], x[:, d])

    return theta

def index_mapping():
    index = 0
    index_map = {}

    for d in range(0, dof.degree):
        for i in range(0, d + 1):
            if (i == d):
                index_set = (i)
                index_map[index_set] = index
                index += 1

    return index_map

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
    transformation = np.zeros((dof.dof, dof.dof))
    index_map = index_mapping()
    index = 0

    mat = h2o_simple_transformation()

    for d in range(0, dof.degree):
        for i in range(0, d + 1):
            if (i == d):
                transformation[index, index] = mat[i, i]
                if (i >= 2):
                    new_index_set = (i - 2)
                    new_index = index_map[new_index_set]
                    transformation[new_index, index] = mat[i - 2, i]

                index += 1

    return transformation

def hermite_to_ordinary(theta):
    transformation = h2o_transformation_matrix() 
    ordinary_theta = np.matmul(transformation, theta)
    return ordinary_theta

def ordinary_to_hermite(theta):
    transformation = np.linalg.inv(h2o_transformation_matrix())
    hermite_theta = np.matmul(transformation, theta)
    return hermite_theta

def norm_error(true_theta, estimated_theta):
    errors = []
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.ordinary - estimated_theta.ordinary), 2))))
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.hermite - estimated_theta.hermite), 2))))
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.sparse_ordinary - estimated_theta.sparse_ordinary), 2))))
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.sparse_hermite - estimated_theta.sparse_hermite), 2))))
    return errors

def theta_sparsity(theta):
    threshold = 0.1 * np.max(np.abs(theta))
    theta[np.abs(theta) < threshold] = 0.
    return theta

def compute_error(estimated, true, type = 'L2norm'):
    if (type = 'L2norm'):
        return (np.sqrt(np.sum(np.power(np.abs(true_theta.ordinary - estimated_theta.ordinary), 2))))

    if (type = 'precision'):

    if (type = 'recall'):
        