import numpy as np

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x):
    theta = np.full((x.shape[0], ddd.dof), 1.)
    index_map = index_mapping()

    for index in index_map:
        for d in range(ddd.dim):
            theta[:, index_map[index]] *= np.power(x[:, d], index[d])

    return theta

def H(x, degree):
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
            theta[:, index_map[index]] *= np.H(x[:, d], index[d])

    return theta

def h2o_simple_transformation():
    mat = np.zeros((ddd.terms, ddd.terms))
    mat[0, 0] = 0.63161877774606470129
    mat[1, 1] = 0.63161877774606470129
    mat[2, 2] = 0.44662192086900116570
    mat[0, 2] = -mat[2, 2]
    mat[3, 3] = 0.25785728623970555997
    mat[1, 3] = -3 * mat[3, 3]

    return mat

# for the dim-dimensional, terms-hermite terms case, creating the transformation matrix for
# any index mapping provided
def h2o_transformation_matrix():
    transformation = np.full((ddd.dof, ddd.dof), 1.)
    index_map = index_mapping()
    mat = h2o_simple_transformation()

    for row_index in index_map:
        for col_index in index_map:
            for d in range(ddd.dim):
                transformation[index_map[row_index], index_map[col_index]] *= mat[row_index[d], col_index[d]]
                
    return transformation

def hermite_to_ordinary(theta):
    transformation = h2o_transformation_matrix() 
    ordinary_theta = np.matmul(transformation, theta)
    return ordinary_theta

def ordinary_to_hermite(theta):
    transformation = np.linalg.inv(h2o_transformation_matrix())
    hermite_theta = np.matmul(transformation, theta)
    return hermite_theta

def compute_error(estimated, true, errors_computed):
    errors = []

    # regression metric
    # L1 norm
    if (errors_computed[0]):
        errors.append(np.sum(np.abs(true - estimated)))

    # L2 norm
    if (errors_computed[1]):
        errors.append(np.sqrt(np.sum(np.power(true - estimated, 2))))

    # classification metric, P = value is zero, N = value is non-zero
    # true positive => true was zero and estimated was zero
    TP = np.sum(np.logical_and(true == 0., estimated == 0.))
    # true negative => true was non-zero and estimated was non-zero
    TN = np.sum(np.logical_and(true != 0., estimated != 0.))
    # false positive => true was non-zero and estimated was zero
    FP = np.sum(np.logical_and(true != 0., estimated == 0.))
    # false negative => true was zero and estimated was non-zero
    FN = np.sum(np.logical_and(true == 0., estimated != 0.))

    # precision = true positives / total estimated positives {TP / (TP + FP)}
    if (errors_computed[2]):
        errors.append(TP / (TP + FP))

    # recall = true positives / total true positives {TP / (TP + FN)}
    if (errors_computed[3]):
        errors.append(TP / (TP + FN))
        
    # accuracy = total true / total predictions {(TP + TN) / (TP + TN + FP + FN)}
    if (errors_computed[4]):
        errors.append((TP + TN) / (TP + TN + FP + FN))