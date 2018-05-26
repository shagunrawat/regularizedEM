import numpy as np
import bridgeEM

# initializing dimensions of the system and the highest number of hermite terms 
# (degree of hermite polynomial = number of hermite terms - 1)
ddd(terms = 4, dim = 1)

def system_drift(sim_param, x):
    derivatives = np.zeros((x.shape[0], x.shape[1]))
    derivatives[:, 0] = sim_param.alpha + sim_param.beta * x[:, 0] + sim_param.gamma * np.power(x[:, 0], 2)
    return derivatives 

def true_theta(sim_param):
    theta = np.zeros((ddd.dof, ddd.dim))
    theta[0, 0] = sim_param.alpha
    theta[1, 0] = sim_param.beta
    theta[2, 0] = sim_param.gamma
    return theta

def system_diffusion(sim_param):
    return np.dot(sim_param.gvec, np.random.standard_normal(ddd.dim))

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