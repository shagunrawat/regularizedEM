import numpy as np
import pickle
import os

# for running the code using a job array on cluster
parvalue = int(os.environ['SGE_TASK_ID'])

# load data, noise_2 is data with noise = 0.05
with open('./data/noise_2.pkl','rb') as f:
    allx, allt, x_without_noise, euler_param, sim_param = pickle.load(f)

# picking 10 timeseries and the coarseness of the observed data
x = allx[:, 0::20, :] # picking every 20th term to get a total of 51 time points
t = allt[:, 0::20] # 51 time points, coarse data

data_param = data_parameters(theta = 0.5 * np.random.rand(ddd.dof, ddd.dim), gvec = sim_param.gvec)

print("Data shape:", x.shape)
print("Theta shape:", data_param.theta.shape)
print("Theta:", data_param.theta)

# parvalue number of sub intervals
# Note : numsubinterval = 1 => only observed data points, no intermediate brownian bridges
em_param = exp_max_parameters(tol = 1e-2, burninpaths = 10, mcmcpaths = 100, numsubintervals = parvalue, niter = 100)

# call to EM which returns the final error and estimated theta value
error_list, theta_list = exp_max(x, t, em_param, data_param)
error_results = (estimated_theta, true_theta)
estimated_theta = theta_transformations(theta=theta_list[-1], theta_type='hermite')
true_theta = theta_transformations(theta=dc.true_theta(sim_param), theta_type='ordinary')

# save to file
with open('./varying_subintervals/tp_51/subint_' + str(parvalue) + '.pkl','wb') as f:
    pickle.dump([x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param], f)
