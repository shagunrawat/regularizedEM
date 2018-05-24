import numpy as np
import scipy.special

class ddd:
	degree = -1
	dim = -1
	dof = -1
	@classmethod
	def set(ddd, degree, dim):
            ddd.degree = degree
            ddd.dim = dim
            ddd.dof = int(scipy.special.binom(degree + dim - 1, dim))

class em:
	def __init__(self, tol, burninpaths, mcmcpaths, numsubintervals, niter, dt):
		self.tol = tol	# tolerance for error in the theta value
		self.burninpaths = burninpaths 	# burnin paths for mcmc
		self.mcmcpaths = mcmcpaths	# sampled paths for mcmc
		self.numsubintervals = numsubintervals	# number of sub intervals in each interval [x_i, x_{i+1}] for the Brownian bridge
		self.niter = niter	# threshold for number of EM iterations, after which EM returns unsuccessfully
		self.h = dt / numsubintervals	# time step for EM

class data:
	def __init__(self, theta, gvec):
		self.theta = theta
		self.gvec = gvec

class euler_maruyama:
	def __init__(self, numsteps, savesteps, ft, ic, it, numpaths):
		self.numsteps = numsteps
		self.savesteps = savesteps
		self.ft = ft
		self.h = ft / numsteps
		self.ic = ic
		self.it = it
		self.numpaths = numpaths

class system:
	def __init__(self, alpha, beta, gamma, gvec):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.gvec = gvec

class theta_transformations:
	def __init__(self, theta, theta_type=None):
		if theta_type is 'ordinary':
			self.ordinary = theta
			self.hermite = ordinary_to_hermite(theta)
			self.sparse_ordinary = theta_sparsity(self.ordinary)
			self.sparse_hermite = theta_sparsity(self.hermite)
		if theta_type is 'hermite':
			self.ordinary = hermite_to_ordinary(theta)
			self.hermite = theta
			self.sparse_ordinary = theta_sparsity(self.ordinary)
			self.sparse_hermite = theta_sparsity(self.hermite)
