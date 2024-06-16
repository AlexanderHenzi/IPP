# -*- coding: utf-8 -*-

import numpy as np 

from qosa import QoseIndices
from qosa.tests import AdditiveGaussian, ExponentialGaussian, Ishigami


# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Additive Gaussian model
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

# ###################################
# # Dimension 2, Independent inputs #
# ###################################
# print("Additive Gaussian model, dim 2, idt inputs \n") 

# alpha = np.array([0.1, 0.5, 0.75, 0.95])
# n_alphas = alpha.shape[0]
# dim = 2
# means = [0, 0]
# std = [1, 2]
# beta = [1, 1]

# # Theoretical indices
# model = AdditiveGaussian(dim=dim, means=means, std=std, beta=beta)
# model.alpha = alpha
# _, _, idt_shapley_qosa_indices = model.qosa_indices
# print(idt_shapley_qosa_indices, "\n")

# # Estimated indices
# n_upsilon = 10**3
# estimation_method = "exact"
# n_perms = None
# n_outer = 10**2
# n_inner = 10**2
# n_boot = 1

# qose = QoseIndices(model.input_distribution)
# qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
# qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
# print(qose_results.qose_indices.T, "\n")

# #################################
# # Dimension 2, Dependent inputs #
# #################################
# print("Additive Gaussian model, dim 2, dpt inputs \n") 

# alpha = np.array([0.1, 0.5, 0.75, 0.95])
# n_alphas = alpha.shape[0]
# dim = 2
# means = [0, 0]
# std = [1, 2]
# beta = [1, 1]
# correlation = [0.7]
    
# # Theoretical indices
# model = AdditiveGaussian(dim=dim, means=means, std=std, beta=beta)
# model.copula_parameters = correlation
# model.alpha = alpha
# _, _, idt_shapley_qosa_indices = model.qosa_indices
# print(idt_shapley_qosa_indices, "\n")

# # Estimated indices
# n_upsilon = 10**3
# estimation_method = "exact"
# n_perms = None
# n_outer = 10**2
# n_inner = 10**2
# n_boot = 1

# qose = QoseIndices(model.input_distribution)
# qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
# qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
# print(qose_results.qose_indices.T, "\n")

# ###################################
# # Dimension 3, Independent inputs #
# ###################################
# print("Additive Gaussian model, dim 3, idt inputs \n") 

# alpha = np.array([0.1, 0.5, 0.75, 0.95])
# n_alphas = alpha.shape[0]
# dim = 3
# means = [0, 0, 0]
# std = [1, 2, 3]
# beta = [1, 1, 1]

# # Theoretical indices
# model = AdditiveGaussian(dim=dim, means=means, std=std, beta=beta)
# model.alpha = alpha
# _, _, idt_shapley_qosa_indices = model.qosa_indices
# print(idt_shapley_qosa_indices, "\n")

# # Estimated indices
# n_upsilon = 10**3
# estimation_method = "exact"
# n_perms = None
# n_outer = 10**2
# n_inner = 10**2
# n_boot = 1

# qose = QoseIndices(model.input_distribution)
# qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
# qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
# print(qose_results.qose_indices.T, "\n")

# #################################
# # Dimension 3, Dependent inputs #
# #################################
# print("Additive Gaussian model, dim 3, dpt inputs \n") 

# alpha = np.array([0.1, 0.5, 0.75, 0.95])
# n_alphas = alpha.shape[0]
# dim = 3
# means = [0, 0, 0]
# std = [1, 2, 3]
# beta = [1, 1, 1]
# correlation = [0, 0, 0.7]
    
# # Theoretical indices
# model = AdditiveGaussian(dim=dim, means=means, std=std, beta=beta)
# model.copula_parameters = correlation
# model.alpha = alpha
# _, _, idt_shapley_qosa_indices = model.qosa_indices
# print(idt_shapley_qosa_indices, "\n")

# # Estimated indices
# n_upsilon = 10**3
# estimation_method = "exact"
# n_perms = None
# n_outer = 10**2
# n_inner = 10**2
# n_boot = 1

# qose = QoseIndices(model.input_distribution)
# qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
# qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
# print(qose_results.qose_indices.T, "\n")

# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Exponential Gaussian model
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

###################################
# Dimension 2, Independent inputs #
###################################
# print("Exponential Gaussian model, dim 2, idt inputs \n") 

# alpha = np.array([0.2, 0.4, 0.6, 0.8])
# n_alphas = alpha.shape[0]
# dim = 2
# means = [0, 0]
# std = [1, 2]
# beta = [1, 1]

# # Theoretical indices
# model = ExponentialGaussian(dim=dim, means=means, std=std, beta=beta)
# model.alpha = alpha
# _, _, idt_shapley_qosa_indices = model.qosa_indices
# print(idt_shapley_qosa_indices, "\n")

# # Estimated indices
# n_upsilon = 10**5
# estimation_method = "exact"
# n_perms = None
# n_outer = 10**4
# n_inner = 10**1
# n_boot = 1

# qose = QoseIndices(model.input_distribution)
# qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
# qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
# print(qose_results.qose_indices.T, "\n")

#################################
# Dimension 2, Dependent inputs #
#################################
print("Exponential Gaussian model, dim 2, dpt inputs \n") 

alpha = np.array([0.2, 0.4, 0.6, 0.8])
n_alphas = alpha.shape[0]
dim = 2
means = [0, 0]
std = [1, 2]
beta = [1, 1]
correlation = [0.75]
    
# Theoretical indices
model = ExponentialGaussian(dim=dim, means=means, std=std, beta=beta)
model.copula_parameters = correlation
model.alpha = alpha
_, _, idt_shapley_qosa_indices = model.qosa_indices
print(idt_shapley_qosa_indices, "\n")

# Estimated indices
n_upsilon = 10**5
estimation_method = "exact"
n_perms = None
n_outer = 6*10**4
n_inner = 6*10**4
n_boot = 1

qose = QoseIndices(model.input_distribution)
qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
print(qose_results.qose_indices.T, "\n")

# # ###################################
# # # Dimension 3, Independent inputs #
# # ###################################
# print("Exponential Gaussian model, dim 3, idt inputs \n") 

# alpha = np.array([0.2, 0.4, 0.6, 0.8])
# n_alphas = alpha.shape[0]
# dim = 3
# means = [0, 0, 0]
# std = [1, 2, 3]
# beta = [1, 1, 1]

# # Theoretical indices
# model = ExponentialGaussian(dim=dim, means=means, std=std, beta=beta)
# model.alpha = alpha
# _, _, idt_shapley_qosa_indices = model.qosa_indices
# print(idt_shapley_qosa_indices, "\n")

# # Estimated indices
# n_upsilon = 10**5
# estimation_method = "exact"
# n_perms = None
# n_outer = 5*10**4
# n_inner = 5*10**4
# n_boot = 1

# qose = QoseIndices(model.input_distribution)
# qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
# qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
# print(qose_results.qose_indices.T, "\n")

# #################################
# # Dimension 3, Dependent inputs #
# #################################
# print("Exponential Gaussian model, dim 3, dpt inputs \n") 

# alpha = np.array([0.2, 0.4, 0.6, 0.8])
# n_alphas = alpha.shape[0]
# dim = 3
# means = [0, 0, 0]
# std = [1, 2, 3]
# beta = [1, 1, 1]
# correlation = [0, 0, 0.75]
    
# # Theoretical indices
# model = ExponentialGaussian(dim=dim, means=means, std=std, beta=beta)
# model.copula_parameters = correlation
# model.alpha = alpha
# _, _, idt_shapley_qosa_indices = model.qosa_indices
# print(idt_shapley_qosa_indices, "\n")

# # Estimated indices
# n_upsilon = 10**5
# estimation_method = "exact"
# n_perms = None
# n_outer = 4*10**4
# n_inner = 10**1
# n_boot = 1

# qose = QoseIndices(model.input_distribution)
# qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
# qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
# print(qose_results.qose_indices.T, "\n")


# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Ishigami model
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

alpha = np.array([0.2, 0.4, 0.6, 0.8])
alpha = np.linspace(start=0.1, stop=0.9, num=9)
n_alphas = alpha.shape[0]

model = Ishigami()

# Estimated indices
n_upsilon = 10**3
estimation_method = "exact"
n_perms = None
n_outer = 10**3
n_inner = 10**2
n_boot = 1

qose = QoseIndices(model.input_distribution)
qose.build_sample(model=model, n_upsilon=n_upsilon, n_perms=n_perms, n_outer=n_outer, n_inner=n_inner)
qose_results = qose.compute_indices(alpha=alpha, n_boot=n_boot)
print(qose_results.qose_indices.T, "\n")
