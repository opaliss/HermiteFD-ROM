"""Module to compute the bump-on-tail instability full-order model (FOM) SVD

Author: Opal Issan
Date: Dec 27th, 2024
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np

# set up configuration parameters
# number of mesh points in x
Nx = 351
# number of spectral expansions
Nv = 250
# Artificial collisional frequency
nu = 15
# number of moments to solve for M
M = 3
# delta t
dt = 1e-2
# time span
T0 = 0
T = 40

sol_midpoint_u = np.load("../data/FOM/two_stream/sample_1/sol_FOM_u_" + str(Nv) + "_nu_" + str(nu) + "_" + str(T0) + "_" + str(T) + ".npy")
sol_midpoint_t = np.load("../data/FOM/two_stream/sample_1/sol_FOM_t_" + str(Nv) + "_nu_" + str(nu) + "_" + str(T0) + "_" + str(T) + ".npy")

# compute the SVD
# electrons species (1)
U_e1, S_e1, _ = np.linalg.svd(sol_midpoint_u[M*Nx:Nx*Nv, :], full_matrices=False)
print("e1 SVD computed!")
# electrons species (2)
U_e2, S_e2, _ = np.linalg.svd(sol_midpoint_u[Nv*Nx+M*Nx:, :], full_matrices=False)
print("e2 SVD computed!")

# basis save first 1000 modes (we dont really use more than that)
np.save("../data/ROM/two_stream/SVD/basis_SVD_e1_" + str(T) + "_" + str(T) + "_M_" + str(M) + ".npy", U_e1[:, :1000])
np.save("../data/ROM/two_stream/SVD/basis_SVD_e2_" + str(T) + "_" + str(T) + "_M_" + str(M) + ".npy", U_e2[:, :1000])

# singular values
np.save("../data/ROM/two_stream/SVD/singular_values_SVD_e1_" + str(T0) + "_" + str(T) + "_M_" + str(M) + ".npy", S_e1)
np.save("../data/ROM/two_stream/SVD/singular_values_SVD_e2_" + str(T0) + "_" + str(T) + "_M_" + str(M) + ".npy", S_e2)
