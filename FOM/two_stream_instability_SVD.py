"""Module to compute the bump-on-tail instability full-order model (FOM) SVD

Author: Opal Issan
Date: Dec 27th, 2024
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.setup_FOM_two_stream import SimulationSetupTwoStreamFOM

setup = SimulationSetupTwoStreamFOM(Nx=251,
                                    Nv=350,
                                    epsilon=0.1,
                                    alpha_e1=0.5,
                                    alpha_e2=0.5,
                                    alpha_i=np.sqrt(2 / 1836),
                                    u_e1=-1.05,
                                    u_e2=1.05,
                                    u_i=0,
                                    L=2 * np.pi,
                                    dt=1e-2,
                                    T0=0,
                                    T=40,
                                    nu_e1=15,
                                    nu_e2=15,
                                    n0_e1=0.5,
                                    n0_e2=0.5,
                                    construct_B=False)

M = 3
u_train = [1.05, 1.06, 1.07, 1.08, 1.09]

sol_midpoint_u = np.load("../data/FOM/two_stream/sample_" + str(u_train[0]) + "/sol_FOM_u_" + str(setup.Nv) + "_nu_" + str(setup.nu_e1) + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy")

for kk in range(1, len(u_train)):
    print(kk)
    # update the standard deviation parameter
    new_data = np.load("../data/FOM/two_stream/sample_" + str(u_train[kk]) + "/sol_FOM_u_" + str(setup.Nv) + "_nu_" + str(setup.nu_e1) + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy")
    sol_midpoint_u = np.append(sol_midpoint_u, new_data, axis=1)


sol_midpoint_e1 = sol_midpoint_u[M*setup.Nx:setup.Nx*setup.Nv, :]
sol_midpoint_e2 = sol_midpoint_u[setup.Nv*setup.Nx+M*setup.Nx:, :]

# compute the SVD
# electrons species (1)
U_e1, S_e1, _ = np.linalg.svd(sol_midpoint_e1, full_matrices=False)
print("e1 SVD computed!")

# electrons species (2)
U_e2, S_e2, _ = np.linalg.svd(sol_midpoint_e2, full_matrices=False)
print("e2 SVD computed!")

# basis save first 1000 modes (we dont really use more than that)
np.save("../data/ROM/two_stream/basis_SVD_e1_" + str(setup.T0) + "_" + str(setup.T) + "_M_" + str(M) + ".npy", U_e1[:, :1000])
np.save("../data/ROM/two_stream/basis_SVD_e2_" + str(setup.T0) + "_" + str(setup.T) + "_M_" + str(M) + ".npy", U_e2[:, :1000])

# singular values
np.save("../data/ROM/two_stream/singular_values_SVD_e1_" + str(setup.T0) + "_" + str(setup.T) + "_M_" + str(M) + ".npy", S_e1)
np.save("../data/ROM/two_stream/singular_values_SVD_e2_" + str(setup.T0) + "_" + str(setup.T) + "_M_" + str(M) + ".npy", S_e2)

