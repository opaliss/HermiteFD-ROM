"""Module to run the adaptive (or non-adaptive) Hermite linear_landau Landau damping full-order model (FOM) testcase

Author: Opal Issan
Date: Dec 22nd, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.FOM import nonlinear_full, charge_density
from operators.implicit_midpoint_FOM import implicit_midpoint_solver_FOM
from operators.setup_FOM import SimulationSetupFOM
from operators.poisson_solver import fft_solver, gmres_solver, fft_solver_Ax_b
import time
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def rhs(y):
    # electric field computed
    rho = charge_density(alpha_e=setup.alpha_e,
                         alpha_i=setup.alpha_i,
                         q_e=setup.q_e,
                         q_i=setup.q_i,
                         C0_e=y[:setup.Nx],
                         C0_i=C0_ions)

    E = gmres_solver(rhs=rho, D=setup.D)
    # print("residual = ", np.mean(np.abs(setup.D @ E - rho)))

    # evolving only electrons
    return setup.A_e @ y + nonlinear_full(E=E,
                                          psi=y,
                                          Nv=setup.Nv,
                                          Nx=setup.Nx,
                                          alpha=setup.alpha_e,
                                          q=setup.q_e,
                                          m=setup.m_e)


if __name__ == "__main__":
    for k_ in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        setup = SimulationSetupFOM(Nx=300,
                                   Nv=100,
                                   epsilon=1e-2,
                                   alpha_e=np.sqrt(2),
                                   alpha_i=np.sqrt(2 / 1836),
                                   u_e=0,
                                   u_i=0,
                                   L=20 * np.pi,
                                   dt=1e-2,
                                   T0=0,
                                   T=20,
                                   nu=10)

        # initial condition: read in result from previous simulation
        y0 = np.zeros(setup.Nv * setup.Nx)
        # first electron 1 species (perturbed)
        y0[:setup.Nx] = (1 + setup.epsilon * np.cos(k_ * np.linspace(0, setup.L, setup.Nx, endpoint=False))) / setup.alpha_e

        # ions (unperturbed)
        C0_ions = np.ones(setup.Nx) / setup.alpha_i

        # start timer
        start_time_cpu = time.process_time()
        start_time_wall = time.time()

        # integrate (implicit midpoint)
        sol_midpoint_u, setup = implicit_midpoint_solver_FOM(y_0=y0,
                                                             right_hand_side=rhs,
                                                             a_tol=1e-8,
                                                             r_tol=1e-4,
                                                             max_iter=100,
                                                             param=setup)

        end_time_cpu = time.process_time() - start_time_cpu
        end_time_wall = time.time() - start_time_wall

        print("runtime cpu = ", end_time_cpu)
        print("runtime wall = ", end_time_wall)

        # make directory
        if not os.path.exists("../data/FOM/linear_landau/sample_" + str(k_)):
            os.makedirs("../data/FOM/linear_landau/sample_" + str(k_))

        # save the runtime
        np.save("../data/FOM/linear_landau/sample_" + str(k_) + "/sol_FOM_u_" + str(setup.Nv) + "_k_" + str(k_) + "_runtime_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

        # save results
        np.save("../data/FOM/linear_landau/sample_" + str(k_) + "/sol_FOM_u_" + str(setup.Nv) + "_k_" + str(k_) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

        np.save("../data/FOM/linear_landau/sample_" + str(k_) + "/sol_FOM_t_" + str(setup.Nv) + "_k_" + str(k_) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

        # save parameters
        # np.save("../data/FOM/linear_landau/sample_" + str(k_) + "/sol_FOM_setup_" + str(setup.Nv) + "_k_" + str(k_) + "_" + str(setup.T0) + "_" + str(setup.T), setup)
