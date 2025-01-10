"""Module to run the ion acoustic instability full-order model (FOM) testcase

Author: Opal Issan
Date: Jan 9th, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.FOM import nonlinear_full, charge_density
from operators.implicit_midpoint_FOM import implicit_midpoint_solver_FOM
from operators.setup_FOM import SimulationSetupFOM
from operators.poisson_solver import gmres_solver
import time
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def rhs(y):
    # charge density computed
    rho = charge_density(alpha_e=setup.alpha_e,
                         alpha_i=setup.alpha_i,
                         q_e=setup.q_e,
                         q_i=setup.q_i,
                         C0_e=y[:setup.Nx],
                         C0_i=y[setup.N: setup.N + setup.Nx])

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, atol=1e-12, rtol=1e-12)

    # initialize the rhs dydt
    dydt_ = np.zeros(2*setup.N)
    # evolving electrons
    dydt_[:setup.N] = setup.A_e @ y[:setup.N] + nonlinear_full(E=E, psi=y[:setup.N], Nv=setup.Nv, Nx=setup.Nx,
                                                               alpha=setup.alpha_e, q=setup.q_e, m=setup.m_e)
    # evolving ions
    dydt_[setup.N:] = setup.A_i @ y[setup.N:] + nonlinear_full(E=E, psi=y[setup.N:], Nv=setup.Nv, Nx=setup.Nx,
                                                               alpha=setup.alpha_i, q=setup.q_i, m=setup.m_i)
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupFOM(Nx=251,
                               Nv=300,
                               epsilon=1e-2,
                               alpha_e=np.sqrt(2),
                               alpha_i=1/np.sqrt(2),
                               u_e=3,
                               u_i=0,
                               L=8 * np.pi,
                               dt=1e-2,
                               T0=0,
                               T=40,
                               nu=15,
                               m_e=1,
                               m_i=1,
                               ions=True,
                               construct_B=False)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(2 * setup.N)
    # first electron 1 species (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    k_ = 0.25
    # electrons (perturbed)
    y0[:setup.Nx] = (1 + setup.epsilon * np.cos(x_ * k_)) / setup.alpha_e
    # ions (unperturbed)
    y0[setup.N: setup.N + setup.Nx] = 1 / setup.alpha_i

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_FOM(y_0=y0,
                                                         right_hand_side=rhs,
                                                         a_tol=1e-12,
                                                         r_tol=None,
                                                         max_iter=100,
                                                         param=setup)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # make directory
    if not os.path.exists("../data/FOM/ion_acoustic/sample_" + str(setup.u_e)):
        os.makedirs("../data/FOM/ion_acoustic/sample_" + str(setup.u_e))

    # save the runtime
    np.save("../data/FOM/ion_acoustic/sample_" + str(setup.u_e) + "/sol_FOM_u_" + str(setup.Nv) + "_u_e_" + str(
        setup.u_e) + "_runtime_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../data/FOM/ion_acoustic/sample_" + str(setup.u_e) + "/sol_FOM_u_" + str(setup.Nv) + "_u_e_" + str(
        setup.u_e) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../data/FOM/ion_acoustic/sample_" + str(setup.u_e) + "/sol_FOM_t_" + str(setup.Nv) + "_u_e_" + str(
        setup.u_e) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)
