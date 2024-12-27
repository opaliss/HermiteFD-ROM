"""Module to run the bump-on-tail instability full-order model (FOM) testcase

Author: Opal Issan
Date: Dec 26th, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.FOM import nonlinear_full, charge_density_two_stream
from operators.implicit_midpoint_FOM import implicit_midpoint_solver_FOM
from operators.setup_FOM_two_stream import SimulationSetupTwoStreamFOM
from operators.poisson_solver import gmres_solver, fft_solver, linear_solver_v2
import time
import numpy as np


def rhs(y):
    # charge density computed for poisson's equation
    rho = charge_density_two_stream(C0_e1=y[:setup.Nx],
                                    C0_e2=y[setup.Nx * setup.Nv: setup.Nx * (setup.Nv + 1)],
                                    C0_i=C0_ions, alpha_e1=setup.alpha_e1, alpha_e2=setup.alpha_e2,
                                    alpha_i=setup.alpha_i, q_e1=setup.q_e1, q_e2=setup.q_e2, q_i=setup.q_i)

    # electric field computed
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, atol=1e-10, rtol=1e-10)

    # initialize the rhs dydt
    dydt_ = np.zeros(len(y))
    # evolving electrons
    # electron species (1) => bulk
    dydt_[:setup.Nv * setup.Nx] = setup.A_e1 @ y[:setup.Nv * setup.Nx] \
                                  + nonlinear_full(E=E, psi=y[:setup.Nv * setup.Nx], Nv=setup.Nv, Nx=setup.Nx,
                                                    q=setup.q_e1, m=setup.m_e1, alpha=setup.alpha_e1)

    # electron species (2) => bump
    dydt_[setup.Nv * setup.Nx:] = setup.A_e2 @ y[setup.Nv * setup.Nx:] \
                                  + nonlinear_full(E=E, psi=y[setup.Nv * setup.Nx:], Nv=setup.Nv, Nx=setup.Nx,
                                                   q=setup.q_e2, m=setup.m_e2, alpha=setup.alpha_e2)
    return dydt_


if __name__ == "__main__":
    for u_e2 in [4, 4.1, 4.2, 4.3, 4.35, 4.4, 4.5, 4.6, 4.7]:
        setup = SimulationSetupTwoStreamFOM(Nx=151,
                                            Nv=250,
                                            epsilon=1e-2,
                                            alpha_e1=0.5,
                                            alpha_e2=0.5,
                                            alpha_i=np.sqrt(2 / 1836),
                                            u_e1=-1,
                                            u_e2=1,
                                            u_i=0,
                                            L=2 * np.pi,
                                            dt=1e-2,
                                            T0=0,
                                            T=40,
                                            nu_e1=5,
                                            nu_e2=5,
                                            n0_e1=0.5,
                                            n0_e2=0.5)

        # initial condition: read in result from previous simulation
        y0 = np.zeros(2 * setup.Nv * setup.Nx)
        # first electron 1 species (perturbed)
        x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
        y0[:setup.Nx] = setup.n0_e1 * (1 + setup.epsilon * np.cos(x_)) / setup.alpha_e1
        # second electron species (unperturbed)
        y0[setup.Nv * setup.Nx: setup.Nv * setup.Nx + setup.Nx] = setup.n0_e1 * (1 + setup.epsilon * np.cos(x_)) / setup.alpha_e1
        # ions (unperturbed + static)
        C0_ions = np.ones(setup.Nx) / setup.alpha_i

        # start timer
        start_time_cpu = time.process_time()
        start_time_wall = time.time()

        # integrate (implicit midpoint)
        sol_midpoint_u, setup = implicit_midpoint_solver_FOM(y_0=y0,
                                                             right_hand_side=rhs,
                                                             r_tol=1e-6,
                                                             a_tol=1e-10,
                                                             max_iter=100,
                                                             param=setup)

        end_time_cpu = time.process_time() - start_time_cpu
        end_time_wall = time.time() - start_time_wall

        # make directory
        if not os.path.exists("../data/FOM/two_stream/sample_" + str(setup.u_e2)):
            os.makedirs("../data/FOM/two_stream/sample_" + str(setup.u_e2))

        print("runtime cpu = ", end_time_cpu)
        print("runtime wall = ", end_time_wall)
        np.save("../data/FOM/two_stream/sample_" + str(setup.u_e2) + "/sol_FOM_u_" + str(setup.Nv) + "_nu_" + str(
            setup.nu_e1) + "_runtime_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

        # save results
        np.save("../data/FOM/two_stream/sample_" + str(setup.u_e2) + "/sol_FOM_u_" + str(setup.Nv) + "_nu_" + str(
            setup.nu_e1) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)
        np.save("../data/FOM/two_stream/sample_" + str(setup.u_e2) + "/sol_FOM_t_" + str(setup.Nv) + "_nu_" + str(
            setup.nu_e1) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

