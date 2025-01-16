"""Module to run the two=stream instability reduced-order model (ROM) testcase

Author: Opal Issan
Date: Jan 9th, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.FOM import nonlinear_full, charge_density_two_stream
from operators.implicit_midpoint_ROM_two_stream import implicit_midpoint_solver_ROM
from operators.setup_ROM_two_stream import SimulationSetupTwoStreamROM
from operators.poisson_solver import gmres_solver
import time


def rhs(y):
    # charge density computed for poisson's equation
    rho = charge_density_two_stream(C0_e1=y[:setup.Nx],
                                    C0_e2=y[setup.NF + setup.Nr: setup.Nx + setup.NF + setup.Nr],
                                    C0_i=C0_ions, alpha_e1=setup.alpha_e1, alpha_e2=setup.alpha_e2,
                                    alpha_i=setup.alpha_i, q_e1=setup.q_e1, q_e2=setup.q_e2, q_i=setup.q_i)

    # electric field computed
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, atol=1e-12, rtol=1e-12)

    # initialize the rhs dydt
    dydt_ = np.zeros(len(y))
    # evolve the first M fluid coefficient species (1)
    dydt_[:setup.NF] = setup.A_F_e1 @ y[:setup.NF] \
                       + nonlinear_full(E=E, psi=y[:setup.NF], Nx=setup.Nx, Nv=setup.M, alpha=setup.alpha_e1, q=setup.q_e1, m=setup.m_e1) \
                       + setup.G_F_e1 @ y[setup.NF:setup.NF + setup.Nr]

    # evolve the rest (kinetic portion) species (1)
    dydt_[setup.NF: setup.NF + setup.Nr] = setup.A_K_e1 @ y[setup.NF: setup.NF + setup.Nr] \
                                           + setup.B_K_e1 @ np.kron(y[setup.NF: setup.NF + setup.Nr], E) \
                                           + setup.G_K_e1 @ y[:setup.NF] \
                                           + setup.J_K_e1 @ (y[setup.NF - setup.Nx: setup.NF] * E)

    # evolve the first M fluid coefficient species (2)
    dydt_[setup.NF + setup.Nr: 2 * setup.NF + setup.Nr] = setup.A_F_e2 @ y[setup.NF + setup.Nr: 2 * setup.NF + setup.Nr] \
                                                          + nonlinear_full(E=E, psi=y[setup.NF + setup.Nr: 2 * setup.NF + setup.Nr],
                                                                           Nx=setup.Nx, Nv=setup.M, alpha=setup.alpha_e2,
                                                                           q=setup.q_e2, m=setup.m_e2) \
                                                          + setup.G_F_e2 @ y[2 * setup.NF + setup.Nr:]

    # evolve the rest (kinetic portion) species (2)
    dydt_[2 * setup.NF + setup.Nr:] = setup.A_K_e2 @ y[2 * setup.NF + setup.Nr:] \
                                      + setup.B_K_e2 @ np.kron(y[2 * setup.NF + setup.Nr:], E) \
                                      + setup.G_K_e2 @ y[setup.NF + setup.Nr: 2 * setup.NF + setup.Nr] \
                                      + setup.J_K_e2 @ (y[2 * setup.NF + setup.Nr - setup.Nx: 2 * setup.NF + setup.Nr] * E)
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupTwoStreamROM(Nx=251,
                                        Nv=350,
                                        epsilon=0.1,
                                        alpha_e1=0.5,
                                        alpha_e2=0.5,
                                        alpha_i=np.sqrt(2 / 1836),
                                        u_e1=-1.065,
                                        u_e2=1.065,
                                        u_i=0,
                                        L=2 * np.pi,
                                        dt=1e-2,
                                        T0=0,
                                        T=30,
                                        nu_e1=15,
                                        nu_e2=15,
                                        n0_e1=0.5,
                                        n0_e2=0.5,
                                        Nr=120,
                                        M=3,
                                        problem_dir="two_stream",
                                        Ur_e1=np.load("../data/ROM/two_stream/basis_SVD_e1_0_30_M_3.npy"),
                                        Ur_e2=np.load("../data/ROM/two_stream/basis_SVD_e2_0_30_M_3.npy"),
                                        construct=True)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(2 * setup.NF + 2 * setup.Nr)
    # first electron 1 species (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    y0[:setup.Nx] = setup.n0_e1 * (np.ones(setup.Nx) + setup.epsilon * np.cos(x_)) / setup.alpha_e1
    # second electron species (unperturbed)
    y0[setup.NF + setup.Nr: setup.NF + setup.Nr + setup.Nx] = setup.n0_e2 * (np.ones(setup.Nx) + setup.epsilon * np.cos(x_)) / setup.alpha_e2
    # ions (unperturbed + static)
    C0_ions = np.ones(setup.Nx) / setup.alpha_i

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver_ROM(y_0=y0,
                                                  right_hand_side=rhs,
                                                  r_tol=None,
                                                  a_tol=1e-12,
                                                  max_iter=100,
                                                  setup=setup)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    # make directory
    if not os.path.exists("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M)):
        os.makedirs("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M))

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)
    np.save("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M) + "/sol_ROM_u_" + str(
        setup.Nr) + "_runtime_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M) + "/sol_ROM_u_" + str(
        setup.Nr) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)
    np.save("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M) + "/sol_ROM_t_" + str(
        setup.Nr) +  "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)
