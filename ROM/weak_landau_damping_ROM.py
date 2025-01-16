"""Module to run the weak_landau Landau damping reduced-order model (ROM) testcase

Author: Opal Issan
Date: Dec 26th, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.FOM import nonlinear_full, charge_density
from operators.implicit_midpoint_ROM_two_stream import implicit_midpoint_solver_ROM
from operators.setup_ROM import SimulationSetupROM
from operators.poisson_solver import gmres_solver
import time


def rhs(y):
    # electric field computed (poisson solver)
    rho = charge_density(alpha_e=setup.alpha_e, alpha_i=setup.alpha_i, q_e=setup.q_e, q_i=setup.q_i, C0_e=y[:setup.Nx], C0_i=C0_ions)

    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, atol=1e-12, rtol=1e-12)

    # initialize
    dydt_ = np.zeros(len(y))
    # solve for the first M fluid coefficient
    dydt_[:setup.NF] = setup.A_F_e @ y[: setup.NF] \
                       + nonlinear_full(E=E, psi=y[: setup.NF], Nx=setup.Nx, Nv=setup.M, alpha=setup.alpha_e, q=setup.q_e, m=setup.m_e) \
                       + setup.G_F_e @ y[setup.NF:]

    # evolve the rest (kinetic portion)
    dydt_[setup.NF:] = setup.A_K_e @ y[setup.NF:] \
                       + setup.B_K_e @ np.kron(y[setup.NF:], E) \
                       + setup.G_K_e @ y[: setup.NF] \
                       + setup.J_K_e @ (y[setup.NF - setup.Nx: setup.NF] * E)

    return dydt_


if __name__ == "__main__":
    for M in range(3, 7):
        for Nr in range(5, 45, 5):
            setup = SimulationSetupROM(Nx=151,
                                       Nv=20,
                                       epsilon=1e-2,
                                       alpha_e=0.75,
                                       alpha_i=np.sqrt(2 / 1836),
                                       u_e=0,
                                       u_i=0,
                                       L=2 * np.pi,
                                       dt=1e-2,
                                       T0=0,
                                       T=80,
                                       nu=10,
                                       Nr=Nr,
                                       M=M,
                                       problem_dir="weak_landau",
                                       Ur_e=np.load("../data/ROM/weak_landau/basis_" + str(M) + ".npy"),
                                       construct=True,
                                       ions=False)

            # save the reduced operators
            # setup.save_operators()

            # initial condition: read in result from previous simulation
            y0 = np.zeros(setup.NF + setup.Nr)
            # first electron 1 species (perturbed)
            x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
            y0[:setup.Nx] = (1 + setup.epsilon * np.cos(x_)) / setup.alpha_e
            # ions (unperturbed)
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
                                                          setup=setup,
                                                          windowing=False)

            end_time_cpu = time.process_time() - start_time_cpu
            end_time_wall = time.time() - start_time_wall

            # make directory
            if not os.path.exists("../data/ROM/weak_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M)):
                os.makedirs("../data/ROM/weak_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M))

            # ROM performance
            print("runtime cpu = ", end_time_cpu)
            print("runtime wall = ", end_time_wall)
            np.save("../data/ROM/weak_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_u_" + str(
                setup.Nr) + "_nu_" + str(setup.nu) + "_runtime_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

            # save results
            np.save("../data/ROM/weak_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_u_" + str(
                setup.Nr) + "_nu_" + str(setup.nu) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)
            np.save("../data/ROM/weak_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_t_" + str(
                setup.Nr) + "_nu_" + str(setup.nu) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)
