"""Module to run the linear Landau damping reduced-order model (ROM) testcase

Author: Opal Issan
Date: Dec 23rd, 2024
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.FOM import nonlinear_full, charge_density
from operators.implicit_midpoint_ROM_two_stream import implicit_midpoint_solver_ROM
from operators.setup_ROM import SimulationSetupROM
from operators.poisson_solver import gmres_solver
import time
import scipy


def rhs(y):
    # electric field computed
    # electric field computed
    rho = charge_density(alpha_e=setup.alpha_e, alpha_i=setup.alpha_i,
                         q_e=setup.q_e, q_i=setup.q_i,  C0_e=y[:setup.Nx], C0_i=C0_ions)

    E = gmres_solver(rhs=rho, D=setup.D)

    # initialize
    dydt_ = np.zeros(len(y), dtype="complex128")
    # solve for the first M fluid coefficient
    dydt_[:setup.NF] = setup.A_F_e @ y[: setup.NF] \
                       + nonlinear_full(E=E, psi=y[: setup.NF], Nx=setup.Nx, Nv=setup.M, alpha=setup.alpha_e,
                                        q=setup.q_e, m=setup.m_e) \
                       + setup.G_F_e @ y[setup.NF: setup.NF + setup.Nr]

    # evolve the rest (kinetic portion)
    dydt_[setup.NF: setup.NF + setup.Nr] = setup.A_K_e @ y[setup.NF: setup.NF + setup.Nr] \
                                           + setup.B_K_e @ np.kron(y[setup.NF: setup.NF + setup.Nr], E) \
                                           + setup.G_K_e @ y[: setup.NF] \
                                           + setup.J_K_e @ scipy.signal.convolve(in1=E, in2=y[(setup.M - 1) * setup.Nx: setup.M * setup.Nx], mode="same")

    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupROM(Nx=20,
                               Nx_total=41,
                               Nv=100,
                               epsilon=1e-2,
                               alpha_e=0.9,
                               alpha_i=np.sqrt(2 / 1836),
                               u_e=0,
                               u_i=0,
                               L=2 * np.pi,
                               dt=0.01,
                               T0=0,
                               T=30,
                               nu=0,
                               Nr=100,
                               M=5,
                               problem_dir="linear_landau",
                               Ur_e=np.load("../data/ROM/linear_landau/basis_5.npy"),
                               Ur_i=np.zeros(np.shape(np.load("../data/ROM/linear_landau/basis_5.npy"))),
                               construct=True)

    # save parameters
    # np.save("../data/ROM/two_stream/parameters" + str(setup.Nv) + "_nu_" + str(setup.nu) + "_" + str(setup.T0) + "_" + str(setup.T), setup)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(setup.NF + setup.Nr, dtype="complex128")
    # first electron 1 species (perturbed)
    y0[setup.Nx] = 1 / setup.alpha_e
    y0[setup.Nx + 1] = 0.5 * setup.epsilon / setup.alpha_e
    y0[setup.Nx - 1] = 0.5 * setup.epsilon / setup.alpha_e

    # ions (unperturbed)
    C0_ions = np.zeros(setup.Nx_total)
    C0_ions[setup.Nx] = 1 / setup.alpha_i

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u  = implicit_midpoint_solver_ROM(y_0=y0,
                                                   right_hand_side=rhs,
                                                   r_tol=1e-8,
                                                   a_tol=1e-12,
                                                   max_iter=100,
                                                   setup=setup,
                                                   t_vec=np.linspace(setup.T0, setup.T, int(int(setup.T-setup.T0)/setup.dt)+1),
                                                   Nw=1)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    # make directory
    if not os.path.exists("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M)):
        os.makedirs("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M))

    # ROM performance
    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)
    np.save("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_u_" + str(setup.Nr) + "_nu_" + str(setup.nu) + "_runtime_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_u_" + str(setup.Nr) + "_nu_" + str(setup.nu) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)
    np.save("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_t_" + str(setup.Nr) + "_nu_" + str(setup.nu) + "_" + str(setup.T0) + "_" + str(setup.T), np.linspace(setup.T0, setup.T, int((setup.T-setup.T0)/setup.dt)+1))
