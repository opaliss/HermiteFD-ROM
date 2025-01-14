import numpy as np
import scipy
from operators.setup_ROM import SimulationSetupROM
from operators.poisson_solver import gmres_solver
from operators.FOM import charge_density

for M in range(3, 7):
    for Nr in np.arange(8, 50, 2):
        setup = SimulationSetupROM(Nx=151,
                                   Nv=20,
                                   epsilon=1e-2,
                                   alpha_e=0.5,
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
                                   problem_dir="linear_landau",
                                   Ur_e=np.load("../data/ROM/linear_landau/basis_" + str(M) + ".npy"),
                                   construct=False,
                                   load=False,
                                   ions=False)

        # ions (unperturbed)
        C0_ions = np.ones(setup.Nx) / setup.alpha_i


        # load the simulation results
        sol_u_reduced = np.load("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_u_" + str(setup.Nr) + "_nu_" + str(setup.nu) + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy")
        sol_midpoint_t = np.load("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_t_" + str(setup.Nr) + "_nu_" + str(setup.nu) + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy")

        # project back up the reduced simulation results
        sol_u_ROM = np.zeros((setup.Nx*setup.Nv, len(sol_midpoint_t)))
        sol_u_ROM[:setup.NF, :] = sol_u_reduced[:setup.NF, :]
        sol_u_ROM[setup.NF:, :] = setup.Ur_e @ sol_u_reduced[setup.NF:, :]

        # initialize the states for implicit midpoint (symplectic)
        state_e_midpoint = np.zeros((setup.Nv, setup.Nx + 1, len(sol_midpoint_t)))
        state_i_midpoint = np.zeros((setup.Nv, setup.Nx + 1, len(sol_midpoint_t)))
        # initialize the electric potential
        E_midpoint = np.zeros((setup.Nx + 1, len(sol_midpoint_t)))

        for ii in np.arange(0, len(sol_midpoint_t), 1):
            print("time = ", sol_midpoint_t[ii])
            for n in range(setup.Nv):
                # unwind the flattening to solve the Vlasov-Poisson system
                # electrons
                state_e_midpoint[n, :-1, ii] = sol_u_ROM[n * setup.Nx: (n + 1) * setup.Nx, ii]
                state_e_midpoint[n, -1, ii] = state_e_midpoint[n, 0, ii]

            # immobile ions
            state_i_midpoint[0, :-1, ii] = C0_ions
            state_i_midpoint[0, -1, ii] = state_i_midpoint[0, 0, ii]

            # solve Poisson's equation to obtain an electric field
            rho = charge_density(alpha_e=setup.alpha_e, alpha_i=setup.alpha_i,
                                 q_e=setup.q_e, q_i=setup.q_i,
                                 C0_e=state_e_midpoint[0, :setup.Nx, ii],
                                 C0_i=C0_ions)

            E_midpoint[:-1, ii] = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, atol=1e-12, rtol=1e-12)
            E_midpoint[-1, ii] = E_midpoint[0, ii]

        # compute the electric field k=1 mode amplitude using fft
        E1_midpoint = np.zeros(len(sol_midpoint_t))

        for ii in range(len(sol_midpoint_t)):
            E1_midpoint[ii] = np.abs(scipy.fft.fft(E_midpoint[:, ii]))[1] / setup.Nx  # normalize fft

        np.save("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_E_" + str(setup.Nr) + "_alpha_" + str(setup.alpha_e) + ".npy", E_midpoint)
        np.save("../data/ROM/linear_landau/sample_" + str(setup.alpha_e) + "/M" + str(setup.M) + "/sol_midpoint_E1_" + str(setup.Nr) + "_alpha_" + str(setup.alpha_e) + ".npy", E1_midpoint)