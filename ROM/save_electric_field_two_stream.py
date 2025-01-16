import numpy as np
from operators.setup_ROM_two_stream import SimulationSetupTwoStreamROM
from operators.poisson_solver import gmres_solver
from operators.FOM import charge_density_two_stream

for u_e2 in [1.065, 1.09, 1.1]:
    print("u_e2 = ", u_e2)
    for M in [3, 6, 9]:
        print("M = ", M)
        for Nr in range(120, 210, 10):
            print("Nr = ", Nr)
            setup = SimulationSetupTwoStreamROM(Nx=251,
                                                Nv=350,
                                                epsilon=0.1,
                                                alpha_e1=0.5,
                                                alpha_e2=0.5,
                                                alpha_i=np.sqrt(2 / 1836),
                                                u_e1=-u_e2,
                                                u_e2=u_e2,
                                                u_i=0,
                                                L=2 * np.pi,
                                                dt=1e-2,
                                                T0=0,
                                                T=30,
                                                nu_e1=15,
                                                nu_e2=15,
                                                n0_e1=0.5,
                                                n0_e2=0.5,
                                                Nr=Nr,
                                                M=M,
                                                problem_dir="two_stream",
                                                Ur_e1=np.load("../data/ROM/two_stream/basis/basis_SVD_e1_0_30_M_" + str(M) + ".npy"),
                                                Ur_e2=np.load("../data/ROM/two_stream/basis/basis_SVD_e2_0_30_M_" + str(M) + ".npy"),
                                                construct=False)
            # ions (unperturbed)
            C0_ions = np.ones(setup.Nx) / setup.alpha_i

            # load the simulation results
            sol_u_reduced = np.load("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M) + "/sol_ROM_u_" + str(
                    setup.Nr) + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy")
            sol_midpoint_t = np.load("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M) + "/sol_ROM_t_" + str(
                    setup.Nr) + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy")


            sol_midpoint_u = np.zeros((2*setup.Nx*setup.Nv, len(sol_midpoint_t)))
            # e1
            sol_midpoint_u[:setup.NF, :] = sol_u_reduced[:setup.NF, :]
            sol_midpoint_u[setup.NF:setup.Nx*setup.Nv, :] = setup.Ur_e1 @ sol_u_reduced[setup.NF:setup.Nr+setup.NF, :]
            # e2
            sol_midpoint_u[setup.Nx*setup.Nv:setup.Nx*setup.Nv +setup.NF, :] = sol_u_reduced[setup.NF + setup.Nr: 2*setup.NF + setup.Nr, :]
            sol_midpoint_u[setup.Nx*setup.Nv +setup.NF:, :] = setup.Ur_e2 @ sol_u_reduced[2*setup.NF + setup.Nr:, :]

            # initialize the states for implicit midpoint (symplectic)
            state_e1_midpoint = np.zeros((setup.Nv, setup.Nx + 1, len(sol_midpoint_t)))
            state_e2_midpoint = np.zeros((setup.Nv, setup.Nx + 1, len(sol_midpoint_t)))
            state_i_midpoint = np.zeros((setup.Nv, setup.Nx + 1, len(sol_midpoint_t)))
            # initialize the electric potential
            E_midpoint = np.zeros((setup.Nx + 1, len(sol_midpoint_t)))

            # unwind the flattening to solve the Vlasov-Poisson system
            # electrons species 1
            state_e1_midpoint[:, :-1, :] = np.reshape(sol_midpoint_u[:setup.Nv * setup.Nx, :], (setup.Nv, setup.Nx, len(sol_midpoint_t)))
            state_e1_midpoint[:, -1, :] = state_e1_midpoint[:, 0, :]
            # electrons species 2
            state_e2_midpoint[:, :-1, :] = np.reshape(sol_midpoint_u[setup.Nv * setup.Nx:, :], (setup.Nv, setup.Nx, len(sol_midpoint_t)))
            state_e2_midpoint[:, -1, :] = state_e2_midpoint[:, 0, :]

            for ii in np.arange(0, len(sol_midpoint_t), 1):
                # immobile ions
                state_i_midpoint[0, :-1, ii] = C0_ions
                # enforce periodicity
                state_i_midpoint[0, -1, ii] = C0_ions[0]

                # solve Poisson's equation to obtain an electric field
                rho = charge_density_two_stream(alpha_e1=setup.alpha_e1, alpha_e2=setup.alpha_e2, alpha_i=setup.alpha_i,
                                                q_e1=setup.q_e1, q_e2=setup.q_e2, q_i=setup.q_i,
                                                C0_e1=state_e1_midpoint[0, :-1, ii], C0_e2=state_e2_midpoint[0, :-1, ii],
                                                C0_i=C0_ions)

                E_midpoint[:-1, ii] = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, atol=1e-12, rtol=1e-12)
                E_midpoint[-1, ii] = E_midpoint[0, ii]

            np.save("../data/ROM/two_stream/sample_" + str(setup.u_e2) + "/M" + str(setup.M) + "/E_ROM_" + str(setup.Nr) + ".npy", E_midpoint)