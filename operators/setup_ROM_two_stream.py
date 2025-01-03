import numpy as np
from operators.FOM import A1, A2, A3, Q, B
from operators.finite_difference import ddx_central
from operators.ROM import theta_matrix, xi_matrix
from operators.setup_ROM import get_kinetic_reduced_A_matrix, get_kinetic_reduced_B_matrix, \
    get_kinetic_reduced_G_matrix, get_fluid_reduced_G_matrix, get_D_inv, get_kinetic_reduced_B_alternative


class SimulationSetupTwoStreamROM:
    def __init__(self, Nx, Nv, epsilon, alpha_e1, alpha_e2, alpha_i, u_e1, u_e2, u_i, L, dt, T0, T, nu_e1, nu_e2,
                 M, Nr, Ur_e1, Ur_e2, problem_dir, n0_e1, n0_e2, m_e1=1, m_e2=1, m_i=1836, q_e1=-1, q_e2=-1, q_i=1,
                 nu_i=0, construct=True, load=False):
        # set up configuration parameters
        # grid resolution in x
        self.Nx = Nx
        # spectral resolution in v
        self.Nv = Nv
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity scaling parameter
        self.alpha_e1 = alpha_e1
        self.alpha_e2 = alpha_e2
        self.alpha_i = alpha_i
        # velocity shifting parameter
        self.u_e1 = u_e1
        self.u_e2 = u_e2
        self.u_i = u_i
        # x grid is from 0 to L
        self.L = L
        self.dx = self.L / self.Nx
        # time stepping
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0)/self.dt) + 1)
        # mass normalized
        self.m_e1 = m_e1
        self.m_e2 = m_e2
        self.m_i = m_i
        # charge normalized
        self.q_e1 = q_e1
        self.q_e2 = q_e2
        self.q_i = q_i
        # artificial collisional frequency
        self.nu_e1 = nu_e1
        self.nu_e2 = nu_e2
        self.nu_i = nu_i
        # dimensionality parameters
        self.M = M
        self.Nr = Nr
        self.NF = self.M * self.Nx
        self.NK = (self.Nv - self.M) * self.Nx
        # density coefficients
        self.n0_e1 = n0_e1
        self.n0_e2 = n0_e2

        # matrices
        # Fourier derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=self.dx, periodic=True)
        self.D_inv = get_D_inv(Nx=self.Nx, D=self.D)

        # projection matrices
        self.Ur_e1 = Ur_e1[:, :self.Nr]
        self.Ur_e2 = Ur_e2[:, :self.Nr]

        # problem directory
        self.problem_dir = problem_dir

        if construct:
            self.construct_operators()
        elif load:
            self.load_operators()

    def save_operators(self):
        # save advection matrices
        # kinetic
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) +  "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_K_e1)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) +  "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_K_e2)

        # fluid
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) +  "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_F_e1.todense())
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) +  "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_F_e2.todense())

        # save acceleration matrices
        # kinetic
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_K_e1)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_K_e2)

        # fluid
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_F_e1.todense())
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_F_e2.todense())

        # G matrix
        # kinetic
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_K_e1)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_K_e2)

        # fluid
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_F_e1)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_F_e2)

        # J matrix
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.J_K_e1)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.J_K_e2)


    def load_operators(self):
        # load advection matrices
        # kinetic
        self.A_K_e1 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.A_K_e2 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # fluid
        self.A_F_e1 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.A_F_e2 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # save acceleration matrices
        # kinetic
        self.B_K_e1 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.B_K_e2 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # fluid
        self.B_F_e1 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.B_F_e2 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # G matrix
        # kinetic
        self.G_K_e1 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.G_K_e2 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # fluid
        self.G_F_e1 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.G_F_e2 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # J matrix
        self.J_K_e1 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_e1_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.J_K_e2 = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_e2_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)


    def construct_operators(self):
        # fluid matrices
        # (general) matrix of coefficients (fluid)
        A_diag_F = A2(j=self.M, i=0, D=self.D)
        A_off_F = A1(i=0, j=self.M, D=self.D)
        A_col_F = A3(Nx=self.Nx, Nv=self.Nv, i=0, j=self.M)
        B_F = B(i=0, j=self.M, Nx=self.Nx)

        # affine parameteric operators
        self.A_F_e1 = self.alpha_e1 * A_off_F + self.u_e1 * A_diag_F + self.nu_e1 * A_col_F
        self.A_F_e2 = self.alpha_e2 * A_off_F + self.u_e2 * A_diag_F + self.nu_e2 * A_col_F

        # matrix of coefficient (acceleration)
        self.B_F_e1 = self.q_e1 / self.m_e1 / self.alpha_e1 * B_F
        self.B_F_e2 = self.q_e2 / self.m_e2 / self.alpha_e2 * B_F

        # kinetic matrices
        # (general) matrix of coefficients (advection)
        A_diag_K = A2(i=self.M, j=self.Nv, D=self.D)
        A_off_K = A1(i=self.M, j=self.Nv, D=self.D)
        A_col_K = A3(Nx=self.Nx, Nv=self.Nv, i=self.M, j=self.Nv)

        # affine parameteric operators
        self.A_K_e1 = self.alpha_e1 * get_kinetic_reduced_A_matrix(A=A_off_K, Ur=self.Ur_e1) \
                      + self.u_e1 * get_kinetic_reduced_A_matrix(A=A_diag_K, Ur=self.Ur_e1) \
                      + self.nu_e1 * get_kinetic_reduced_A_matrix(A=A_col_K, Ur=self.Ur_e1)

        self.A_K_e2 = self.alpha_e2 * get_kinetic_reduced_A_matrix(A=A_off_K, Ur=self.Ur_e2) \
                      + self.u_e2 * get_kinetic_reduced_A_matrix(A=A_diag_K, Ur=self.Ur_e2) \
                      + self.nu_e2 * get_kinetic_reduced_A_matrix(A=A_col_K, Ur=self.Ur_e2)

        del A_diag_K, A_col_K, A_off_K

        # matrix of coefficient (acceleration)
        # Fourier transform matrix
        # B_K = B(i=self.M, j=self.Nv, Nx=self.Nx)
        # ;;l
        # self.B_K_e1 = self.q_e1 / self.m_e1 / self.alpha_e1 * get_kinetic_reduced_B_matrix(B=B_K, Ur=self.Ur_e1, Nx=self.Nx, Nv=self.Nv, Nr=self.Nr)
        # self.B_K_e2 = self.q_e2 / self.m_e2 / self.alpha_e2 * get_kinetic_reduced_B_matrix(B=B_K, Ur=self.Ur_e2, Nx=self.Nx, Nv=self.Nv, Nr=self.Nr)
        Q_ = Q(Nx=self.Nx, j=self.Nv, i=self.M, method="sparse")
        self.B_K_e1 = self.q_e1 / self.m_e1 / self.alpha_e1 * get_kinetic_reduced_B_alternative(Ur=self.Ur_e1, Nx=self.Nx, Nr=self.Nr, i=self.M, j=self.Nv, Q=Q_)
        self.B_K_e2 = self.q_e2 / self.m_e2 / self.alpha_e2 * get_kinetic_reduced_B_alternative(Ur=self.Ur_e2, Nx=self.Nx, Nr=self.Nr, i=self.M, j=self.Nv, Q=Q_)

        del Q_

        # sparse coupling matrices
        G_F = - np.sqrt(self.M / 2) * xi_matrix(Nx=self.Nx, Nv=self.M) @ self.D @ theta_matrix(Nx=self.Nx, Nv=self.Nv - self.M).T
        J_K = np.sqrt(2 * self.M) * theta_matrix(Nx=self.Nx, Nv=self.Nv - self.M)

        # affine parametric operators
        self.G_F_e1 = self.alpha_e1 * get_fluid_reduced_G_matrix(G=G_F, Ur=self.Ur_e1)
        self.G_F_e2 = self.alpha_e2 * get_fluid_reduced_G_matrix(G=G_F, Ur=self.Ur_e2)

        self.G_K_e1 = self.alpha_e1 * get_kinetic_reduced_G_matrix(G=-G_F.T, Ur=self.Ur_e1)
        self.G_K_e2 = self.alpha_e2 * get_kinetic_reduced_G_matrix(G=-G_F.T, Ur=self.Ur_e2)

        self.J_K_e1 = self.q_e1 / self.m_e1 / self.alpha_e1 * get_kinetic_reduced_G_matrix(G=J_K, Ur=self.Ur_e1)
        self.J_K_e2 = self.q_e2 / self.m_e2 / self.alpha_e2 * get_kinetic_reduced_G_matrix(G=J_K, Ur=self.Ur_e2)

