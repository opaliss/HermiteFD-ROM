import numpy as np
from operators.FOM import A1, A2, A3, B
from operators.finite_difference import ddx_central
from operators.ROM import theta_matrix, xi_matrix
import scipy


class SimulationSetupROM:
    def __init__(self, Nx, Nv, epsilon, alpha_e, alpha_i, u_e, u_i, L, dt, T0, T, nu,
                 M, Nr, Ur_e, Ur_i, problem_dir, m_e=1, m_i=1836, q_e=-1, q_i=1, construct=True):
        # set up configuration parameters
        # resolution in space
        self.Nx = Nx
        # resolution in velocity
        self.Nv = Nv
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity scaling parameter (mean thermal velocity)
        self.alpha_e = alpha_e
        self.alpha_i = alpha_i
        # velocity shifting parameter (mean bulk velocity)
        self.u_e = u_e
        self.u_i = u_i
        # x grid is from 0 to L
        self.L = L
        # time stepping
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # mass normalized to m_{e}
        self.m_e = m_e
        self.m_i = m_i
        # charge normalized to q_{e}
        self.q_e = q_e
        self.q_i = q_i
        # artificial collisional frequency
        self.nu = nu
        # dimensionality parameters
        self.M = M
        self.Nr = Nr
        self.NF = self.M * self.Nx
        self.NK = (self.Nv - self.M) * self.Nx

        # matrices
        # derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=L/self.Nx, periodic=True)

        # projection matrices
        self.Ur_e = Ur_e[:, :self.Nr]
        self.Ur_i = Ur_i[:, :self.Nr]

        # problem directory
        self.problem_dir = problem_dir

        if construct:
            self.construct_operators()
        else:
            self.load_operators()

    def save_operators(self):
        # save advection matrices
        # kinetic
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) +  "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_K_e)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) +  "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_K_i)

        # fluid
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) +  "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_F_e.todense())
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.A_F_i.todense())

        # save acceleration matrices
        # kinetic
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_K_e)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_K_i)

        # fluid
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_F_e.todense())
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.B_F_i.todense())

        # G matrix
        # kinetic
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_K_e)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_K_i)

        # fluid
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_F_e)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.G_F_i)

        # J matrix
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.J_K_e)
        np.save("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", self.J_K_i)


    def load_operators(self):
        # load advection matrices
        # kinetic
        self.A_K_e = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.A_K_i = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # fluid
        self.A_F_e = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.A_F_i = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/A_F_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # save acceleration matrices
        # kinetic
        self.B_K_e = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.B_K_i = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # fluid
        self.B_F_e = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.B_F_i = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/B_F_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # G matrix
        # kinetic
        self.G_K_e = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.G_K_i = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # fluid
        self.G_F_e = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.G_F_i = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/G_F_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)

        # J matrix
        self.J_K_e = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_e_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)
        self.J_K_i = np.load("../data/ROM/" + str(self.problem_dir) + "/operators/J_K_i_M_" + str(self.M) + "_Nr_" + str(self.Nr) + "_" + str(self.T0) + "_" + str(self.T) + ".npy", allow_pickle=True)


    def construct_operators(self):
        # fluid matrices
        # matrix of coefficients (advection)
        A_diag_F = A2(j=self.M, i=0, D=self.D)
        A_off_F = A1(i=0, j=self.M, D=self.D)
        A_col_F = A3(Nx=self.Nx, Nv=self.Nv, i=0, j=self.M)

        self.A_F_e = self.alpha_e * A_off_F + self.u_e * A_diag_F + self.nu * A_col_F
        self.A_F_i = self.alpha_i * A_off_F + self.u_i * A_diag_F + self.nu * A_col_F

        del A_diag_F, A_off_F, A_col_F

        # matrix of coefficient (acceleration)
        self.B_F_e = (self.q_e / (self.m_e * self.alpha_e)) * B(i=0, j=self.M, Nx=self.Nx)
        self.B_F_i = (self.q_i / (self.m_i * self.alpha_i)) * B(i=0, j=self.M, Nx=self.Nx)

        # kinetic matrices
        # matrix of coefficients (advection)
        A_diag_K = A2(j=self.Nv-self.M, i=self.M, D=self.D)
        A_off_K = A1(i=self.M, j=self.Nv, D=self.D)
        A_col_K = A3(Nx=self.Nx, Nv=self.Nv, i=self.M, j=self.Nv)

        # reduced diagonal A matrix
        self.A_K_e = self.u_e * self.get_kinetic_reduced_A_matrix(A=A_diag_K, Ur=self.Ur_e) \
                   + self.alpha_e * self.get_kinetic_reduced_A_matrix(A=A_off_K, Ur=self.Ur_e) \
                   + self.nu * self.get_kinetic_reduced_A_matrix(A=A_col_K, Ur=self.Ur_e)
        self.A_K_i = self.u_i * self.get_kinetic_reduced_A_matrix(A=A_diag_K, Ur=self.Ur_i) \
                   + self.alpha_i * self.get_kinetic_reduced_A_matrix(A=A_off_K, Ur=self.Ur_i) \
                   + self.nu * self.get_kinetic_reduced_A_matrix(A=A_col_K, Ur=self.Ur_i)

        del A_diag_K, A_col_K, A_off_K

        # matrix of coefficient (acceleration)
        B_K = B(Nx=self.Nx, i=self.M, j=self.Nv)

        self.B_K_e = (self.q_e / (self.m_e * self.alpha_e)) * self.get_kinetic_reduced_B_matrix(B=B_K, Ur=self.Ur_e, Nx=self.Nx)
        self.B_K_i = self.get_kinetic_reduced_B_matrix(B=B_K, Ur=self.Ur_i, Nx=self.Nx)

        del B_K

        # sparse coupling matrices
        G_F = - np.sqrt(self.M / 2) * xi_matrix(Nx=self.Nx, Nv=self.M) @ self.D @ theta_matrix(Nx=self.Nx, Nv=self.Nv - self.M).T
        upsilon_K = np.sqrt(2 * self.M) * theta_matrix(Nx=self.Nx, Nv=self.Nv - self.M)

        self.G_F_e = self.alpha_e * self.get_fluid_reduced_G_matrix(G=G_F, Ur=self.Ur_e)
        self.G_F_i = self.alpha_i * self.get_fluid_reduced_G_matrix(G=G_F, Ur=self.Ur_i)

        self.G_K_e = self.alpha_e * self.get_kinetic_reduced_G_matrix(G=G_F.T, Ur=self.Ur_e)
        self.G_K_i = self.alpha_i * self.get_kinetic_reduced_G_matrix(G=G_F.T, Ur=self.Ur_i)

        self.J_K_e = self.q_e / (self.m_e * self.alpha_e) * self.get_kinetic_reduced_G_matrix(G=upsilon_K, Ur=self.Ur_e)
        self.J_K_i = self.q_i / (self.m_i * self.alpha_i) * self.get_kinetic_reduced_G_matrix(G=upsilon_K, Ur=self.Ur_i)

        del G_F, upsilon_K

    def get_kinetic_reduced_A_matrix(self, A, Ur):
        return Ur.T @ A @ Ur

    def get_kinetic_reduced_B_matrix(self, B, Ur, Nx):
        return Ur.T @ B @ scipy.sparse.kron(Ur, scipy.sparse.identity(n=Nx), format="bsr")

    def get_fluid_reduced_G_matrix(self, G, Ur):
        return G @ Ur

    def get_kinetic_reduced_G_matrix(self, G, Ur):
        return Ur.T @ G