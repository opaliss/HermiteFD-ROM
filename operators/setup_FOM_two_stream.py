import numpy as np
from operators.FOM import A1, A2, A3, B
from operators.finite_difference import ddx_central


class SimulationSetupTwoStreamFOM:
    def __init__(self, Nx,  Nv, epsilon, alpha_e1, alpha_e2, alpha_i, u_e1, u_e2, u_i, L, dt, T0, T,
                 nu, n0_e1, n0_e2, m_e1=1, m_e2=1, m_i=1836, q_e1=-1, q_e2=-1, q_i=1, ions=False, construct_B=False):
        # set up configuration parameters
        # resolution in space
        self.Nx = Nx
        # resolution in velocity
        self.Nv = Nv
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity scaling of electron and ion
        self.alpha_e1 = alpha_e1
        self.alpha_e2 = alpha_e2
        self.alpha_i = alpha_i
        # velocity scaling
        self.u_e1 = u_e1
        self.u_e2 = u_e2
        self.u_i = u_i
        # average density coefficient
        self.n0_e1 = n0_e1
        self.n0_e2 = n0_e2
        # x grid is from 0 to L
        self.L = L
        self.dx = self.L / self.Nx
        # time stepping
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # vector of timestamps
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0) / self.dt) + 1)
        # mass normalized
        self.m_e1 = m_e1
        self.m_e2 = m_e2
        self.m_i = m_i
        # charge normalized
        self.q_e1 = q_e1
        self.q_e2 = q_e2
        self.q_i = q_i
        # artificial collisional frequency
        self.nu = nu

        # matrices
        # Fourier derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=self.dx, periodic=True)

        # matrix of coefficients (advection)
        A_diag = A2(D=self.D, i=0, j=self.Nv)
        A_off = A1(i=0, j=self.Nv, D=self.D)
        A_col = A3(Nx=self.Nx, i=0, j=self.Nv, Nv=self.Nv)

        # A matrices
        self.A_e1 = self.u_e1 * A_diag + self.alpha_e1 * A_off + nu * A_col
        self.A_e2 = self.u_e2 * A_diag + self.alpha_e2 * A_off + nu * A_col

        # if ions evolve
        if ions:
            self.A_i = self.u_i * A_diag + self.alpha_i * A_off + nu * A_col

        if construct_B:
            # matrix of coefficient (acceleration)
            B_mat = B(i=0, j=self.Nv, Nx=self.Nx)

            # B matrices
            self.B_e1 = self.q_e1 / self.m_e1 / self.alpha_e1 * B_mat
            self.B_e2 = self.q_e2 / self.m_e2 / self.alpha_e2 * B_mat

            # if ions evolve
            if ions:
                self.B_i = self.q_i / self.m_i / self.alpha_i * B_mat

