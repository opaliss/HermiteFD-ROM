import numpy as np
from operators.FOM import A1, A2, A3, B
from operators.finite_difference import ddx_central
import scipy

class SimulationSetupFOM:
    def __init__(self, Nx, Nv, epsilon, alpha_e, alpha_i, u_e, u_i, L, dt, T0, T, nu,
                 m_e=1, m_i=1836, q_e=-1, q_i=1, ions=False, problem_dir=None):
        # set up configuration parameters
        # spatial resolution
        self.Nx = Nx
        # velocity resolution
        self.Nv = Nv
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity scaling parameter (mean thermal velocity)
        self.alpha_e = alpha_e
        self.alpha_i = alpha_i
        # velocity shifting parameter (mean fluid velocity)
        self.u_e = u_e
        self.u_i = u_i
        # x grid is from 0 to L
        self.L = L
        self.dx = self.L / self.Nx
        # time stepping delta t
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # vector of timestamps
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0) / self.dt) + 1)
        # mass normalized
        self.m_e = m_e
        self.m_i = m_i
        # charge normalized
        self.q_e = q_e
        self.q_i = q_i
        # artificial collisional frequency
        self.nu = nu
        #
        self.problem_dir = problem_dir

        # matrices
        # Fourier derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=self.dx, periodic=True, order=2)

        # matrix of coefficients (advection)
        A_diag = A2(D=self.D, i=0, j=self.Nv)
        A_off = A1(i=0, j=self.Nv, D=self.D)
        A_col = A3(Nx=self.Nx, Nv=self.Nv, i=0, j=self.Nv)

        self.A_e = self.alpha_e * A_off + self.u_e * A_diag + self.nu * A_col
        if ions:
            self.A_i = self.alpha_i * A_off + self.u_i * A_diag + self.nu * A_col

        self.B_e = self.q_e/self.m_e/self.alpha_e * B(Nx=self.Nx, i=0, j=self.Nv)


