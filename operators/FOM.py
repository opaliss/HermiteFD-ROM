"""Operators of the FOM (AW) Hermite spectral and (central) Finite Difference solver

Author: Opal Issan (oissan@ucsd.edu)
Date: Dec 18th, 2024
"""
import numpy as np
import scipy.special


def psi_ln_aw(n, alpha_s, u_s, v):
    """AW Hermite basis function (iterative approach)

    :param alpha_s: float, velocity scaling parameter
    :param u_s, float, velocity shifting parameter
    :param v: float or array, the velocity coordinate on a grid
    :param n: int, order of polynomial
    :return: float or array,  asymmetrically weighted (AW) hermite polynomial of degree n evaluated at xi
    """
    # scaled velocity coordinate
    xi = (v - u_s)/alpha_s
    # iteratively compute psi_{n}(xi)
    if n == 0:
        return np.exp(-xi ** 2) / np.sqrt(np.pi)
    if n == 1:
        return np.exp(-xi ** 2) * (2*xi)/np.sqrt(2*np.pi)
    else:
        psi = np.zeros((n+1, len(xi)))
        psi[0, :] = np.exp(-xi ** 2) / np.sqrt(np.pi)
        psi[1, :] = np.exp(-xi ** 2) * (2*xi) / np.sqrt(2*np.pi)
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj+1)/2)
            psi[jj+1, :] = (alpha_s * np.sqrt(jj/2) * psi[jj-1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi[n, :]


def nu_func(n, Nv):
    """coefficient for hypercollisions

    :param n: int, index of spectral term
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :return: float, coefficient for hypercollisions
    """
    return n * (n - 1) * (n - 2) / (Nv - 1) / (Nv - 2) / (Nv - 3)


def A3(Nx, Nv, i, j):
    """A3 matrix in linear_landau advection term with nu

    :param i: 0th index (either 0 or M)
    :param j: final index (either M or Nv)
    :param Nv: int, Hermite spectral resolution
    :param Nx: int, finite difference grid resolution
    :return: 2D matrix, A3 matrix in linear_landau advection term
    """
    A = np.zeros((j-i, j-i))
    for index, n in enumerate(range(i, j)):
        # main diagonal
        A[index, index] = nu_func(n=n, Nv=Nv)
    return -scipy.sparse.kron(A, scipy.sparse.identity(n=Nx), format="csr")


def A2(D, i, j):
    """A2 matrix in linear_landau advection term with u

    :param i: 0th index (either 0 or M)
    :param j: final index (either M or Nv)
    :param D: finite difference derivative matrix
    :return: 2D matrix, A3 matrix in linear_landau advection term
    """
    return -scipy.sparse.kron(scipy.sparse.identity(n=j-i), D, format="csr")


def A1(i, j, D):
    """A1 matrix in linear_landau advection term with alpha

    :param i: 0th index (either 0 or M)
    :param j: final index (either M or Nv)
    :param D: finite difference derivative matrix
    :return: 2D matrix, A1 matrix in linear_landau advection term
    """
    A = np.zeros((j-i, j-i))
    for index, n in enumerate(range(i, j)):
        if n != i:
            # lower diagonal
            A[index, index - 1] = np.sqrt(n / 2)
        if n != j - 1:
            # upper diagonal
            A[index, index + 1] = np.sqrt((n + 1) / 2)
    return -scipy.sparse.kron(A, D, format="csr")


def nonlinear_full(E, psi, q, m, alpha, Nv, Nx):
    """

    :param E: vec size Nx, electric field on finite difference mesh
    :param psi: vec size Nx*Nv, vector of all coefficients
    :param q: float, charge of particles
    :param m: float, mass of particles
    :param alpha: float, temperature of particles
    :param Nx: int, grid size in space
    :param Nv: int, spectral resolution in velocity
    :return: N(E, psi)
    """
    res = np.zeros(len(psi))
    for n in range(Nv):
        if n != 0:
            res[n*Nx: (n+1)*Nx] = q/m/alpha * np.sqrt(2*n) * E * psi[(n-1)*Nx: n*Nx]
    return res


def B(Nx, i, j):
    """B matrix of acceleration term

    :param i: 0th index
    :param j: final index
    :param Nx: int, finite difference resolution
    :return: 2D matrix, B matrix of acceleration term
    """
    B = np.zeros((j-i, j-i))
    for index, n in enumerate(range(i, j)):
        # lower diagonal
        if index >= 1:
            B[index, index - 1] = np.sqrt(2 * n)
    Q_mat = Q(Nx=Nx, j=j, i=i)
    return scipy.sparse.kron(scipy.sparse.csr_matrix(B), scipy.sparse.identity(n=Nx), format="csr") @ Q_mat


def Q(Nx, j, i):
    """

    :param Nx: int, finite difference resolution
    :param j: final index
    :param i: 0th index
    :return: 2D matrix, Q matrix of acceleration term
    """
    mat1 = scipy.sparse.identity(n=(j-i)*Nx, format="csr")
    mat2 = scipy.sparse.kron(np.ones((j-i, 1)), scipy.sparse.identity(n=Nx, format="csr"), format="csr")
    return khatri_rao(mat1.T, mat2.T).T


def khatri_rao(mat1, mat2):
    mat = scipy.sparse.kron(mat1[:, 0], mat2[:, 0], format="csr")
    for k in range(1, mat2.shape[1]):
        mat = scipy.sparse.hstack([mat, scipy.sparse.kron(mat1[:, k], mat2[:, k], format="csr")], format="csr")
    return mat


def charge_density(q_e, q_i, alpha_e, alpha_i, C0_e, C0_i):
    """

    :param q_e:
    :param q_i:
    :param alpha_e:
    :param alpha_i:
    :param C0_e:
    :param C0_i:
    :return:
    """
    return q_e * alpha_e * C0_e + q_i * alpha_i * C0_i


def mass(state):
    """mass of the particular state

    :param state: ndarray, electron or ion state
    :return: mass for the state
    """
    return np.sum(state[0, :])


def momentum(state, u_s, alpha_s):
    """momentum of the particular state

    :param state: ndarray, electron or ion state
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: momentum for the state
    """
    return alpha_s * np.sum(state[1, :])/np.sqrt(2) + u_s * np.sum(state[0, :])


def energy_k(state, u_s, alpha_s):
    """kinetic energy of the particular state

    :param state: ndarray, electron or ion state
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: kinetic energy for the state
    """
    return (alpha_s**2) / np.sqrt(2) * np.sum(state[2, :]) \
           + np.sqrt(2) * u_s * alpha_s * np.sum(state[1, :])\
           + (alpha_s**2/2 + u_s**2) * np.sum(state[0, :])


def total_mass(state, alpha_s, dx):
    """total mass of single electron and ion setup

    :param state: ndarray, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :return: total mass of single electron and ion setup
    """
    return mass(state=state) * dx * alpha_s


def total_momentum(state, alpha_s, dx, m_s, u_s):
    """total momentum of single electron and ion setup

    :param state: ndarray, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param Nv: int, the number of velocity spectral terms
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total momentum of single electron and ion setup
    """
    return momentum(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s


def total_energy_k(state, alpha_s, dx,  m_s, u_s):
    """total kinetic energy of single electron and ion setup

    :param state: ndarray, species s  state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total kinetic energy of single electron and ion setup
    """
    return 0.5 * energy_k(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s

