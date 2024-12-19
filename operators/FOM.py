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
    """A3 matrix in linear advection term with nu

    :param i: 0th index (either 0 or M)
    :param j: final index (either M or Nv)
    :param Nv: int, Hermite spectral resolution
    :param Nx: int, finite difference grid resolution
    :return: 2D matrix, A3 matrix in linear advection term
    """
    A = np.zeros((j-i, j-i))
    for index, n in enumerate(range(i, j)):
        # main diagonal
        A[index, index] = nu_func(n=n, Nv=Nv)
    return -scipy.sparse.kron(A, scipy.sparse.identity(n=Nx), format="csr")


def A2(D, i, j):
    """A2 matrix in linear advection term with u

    :param i: 0th index (either 0 or M)
    :param j: final index (either M or Nv)
    :param D: finite difference derivative matrix
    :return: 2D matrix, A3 matrix in linear advection term
    """
    return -scipy.sparse.kron(np.sparse.identity(n=j-i), D, format="csr")


def A1(i, j, D):
    """A1 matrix in linear advection term with alpha

    :param i: 0th index (either 0 or M)
    :param j: final index (either M or Nv)
    :param D: finite difference derivative matrix
    :return: 2D matrix, A1 matrix in linear advection term
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
    return scipy.sparse.kron(B, scipy.sparse.identity(Nx), format="csr") @ Q(Nx=Nx, j=j, i=i)


def Q(Nx, j, i):
    """

    :param Nx: int, finite difference resolution
    :param j: final index
    :param i: 0th index
    :return: 2D matrix, Q matrix of acceleration term
    """
    mat1 = scipy.sparse.identity(n=(j-i)*Nx)
    mat2 = scipy.sparse.kron(np.ones((j-i, 1)), scipy.sparse.identity(n=Nx))
    return scipy.linalg.khatri_rao(mat1.T, mat2.T).T
