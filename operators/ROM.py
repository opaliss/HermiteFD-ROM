"""ROM operators

Author: Opal Issan (oissan@ucsd.edu)
Date: December 18th, 2024
"""
import numpy as np
import scipy


def theta_matrix(Nx, Nv):
    """construct sparse matrix Theta_{K} or Theta_{F}

    :param Nx: int, finite difference resolution
    :param Nv: int, Hermite resolution
    :return: sparse matrix K
    """
    theta = np.zeros((Nv * Nx, Nx))
    theta[:Nx, :Nx] = np.identity(Nx)
    return scipy.sparse.csr_matrix(theta)


def xi_matrix(Nx, Nv):
    """construct sparse matrix Xi_{F}

    :param Nx: int, finite difference resolution
    :param Nv: int, Hermite resolution
    :return: sparse matrix K
    """
    theta = np.zeros((Nv * Nx, Nx))
    theta[(Nv-1) * Nx:, :Nx] = np.identity(Nx)
    return scipy.sparse.csr_matrix(theta)
