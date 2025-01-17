"""Module includes temporal integrator and adaptivity

Authors: Opal Issan (oissan@ucsd.edu)
Version: March 5th, 2024
"""
import numpy as np
from scipy.optimize import newton_krylov
import scipy


def implicit_nonlinear_equation(y_new, y_old, dt, right_hand_side):
    """return the nonlinear equation for implicit midpoint to optimize.

    :param y_new: 1d array, y_{n+1}
    :param y_old: 1d array, y_{n}
    :param dt: float, time step t_{n+1} - t_{n}
    :param right_hand_side: function, a function of the rhs of the dynamical system dy/dt = rhs(y, t)
    :return: y_{n+1} - y_{n} -dt*rhs(y=(y_{n}+y_{n+1})/2, t=t_{n} + dt/2)
    """
    return y_new - y_old - dt * right_hand_side(y=0.5 * (y_old + y_new))


def implicit_midpoint_solver_ROM(y_0, right_hand_side, setup,  Nw=0, r_tol=1e-8, a_tol=1e-15, max_iter=100,
                                 windowing=False):
    """Solve the system

        dy/dt = rhs(y),    y(0) = y0,

    via the implicit midpoint method.

    The nonlinear equation at each time step is solved using Anderson acceleration.

    Parameters
    ----------
    :param setup: object of SimulationSetup with all the simulation setup parameters
    :param max_iter: maximum iterations of nonlinear solver, default is 100
    :param a_tol: absolute tolerance nonlinear solver, default is 1e-15
    :param r_tol: relative tolerance nonlinear solver, default is 1e-8
    :param y_0: initial condition
    :param windowing: boolean
    :param right_hand_side: function of the right-hand-side, i.e. dy/dt = rhs(y, t)

    Returns
    -------
    u: (Nx, Nt) ndarray
        Solution to the ODE at time t_vec; that is, y[:,j] is the
        computed solution corresponding to time t[j].

    """
    # initialize the solution matrix
    y_sol = np.zeros((len(y_0), len(setup.t_vec)))
    y_sol[:, 0] = y_0

    # absolute initial and final time
    Tf = setup.t_vec[-1]
    T0 = setup.t_vec[0]

    # for-loop each time-step
    for tt in range(1, len(setup.t_vec)):
        # print out the current time stamp
        print("\n time = ", setup.t_vec[tt])
        if windowing:
            if setup.t_vec[tt] in np.arange(T0 + int((Tf - T0) / Nw), Tf, int((Tf - T0) / Nw)) + setup.dt:
                # need to update the T0 and T parameters => new window has begun
                print("updating T0 and T :) ", setup.t_vec[tt])
                setup.T0 = int(np.floor(setup.t_vec[tt]))
                setup.T = int(np.floor(setup.t_vec[tt] + (Tf - T0) / Nw))

                # need to update the operators
                Ur_e1_next = np.load(
                    "../data/ROM/" + str(setup.problem_dir) + "/SVD/basis_SVD_e1_" + str(setup.T0) + "_" + str(
                        setup.T) + "_M_" + str(setup.M) + ".npy")[:, :setup.Nr]
                Ur_e2_next = np.load(
                    "../data/ROM/" + str(setup.problem_dir) + "/SVD/basis_SVD_e2_" + str(setup.T0) + "_" + str(
                        setup.T) + "_M_" + str(setup.M) + ".npy")[:, :setup.Nr]

                # project back up
                y_sol_e1 = setup.Ur_e1 @ y_sol[setup.NF: setup.NF + setup.Nr, tt - 1]
                y_sol_e2 = setup.Ur_e2 @ y_sol[2 * setup.NF + setup.Nr:, tt - 1]

                # # print projection error
                # print("e1 projection error = ", np.linalg.norm(y_sol_e1 - Ur_e1_next @ np.conjugate(Ur_e1_next).T @ y_sol_e1)/np.linalg.norm(y_sol_e1))
                # print("e2 projection error = ", np.linalg.norm(y_sol_e2 - Ur_e2_next @ np.conjugate(Ur_e2_next).T @ y_sol_e2)/np.linalg.norm(y_sol_e2))

                # need to re-project the initial condition
                # kinetic portion electron (1)
                y_sol[:, tt - 1][setup.NF: setup.NF + setup.Nr] = np.conjugate(Ur_e1_next).T @ y_sol_e1
                # kinetic portion electron (2)
                y_sol[:, tt - 1][2 * setup.NF + setup.Nr:] = np.conjugate(Ur_e2_next).T @ y_sol_e2

                # reload the operators and POD basis_20 for this window
                setup.Ur_e1 = Ur_e1_next
                setup.Ur_e2 = Ur_e2_next
                setup.load_operators()

        y_sol[:, tt] = newton_krylov(F=lambda y: implicit_nonlinear_equation(y_new=y,
                                                                             y_old=y_sol[:, tt - 1],
                                                                             right_hand_side=right_hand_side,
                                                                             dt=setup.dt),
                                     xin=y_sol[:, tt - 1],
                                     maxiter=max_iter,
                                     method='lgmres',
                                     f_tol=a_tol,
                                     f_rtol=r_tol,
                                     verbose=True)
    return y_sol
