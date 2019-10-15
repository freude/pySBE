import numpy as np
from scipy.integrate import simps
from sbe.constants import *


def int_matrix_2D(k, eps):
    """
    2D Coulomb potential in reciprocal space

    :param k:
    :param eps:
    :return:
    """

    l_k = np.size(k)
    V = np.zeros((l_k, l_k))

    l_phi = 3000
    phi = np.linspace(0, 2 * np.pi, l_phi, endpoint=False)              # angle array

    for j1 in range(l_k):
        for j2 in range(l_k):

            if j1 == j2:
                start = 1
            else:
                start = 0

            # angular integration
            q = np.sqrt(k[j1] ** 2 + k[j2] ** 2 - 2.0 * k[j1] * k[j2] * np.cos(phi[start:]))
            epss = 1.0
            integrand = epss * e ** 2 / (2.0 * ((2*np.pi) ** 2) * eps * eps0 * q)
            V[j1][j2] = k[j2] * simps(integrand, dx=np.abs(phi[2] - phi[1]))

    V[0, 0] = 0.0

    return V


def int_matrix_3D(k, eps):
    """
    3D Coulomb potential in reciprocal space

    :param k:
    :param eps:
    :return:
    """

    l_k = np.size(k)
    V = np.zeros((l_k, l_k))

    l_phi = 500
    theta = np.linspace(0, np.pi, l_phi // 2, endpoint=False)  # angle array

    for j1 in range(l_k):
        for j2 in range(l_k):

            if j1 == j2:
                start = 1
            else:
                start = 0

            # angular integration
            q = np.sqrt(k[j1] ** 2 + k[j2] ** 2 - 2.0 * k[j1] * k[j2] * np.cos(theta[start:]))
            epss = 1.0
            integrand = 2 * np.pi * np.sin(theta[start:]) * epss * e ** 2 / (((2*np.pi) ** 3) * eps * eps0 * (q ** 2))
            V[j1][j2] = k[j2] * k[j2] * simps(integrand, dx=np.abs(theta[2] - theta[1]))

    V[0, 0] = 0.0

    return V


def exchange(k, ne, nh, V):
    """
    Exchange energy

    :param k:
    :param eps:
    :return:
    """

    l_k = np.size(k)
    stk = k[3] - k[2]

    exce = np.zeros(l_k)

    for j in range(l_k):
        exce[j] = simps(V[j, :] * (ne + nh), dx=stk)

    return exce


def int_matrix(k, eps, dim=3):
    """
    Coulomb potential in reciprocal space

    :param k:
    :param eps:
    :return:
    """

    if dim == 3:
        return int_matrix_3D(k, eps)
    elif dim == 2:
        return int_matrix_2D(k, eps)
    else:
        raise ValueError("Wrong value of the parameter dim")
