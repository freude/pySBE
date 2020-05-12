import numpy as np
from scipy.integrate import simps
from sbe.constants import *


def pot1D(a):
    from scipy.special import sici
    si, ci = sici(a)
    return np.sin(a) * (np.pi - 2 * si) - 2 * np.cos(a) * ci


def int_matrix_anisotr(k, eps1, eps2, dist, k_lim1, k_lim2, R=0.4e-9):
    """
    1D Coulomb potential in reciprocal space

    :param k:
    :param eps:
    :return:
    """

    l_k = np.size(k)
    V1 = np.zeros((l_k, l_k))
    V2 = np.zeros((l_k, l_k))
    V3 = np.zeros((l_k, l_k))

    for j1 in range(l_k):
        for j2 in range(l_k):
            # angular integration
            q = np.abs(k[j1] - k[j2])
            V1[j1][j2] = np.log(1.0 + eps1 / eps2 * (k_lim1 ** 2) / (q ** 2)) / eps1
            V2[j1][j2] = np.log((eps2 * q ** 2 + eps1 * k_lim2 ** 2) / (eps2 * q ** 2 + eps1 * k_lim1 ** 2)) / eps1
            V3[j1][j2] = 0.1e19 / (eps2 * q ** 2 + eps1 * dist ** 2)
            # if j1 == j2:
            #     V[j1, j2] = 1.0

    V1[0, 0] = 0.0
    epss = 1.0

    for j1 in range(1, l_k):
        V1[j1, j1] = V1[j1, j1-1]*5

    return V1 * R * epss * e ** 2 / (4.0 * np.pi * (2*np.pi) * eps0),\
           V2 * R * epss * e ** 2 / (4.0 * np.pi * (2*np.pi) * eps0),\
           V3 * R * epss * e ** 2 / (4.0 * np.pi * (2*np.pi) * eps0)


def int_matrix_1D(k, eps, R=0.4e-9):
    """
    1D Coulomb potential in reciprocal space

    :param k:
    :param eps:
    :return:
    """

    l_k = np.size(k)
    V = np.zeros((l_k, l_k))

    for j1 in range(l_k):
        for j2 in range(l_k):
            # angular integration
            q = np.abs(k[j1] - k[j2])
            q1 = np.abs(k[j1] + k[j2])
            # V[j1][j2] = pot1D(q*R) * np.exp(-0*1.5*q*R) + pot1D(q1*R) * np.exp(-0*1.5*q*R)
            V[j1][j2] = np.log(1.0+R/(q**2))/5
            # if j1 == j2:
            #     V[j1, j2] = 1.0

    V[0, 0] = 0.0
    epss = 1.0

    for j1 in range(1, l_k):
        V[j1, j1] = V[j1, j1-1] * 2

    return V * epss * e ** 2 / (4.0 * np.pi * (2*np.pi) * eps * eps0)


def int_matrix_aniso(k, eps, dim=1):
    """
    1D Coulomb potential in reciprocal space

    :param k:
    :param eps:
    :return:
    """

    l_k = np.size(k)
    V = np.zeros((l_k, l_k))

    for j1 in range(l_k):
        for j2 in range(l_k):
            q = np.abs(k[j1] - k[j2])
            if j1 == j2:
                V[j1][j2] = 0
            else:
                A = 0.6e9
                V[j1][j2] = 4 * np.pi * np.log(A / (q) + 1.0) * e ** 2 / (2.0 * ((2 * np.pi) ** 2) * eps * eps0)

    return V


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


def int_matrix(k, eps, dim=3, R=0):
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
    elif dim == 1:
        R = 0.15e-9
        return int_matrix_1D(k, eps, R)
    else:
        raise ValueError("Wrong value of the parameter dim")


if __name__ == '__main__':

    b = 1
    a = np.linspace(0, 5, 1000)

    import matplotlib.pyplot as plt
    from scipy.special import kv
    # f1 = np.sin(a*b)*(np.pi - 2*si)-0*2*np.cos(a*b)*ci
    # f2 = 0*np.sin(a * b) * (np.pi - 2 * si) - 2 * np.cos(a * b) * ci
    f3_0 = pot1D(a*b)
    f3_1 = pot1D(a*b*2)
    f3_2 = pot1D(a*b*3)
    f4 = kv(0, 0.5*a * b)
    f5 = -2 * 0.5772156649015328606065120900824024310421 + np.log(-1.0/((a*b)**2))
    f7 = np.pi/b * np.exp(-b*a)
    f8 = -np.log(a*b) - 0.5772156649015328606065120900824024310421

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    # plt.plot(f1)
    # plt.plot(f2)
    # plt.plot(np.log(f3))
    # plt.plot(np.log(f4))
    # plt.plot(np.log(f5))

    ax.plot(a, f3_0)
    # plt.plot(f3_1)
    # plt.plot(f3_2)
    # plt.plot(1.0/(a*b*30))
    ax.plot(a, f4)
    # plt.plot(f5)
    ax.plot(a, f8)

    ax.legend(['1', '2', '3'])
    ax.set_xlabel('Wave vector (a.u)')
    ax.set_ylabel('Potential (a.u)')

    ax1 = fig.add_subplot(2, 1, 2)

    # plt.plot(f1)
    # plt.plot(f2)
    # plt.plot(np.log(f3))
    # plt.plot(np.log(f4))
    # plt.plot(np.log(f5))

    ax1.plot(a, f3_0)
    # plt.plot(f3_1)
    # plt.plot(f3_2)
    # plt.plot(1.0/(a*b*30))
    ax1.plot(a, f4)
    # plt.plot(f5)
    ax1.plot(a, f7)
    ax1.set_xlabel('Wave vector (a.u)')
    ax1.set_ylabel('Potential (a.u)')
    ax1.set_yscale('log')
    ax1.legend(['1', '2', '3'])

    plt.show()
