import numpy as np
from scipy.integrate import simps
from constants import *


def int_matrix(k, eps):

    l_k = np.size(k)
    V = np.zeros((l_k, l_k))

    l_phi = 3000
    phi = np.linspace(0, 2 * np.pi, l_phi, endpoint=False)              # angle array

    for j1 in xrange(l_k):
        for j2 in xrange(l_k):

            if j1 == j2:
                start = 1
            else:
                start = 0

            # angular integration
            q = np.sqrt(k[j1] ** 2 + k[j2] ** 2 - 2.0 * k[j1] * k[j2] * np.cos(phi[start:]))
            epss = 1.0
            integrand = epss * e ** 2 / (8.0 * (np.pi ** 2) * eps * eps0 * q)
            V[j1][j2] = simps(integrand, dx=np.abs(phi[2] - phi[1]))

    V[0, 0] = 0.0

    return V


def exchange(k, ne, nh, V):

    l_k = np.size(k)
    stk = k[3] - k[2]

    exce = np.zeros(l_k)

    for j in range(l_k):
        exce[j] = simps(V[j, :] * (ne + nh) * k, dx=stk)

    return exce