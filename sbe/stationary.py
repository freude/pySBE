import numpy as np
import scipy
import matplotlib.pyplot as plt
from sbe.int_matrix import int_matrix_2D as int_matrix
from sbe.constants import h, e, m0, kb, eps0
from sbe.int_matrix import exchange


def plasmon_pole(q, omega, damp, omega_pl):

    eps = 1.0 + (omega_pl ** 2) / (np.abs(omega + 1j * damp)**2 - omega_pl ** 2)

    return eps


def ksi0(energy, omega, nh, ne, damp):
    """Zero-order electric susceptibility

    :param freq_array:
    :param omega:
    :param nh:
    :param ne:
    :param damp:
    :return:
    """

    if isinstance(energy, np.ndarray) or isinstance(energy, list):
        ans = np.zeros((len(energy), len(omega)), dtype=np.complex)
        for j, item in enumerate(energy):
            ans[j, :] = -(1.0 - ne - nh) / (energy[j] + 1j * damp - h * omega)
        return ans
    else:
        return -(1.0 - ne - nh) / (energy + 1j * damp - h * omega)


def vertex(energy, omega, nh, ne, damp, mu, V, measure):

    ans = np.zeros((len(energy), len(omega)), dtype=np.complex)

    for j, en in enumerate(energy):

        ksi = ksi0(en, omega, nh, ne, damp)

        M = np.diag(1.0 / ksi) - V * measure
        ans[j, :] = np.linalg.inv(M) @ mu
        # ans[j, :] = scipy.linalg.solve(M, mu.T)

    return ans


def polarization(fff, dim, params, bs, Ef_h, Ef_e, Tempr, V, VV1, VV2):

    # ----------------------- parse inputs -----------------------

    k = np.array(bs[0])
    Eh = bs[1] / h
    Ee = bs[2] / h
    mu = np.array(bs[3])
    # mu = mu - mu + mu[0]

    stk = k[4] - k[3]   # step in the k-space grid

    eps = params.eps
    n_reff = params.n_reff

    # ------------------------------------------------------------

    damp = 0.1 * e  # damping
    omega = Ee - Eh

    # ----------------- Distribution functions -------------------

    ne = 1.0 / (1 + np.exp((Ee * h - Ef_e) / kb / Tempr))
    nh = 1.0 / (1 + np.exp(-(Eh * h - Ef_h) / kb / Tempr))

    # --------------------------- Exchange energy ----------------

    # exce = exchange(k, ne, nh, V)
    # omega -= exce

    aaa = np.hstack((np.ones(len(k)), np.ones(len(k))*3))
    aaa1 = np.hstack((np.ones(len(k)), np.ones(len(k)) / 3))

    epsilon = 1.0
    # epsilon = plasmon_pole(0, omega[0], damp / h, omega[0]) * 2e-3
    # print(epsilon)
    V = V * epsilon

    M = vertex(fff*h, omega - omega[0], nh, ne, damp, mu, V, measure=stk)
    # p = np.imag(np.sum(M * np.tile(k ** 2, (len(fff), 1)), axis=1))
    p = 1.0 + 4 * np.pi * np.sum(M * np.tile(mu * aaa, (len(fff), 1)), axis=1) * stk
    # p = np.imag(np.sum(M, axis=1)) * stk
    return p / (4.0 * np.pi * eps0 * eps)


if __name__ == '__main__':

    l_k = 500
    wave_vector = np.linspace(0.0, 2.0e9, l_k)
    stk = wave_vector[3] - wave_vector[2]

    energy = np.linspace(-0.1, 0.1, l_k) * e

    omega = 1.0 * e / h + h*(wave_vector ** 2) / m0 / 0.5

    ne = 0.0
    nh = 0.0
    damp = 0.001 * e

    # V = int_matrix(wave_vector, 12)
    VV1 = np.pi * 1e9 ** 2

    from sbe.c import gen_pot
    V = gen_pot(wave_vector, 3, 3, 1e9, 2e9, 1e9, 2e9) / VV1*2
    #V=np.zeros((len(wave_vector), len(wave_vector)))

    plt.contourf(V)
    plt.show()

    ksi = ksi0(energy, omega - omega[0], nh, ne, damp)

    plt.contourf(np.abs(ksi))
    plt.show()

    plt.plot(np.sum(np.imag(ksi), axis=1))
    plt.show()

    mu = wave_vector - wave_vector + 1.0

    M = vertex(energy, omega - omega[0], nh, ne, damp, mu, V, measure=stk)

    # plt.contourf(np.abs(M))
    # plt.show()
    # plt.plot(p)
    plt.plot(energy, np.sum(np.imag(M) * np.tile(mu, (len(energy), 1)), axis=1))
    plt.show()
