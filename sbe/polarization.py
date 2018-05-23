import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from constants import *
from int_matrix import exchange


def polarization(fff, params, bs, Ef_h, Ef_e, Tempr, conc, V, E_field):
    # ----------------------- parse inputs -----------------------

    k = bs[0]
    Eh = bs[1] / h
    Ee = bs[2] / h
    mu = bs[3]

    l_k = np.size(k)  # length of k array
    l_f = np.size(fff)  # length of frequency array
    l_t = 10000  # length of time array

    # -------------------- Material parameters -------------------

    eps = params.eps
    n_reff = params.n_reff
    vol = 1.0

    # ----------------------------time - ------------------------------

    t_min = 0.0  # min time
    t_max = 3.0e-12  # max time

    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # -----------------Many - particle characteristics - ----------------

    exce = np.zeros(l_k)  # Exchange energy

    # -------------------------Polarization - -------------------------

    A = np.zeros(l_k, dtype=np.complex)
    pp = np.zeros((l_t, l_k), dtype=np.complex)
    P = np.zeros(l_t, dtype=np.complex)
    E_ft = np.zeros(l_t, dtype=np.complex)

    # # ----------------Formation of arrays - --------------------

    stk = k[4] - k[3]

    # --------------------------------------------------------

    damp = 0.0012 * e  # damping

    # ----------------------Plasma parameters-----------------

    # Ef_e=Eg+0.02 * e
    # Ef_h=-0.003 * e
    # Tempr=300
    # conc=1.0e14
    # conc=0.0

    # ---------------------Transition frequency----------------------
    omega = Ee - Eh

    # -----------------Distribution functions--------------------
    ne = 1.0 / (1 + np.exp((Ee * h - Ef_e) / kb / Tempr))
    nh = 1.0 / (1 + np.exp((Eh * h - Ef_h) / kb / Tempr))

    Eg = h * omega[0]

    # concentration(k, l_k, 1.0-nh, conc1);
    # print *, 'concentration1=', conc1

    # -----------------------------------------------------------

    # call dielectrical_const(k, l_k, mh, me, 0.0 * minval(Ee), nh[1], ne[1], Tempr, conc, eps)

    # ----------------------Interaction matrix--------------------

    # call int_matrix(k, k1, l_k, eps, V, mh, me, Tempr, conc, ne[1], nh[1])
    exce = exchange(k, ne, 1.0 - nh, V)

    # -----------Solving paramsuctor Bloch equations------------

    line = 1

    for j2 in xrange(1, l_t):
        for j1 in xrange(l_k):
            A[j1] = trapz(V[j1, :] * pp[j2 - 1, :] * k, dx=stk)

            RS = -1j * (omega[j1] - Eg / h - exce[j1] / h) * pp[j2 - 1, j1] - \
                 1j * (ne[j1] - nh[j1]) * (mu[j1] * E_field(t[j2 - 1]) + A[j1]) / h - \
                 damp * pp[j2 - 1][j1] / h

            kk1 = RS

            kk2 = -1j * (omega[j1] - Eg / h - exce[j1] / h) * (pp[j2 - 1][j1] + stt * kk1 / 2.0) - \
                  1j * (ne[j1] - nh[j1]) * (mu[j1] * E_field(t[j2 - 1] + stt / 2) + A[j1]) / h - \
                  damp * (pp[j2 - 1][j1] + stt * kk1 / 2.0) / h

            kk3 = -1j * (omega[j1] - Eg / h - exce[j1] / h) * (pp[j2 - 1][j1] + stt * kk2 / 2.0) - \
                  1j * (ne[j1] - nh[j1]) * (mu[j1] * E_field(t[j2 - 1] + stt / 2) + A[j1]) / h - \
                  damp * (pp[j2 - 1][j1] + stt * kk2 / 2.0) / h

            kk4 = -1j * (omega[j1] - Eg / h - exce[j1] / h) * (pp[j2 - 1][j1] + stt * kk3) - \
                  1j * (ne[j1] - nh[j1]) * (mu[j1] * E_field(t[j2 - 1] + stt) + A[j1]) / h - \
                  damp * (pp[j2 - 1][j1] + stt * kk3) / h

            pp[j2, j1] = pp[j2 - 1, j1] + (stt / 6) * (kk1 + 2.0 * kk2 + 2.0 * kk3 + kk4)
            P[j2] += 2.0 * pi / vol * mu[j1] * k[j1] * pp[j2][j1] * stk
            # print("{}: pp={:.2E} ne={:.2E} nh={:.2E} exce={:.2E} A={:.2E}".format(j2,
            #                                                                       pp[j2, j1],
            #                                                                       ne[j1], nh[j1],
            #                                                                       exce[j1], A[j1]))

        E_ft[j2] = E_field(t[j2 - 1])

        if (j2 % 400) == 0.0:
            del line
            line, = plt.plot(np.real(P[:j2 - 1]))
            plt.pause(0.05)

    # ----------------- Fourier transformation ------------------
    #
    # ES = np.fft.fftshift(np.fft.fft(E_ft))
    # PS = np.fft.fftshift(np.fft.fft(P))
    # PSr = (fff + Eg / h) * np.imag(PS / ES) / (c * n_reff)

    ES = np.zeros(l_f, dtype=np.complex)
    PS = np.zeros(l_f, dtype=np.complex)
    PSr = np.zeros(l_f)

    for j in xrange(l_f):
        for j1 in xrange(l_t):
            ES[j] += E_ft[j1] * np.exp(1j * fff[j] * t[j1]) * stt
            PS[j] += P[j1] * np.exp(1j * fff[j] * t[j1]) * stt / (4.0 * pi * eps0 * eps)
        PSr[j] = (fff[j] + Eg / h) * np.imag(PS[j] / ES[j]) / (c * n_reff)

    return PSr
