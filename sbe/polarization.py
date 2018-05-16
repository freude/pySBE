import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
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
    l_t = 5500  # length of time array

    # -------------------- Material parameters -------------------

    eps = params.eps
    Eg = params.Eg
    me = params.me
    mh = params.mh
    n_reff = params.n_reff
    vol = 1.0

    # ----------------------------time - ------------------------------

    t_min = 0.0  # min time
    t_max = 0.5e-12  # max time

    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # -----------------Many - particle characteristics - ----------------

    exce = np.zeros(l_k)  # Exchange energy

    # -------------------------Polarization - -------------------------

    A = np.zeros(l_k)
    pp = np.zeros((l_t, l_k))
    P = np.zeros(l_t)
    PS = np.zeros(l_f)
    E_ft = np.zeros(l_t)

    # # ----------------Formation of arrays - --------------------

    stk = k[4] - k[3]

    # --------------------------------------------------------

    damp = 0.0005 * e  # damping

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
    exce = exchange(k, ne, nh, V)

    # -----------Solving paramsuctor Bloch equations------------

    for j2 in xrange(1, l_t):
        for j1 in xrange(l_k):
            A[j1] = simps(V[j1, :] * pp[j2 - 1, :] * k, dx=stk)

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
            print("{}: pp={} ne={} nh={} exce={} A={}".format(j2, pp[j2, j1], ne[j1], nh[j1], exce[j1], A[j1]))

        E_ft[j2] = E_field(t[j2 - 1])

        if j2 == 50:
            line1, = plt.plot(P[:j2 - 1])
        elif j2 > 50 and (j2 % 50) == 0.0:
            line1.set_ydata(P[:j2 - 1])
            plt.draw()
            plt.pause(0.05)

    # ----------------- Fourier transformation ------------------

    ES = np.zeros(l_t)
    PS = np.zeros(l_t)
    PSr = np.zeros(l_f)

    for j in xrange(l_f):
        for j1 in xrange(l_t):
            ES[j] += E_ft[j1] * np.exp(1j * (fff[j] - Eg / h) * t[j1]) * stt
            PS[j] += P[j1] * np.exp(1j * (fff[j] - Eg / h) * t[j1]) * stt / (4.0 * pi * eps0 * eps)
        PSr[j] = (fff[j] + Eg / h) * np.imag(PS[j] / ES[j]) / (c * n_reff)

    return PSr
