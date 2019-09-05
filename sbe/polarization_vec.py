import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import constants as const
from int_matrix import exchange


def polarization(fff, params, bs, Ef_h, Ef_e, Tempr, conc, V, E_field):
    # ----------------------- parse inputs -----------------------

    k = bs[0]
    Eh = bs[1] / const.h
    Ee = bs[2] / const.h
    mu = bs[3]

    l_k = np.size(k)  # length of k array
    l_f = np.size(fff)  # length of frequency array
    l_t = 7000  # length of time array

    # --------------------- Material parameters -------------------

    eps = params.eps
    n_reff = params.n_reff
    vol = 1.0

    # ---------------------------- time -------------------------------

    t_min = 0.0  # min time
    t_max = 3.0e-12  # max time

    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # -----------------Many - particle characteristics - ----------------

    exce = np.zeros(l_k)  # Exchange energy

    # ------------------------- Polarization --------------------------

    A = np.zeros(l_k, dtype=np.complex)
    pp = np.zeros((l_t, l_k), dtype=np.complex)
    P = np.zeros(l_t, dtype=np.complex)
    E_ft = np.zeros(l_t, dtype=np.complex)

    # # ---------------- Formation of arrays ---------------------

    stk = k[4] - k[3]

    # --------------------------------------------------------

    damp = 0.0012 * const.e  # damping

    # ----------------------Plasma parameters-----------------

    # Ef_e=Eg+0.02 * e
    # Ef_h=-0.003 * e
    # Tempr=300
    # conc=1.0e14
    # conc=0.0

    # ---------------------Transition frequency----------------------
    omega = Ee - Eh

    # -----------------Distribution functions--------------------
    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 / (1 + np.exp((Eh * const.h - Ef_h) / const.kb / Tempr))

    Eg = const.h * omega[0]

    # concentration(k, l_k, 1.0-nh, conc1);
    # print *, 'concentration1=', conc1

    # -----------------------------------------------------------

    # call dielectrical_const(k, l_k, mh, me, 0.0 * minval(Ee), nh[1], ne[1], Tempr, conc, eps)

    # ----------------------Interaction matrix--------------------

    # call int_matrix(k, k1, l_k, eps, V, mh, me, Tempr, conc, ne[1], nh[1])
    exce = exchange(k, ne, 1.0 - nh, V)

    # -----------Solving paramsuctor Bloch equations------------

    line = 1
    # Tile k for vectorization

    for j2 in range(1, l_t):
        for j1 in range(l_k):
            A[j1] = trapz(V[j1, :] * pp[j2 - 1, :] * k, dx=stk)
        
            
        RS = -1j * (omega - Eg / const.h - exce / const.h) * pp[j2 - 1, :] - \
            1j * (ne - nh) * (mu * E_field(t[j2 - 1]) + A) / const.h - \
            damp * pp[j2 - 1][:] / const.h

        kk1 = RS

        kk2 = -1j * (omega - Eg / const.h - exce / const.h) * (pp[j2 - 1][:] + stt * kk1 / 2.0) - \
            1j * (ne - nh) * (mu * E_field(t[j2 - 1] + stt / 2) + A) / const.h - \
            damp * (pp[j2 - 1][:] + stt * kk1 / 2.0) / const.h

        kk3 = -1j * (omega - Eg / const.h - exce / const.h) * (pp[j2 - 1][:] + stt * kk2 / 2.0) - \
            1j * (ne - nh) * (mu * E_field(t[j2 - 1] + stt / 2) + A) / const.h - \
            damp * (pp[j2 - 1][:] + stt * kk2 / 2.0) /const. h

        kk4 = -1j * (omega - Eg / const.h - exce / const.h) * (pp[j2 - 1][:] + stt * kk3) - \
            1j * (ne - nh) * (mu * E_field(t[j2 - 1] + stt) + A) / const.h - \
            damp * (pp[j2 - 1][:] + stt * kk3) / const.h

        pp[j2, :] = pp[j2 - 1, :] + (stt / 6) * (kk1 + 2.0 * kk2 + 2.0 * kk3 + kk4)
        for j1 in range(l_k):
            P[j2] += 2.0 * const.pi / vol * mu[j1] * k[j1] * pp[j2][j1] * stk
            

        E_ft[j2] = E_field(t[j2 - 1])

        # if (j2 % 400) == 0.0:
        #     del line
        #     line, = plt.plot(np.real(P[:j2 - 1]))
        #     plt.pause(0.05)

    # ----------------- Fourier transformation ------------------
    #
    # ES = np.fft.fftshift(np.fft.fft(E_ft))
    # PS = np.fft.fftshift(np.fft.fft(P))
    # PSr = (fff + Eg / h) * np.imag(PS / ES) / (c * n_reff)

    ES = np.zeros(l_f, dtype=np.complex)
    PS = np.zeros(l_f, dtype=np.complex)
    PSr = np.zeros(l_f)

    for j in range(l_f):
        for j1 in range(l_t):
            ES[j] += E_ft[j1] * np.exp(1j * fff[j] * t[j1]) * stt
            PS[j] += P[j1] * np.exp(1j * fff[j] * t[j1]) * stt / (4.0 * const.pi * const.eps0 * eps)
        PSr[j] = (fff[j] + Eg / const.h) * np.imag(PS[j] / ES[j]) / (const.c * n_reff)

    # ---------------------- Visualization ----------------------

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    ax3 = fig.add_subplot(gs[1, 1:])

    ax2.plot(t / 1e-12, np.real(P)/np.max(np.abs(P)))
    ax2.plot(t / 1e-12, np.imag(P)/np.max(np.abs(P)))
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Polarization (a.u.)')

    ax1.imshow(np.real(pp[1000:0:-1, :]))
    ax1.axis('off')

    ax3.plot(fff * const.h / const.e / 0.0042, np.imag(PS / ES) / np.max(np.abs(PS / ES)))
    ax3.set_xlabel('Scaled energy (E-Eg)/Eb (a.u.)')
    ax3.set_ylabel('Absorption (a.u.)')
    plt.show()

    return PSr
