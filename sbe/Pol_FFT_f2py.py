import numpy as np
import matplotlib.pyplot as plt
import constants as const
from int_matrix import exchange
import P_loop as P_f2py_loop
import fft_loop as fft_f2py_loop


def polarization(fff, dim, params, bs, Ef_h, Ef_e, Tempr, V, E_field, pulse_widths, pulse_delay, pulse_amp, e_phot, debug):

    # ----------------------- parse inputs -----------------------

    k = np.array(bs[0])
    Eh = bs[1] / const.h
    Ee = bs[2] / const.h
    mu = np.array(bs[3])

    l_k = np.size(k)    # length of k array
    l_f = np.size(fff)  # length of frequency array
    l_t = 25000         # length of time array
    stk = k[4] - k[3]   # step in the k-space grid

    eps = params.eps
    n_reff = params.n_reff

    # -------------------------- time ----------------------------

    t_min = 0.0  # min time
    t_max = 1.2e-11  # max time
    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # ------------------------------------------------------------

    damp = 0.003 * const.e  # damping
    omega = Ee - Eh
    Eg = const.h * omega[0]

    # ----------------- Distribution functions -------------------

    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 / (1 + np.exp(-(Eh * const.h - Ef_h) / const.kb / Tempr))

    # --------------------------- Exchange energy ----------------

    exce = exchange(k, ne, nh, V)

    # Call the Fortran routine to caluculate the required arrays.

    P_f2py_loop.loop(dim, l_t, l_k, t, k, stt, stk,
                     omega, Eg, exce, ne, nh,
                     mu, damp, const.h, V,
                     pulse_delay, pulse_widths, pulse_amp, e_phot)

    # Read the arrays from the drive which are saved by the above Fortran Routine.

    pp_real = np.transpose((np.fromfile('pp_real', dtype='float64')).reshape(l_k, l_t))
    pp_imag = np.transpose((np.fromfile('pp_imag', dtype='float64')).reshape(l_k, l_t))
    P_real = np.fromfile('P_real', dtype='float64')
    P_imag = np.fromfile('P_imag', dtype='float64')

    pp = pp_real + 1j * pp_imag
    P = P_real + 1j * P_imag

    E_ft = E_field(t)
    ES = np.zeros(l_f, dtype=np.complex)
    PS = np.zeros(l_f, dtype=np.complex)
    PSr = np.zeros(l_f)

    print("===========FFT Begins=========")

    fft_f2py_loop.loop(E_ft, l_t, l_f, stt, t, fff,
                   P, const.eps0, eps, Eg, const.h,
                   const.c, n_reff)
    
    # for j in range(l_f):
    #     for j1 in range(l_t):
    #         ES[j] += E_ft[j1] * np.exp(1j * fff[j] * t[j1]) * stt
    #         PS[j] += P[j1] * np.exp(1j * fff[j] * t[j1]) * stt / (4.0 * const.pi * const.eps0 * eps)
    #     PSr[j] = 4 * np.pi * (fff[j] + Eg / const.h) * np.imag(PS[j] / ES[j]) / (const.c * n_reff)

    ES_real = np.fromfile('ES_real', dtype='float64')
    ES_imag = np.fromfile('ES_imag', dtype='float64')
    PS_real = np.fromfile('PS_real', dtype='float64')
    PS_imag = np.fromfile('PS_imag', dtype='float64')

    ne_k = np.transpose(np.fromfile('ne', dtype='float64').reshape(l_k, l_t))
    nh_k = np.transpose(np.fromfile('nh', dtype='float64').reshape(l_k, l_t))

    PSr = np.fromfile('PSr', dtype='float64')

    PS = PS_real + 1j * PS_imag
    ES = ES_real + 1j * ES_imag

    # ---------------------- Visualization ----------------------

    if debug:


        fig = plt.figure(figsize=(11, 7), constrained_layout=True)
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(3, 5, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        ax3 = fig.add_subplot(gs[:, 2])
        ax4 = fig.add_subplot(gs[1, 3:])
        ax5 = fig.add_subplot(gs[2, 3:])
        ax6 = fig.add_subplot(gs[0, 3:])

        ax6.plot(t / 1e-12, np.real(E_ft) / np.max(np.abs(E_ft)))
        ax6.plot(t / 1e-12, np.imag(E_ft) / np.max(np.abs(E_ft)))
        ax6.set_xlabel('Time (ps)')
        ax6.set_ylabel('Pump signal (a.u.)')

        ax4.plot(t / 1e-12, np.real(P) / np.max(np.abs(P)))
        ax4.plot(t / 1e-12, np.imag(P) / np.max(np.abs(P)))
        ax4.set_xlabel('Time (ps)')
        ax4.set_ylabel('Polarization (a.u.)')

        ax3.contourf(k/1e9, t[l_k * 4: 0: -1]/1e-12, np.real(pp[l_k * 4: 0: -1, :]), 100)
        ax3.set_xlabel(r'Wave vector (nm$^{-1}$)')
        ax3.set_ylabel('Time (ps)')
        ax3.title.set_text('Microscopic \n polarization')
        # ax3.axis('off')

        # ax2.imshow(ne_k[l_k * 4: 0: -1, :])
        ax2.contourf(k / 1e9, t[l_k * 4: 0: -1] / 1e-12, np.real(ne_k[l_k * 4: 0: -1, :]), 100)
        ax2.set_xlabel(r'Wave vector (nm$^{-1}$)')
        ax2.set_ylabel('Time (ps)')
        ax2.title.set_text('Electron \n distribution')
        # ax2.axis('off')

        # ax1.imshow(nh_k[l_k * 4: 0: -1, :])
        ax1.contourf(k / 1e9, t[l_k * 4: 0: -1] / 1e-12, np.real(nh_k[l_k * 4: 0: -1, :]), 100)
        ax1.set_xlabel(r'Wave vector (nm$^{-1}$)')
        ax1.set_ylabel('Time (ps)')
        ax1.title.set_text('Hole \n distribution')
        # ax1.axis('off')

        ax5.plot(fff * const.h / const.e / 0.0042, PSr / np.max(PSr))
        ax5.set_xlabel('Scaled energy (E-Eg)/Eb (a.u.)')
        ax5.set_ylabel('Absorption (a.u.)')

        plt.pause(1)
        plt.draw()

    return PSr
