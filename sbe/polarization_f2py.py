import numpy as np
import matplotlib.pyplot as plt
import sbe.constants as const
from sbe.int_matrix import exchange
import sbe.P_loop as P_f2py_loop
import sbe.oscillators as osc


def polarization(fff, dim, params, bs, Ef_h, Ef_e, Tempr, V, E_field, pulse_widths, pulse_delay, pulse_amp, e_phot, debug=True):

    # ----------------------- parse inputs -----------------------

    k = np.array(bs[0])
    Eh = bs[1] / const.h
    Ee = bs[2] / const.h
    mu = np.array(bs[3])

    l_k = np.size(k)    # length of k array
    l_f = np.size(fff)  # length of frequency array
    l_t = 20000         # length of time array
    stk = k[4] - k[3]   # step in the k-space grid

    eps = params.eps
    n_reff = params.n_reff

    # -------------------------- time ----------------------------

    t_min = 0.0  # min time
    t_max = 0.5e-12  # max time
    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # ------------------------------------------------------------

    damp = 2.003 * const.e  # damping
    omega = Ee - Eh
    Eg = const.h * omega[0]

    # ----------------- Distribution functions -------------------

    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 / (1 + np.exp(-(Eh * const.h - Ef_h) / const.kb / Tempr))

    # --------------------------- Exchange energy ----------------

    exce = exchange(k, ne, nh, V)

    # Call the Fortran routine to caluculate the required arrays.

    # P_f2py_loop.loop(dim, l_t, l_k, t, k, stt, stk,
    #                  omega, Eg, exce, ne, 1.0 - nh,
    #                  mu, damp, const.h, V,
    #                  pulse_delay, pulse_widths, pulse_amp)

    pp, ne_k, nh_k = osc.oscilators(t, l_t, k, l_k, omega, ne, nh, mu, damp, V,
                                    pulse_delay, pulse_widths, pulse_amp, e_phot)

        (dim, l_t, l_k, t, k, stt, stk,
                     omega, Eg, exce, ne, 1.0 - nh,
                     mu, damp, const.h, V,
                     pulse_delay, pulse_widths, pulse_amp)

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

    for j in range(l_f):
        for j1 in range(l_t):
            ES[j] += E_ft[j1] * np.exp(1j * fff[j] * t[j1]) * stt
            PS[j] += P[j1] * np.exp(1j * fff[j] * t[j1]) * stt / (4.0 * const.pi * const.eps0 * eps)
        PSr[j] = 4 * np.pi * (fff[j] + Eg / const.h) * np.imag(PS[j] / ES[j]) / (const.c * n_reff)

    # ---------------------- Visualization ----------------------

    figs = []

    if debug:

        if len(figs) > 0:
            print('hi')

        figs.append(plt.figure(figsize=(11, 7), constrained_layout=True))
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(2, 3, figure=figs[-1])
        ax1 = figs[-1].add_subplot(gs[:, 0])
        ax2 = figs[-1].add_subplot(gs[0, 1:])
        ax3 = figs[-1].add_subplot(gs[1, 1:])

        ax2.plot(t / 1e-12, np.real(P) / np.max(np.abs(P)))
        ax2.plot(t / 1e-12, np.imag(P) / np.max(np.abs(P)))
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Polarization (a.u.)')

        ax1.imshow(np.real(pp[l_k * 5: 0: -1, :]))
        ax1.axis('off')

        ax3.plot(fff * const.h / const.e / 0.0042, PSr / np.max(PSr))
        ax3.set_xlabel('Scaled energy (E-Eg)/Eb (a.u.)')
        ax3.set_ylabel('Absorption (a.u.)')

        plt.draw()

    return PSr
