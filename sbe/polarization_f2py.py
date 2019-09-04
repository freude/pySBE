import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import constants as const
from int_matrix import exchange

import P_loop as P_f2py_loop

def polarization(fff, params, bs, Ef_h, Ef_e, Tempr, conc, V, E_field):
    # ----------------------- parse inputs -----------------------

    k = np.array(bs[0])
    Eh = bs[1] / const.h
    Ee = bs[2] / const.h
    mu = np.array(bs[3])

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

    pulse_widths = 0.15e-14
    pulse_delay = 10 * pulse_widths


    # -----------------Many - particle characteristics - ----------------

    exce = np.zeros(l_k)  # Exchange energy

    # ------------------------- Polarization --------------------------

    A = np.zeros(l_k, dtype=np.complex)
    #pp = np.zeros((l_t, l_k), dtype=np.complex)
    #P = np.zeros(l_t, dtype=np.complex)
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

    # Call the Fortran routine to caluculate the required arrays.

    P_f2py_loop.loop(l_t,l_k,t,k,stt,stk,
                     omega,Eg,exce,ne,nh,
                     mu,damp,const.h,V,
                     pulse_delay,pulse_widths)

    # Read the arrays from the drive which are saved by the above Fortran Routine.
    pp_real = np.transpose( (np.fromfile('pp_real',dtype='float64')).reshape(l_k,l_t) )
    pp_imag = np.transpose( (np.fromfile('pp_imag',dtype='float64')).reshape(l_k,l_t) )

    P_real = ( (np.fromfile('P_real',dtype='float64'))  )
    P_imag = ( (np.fromfile('P_imag',dtype='float64'))  )

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
