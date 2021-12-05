import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sbe.semiconductors import BandStructure
import sbe.constants as const
from scipy.integrate import solve_ivp
import sbe.P_loop as P_f2py_loop
import sbe.fft_loop as fft_f2py_loop


def absorption(bs):
    k = np.linspace(0, 1e9, 50)
    subbands = bs.get_optical_transition_data(k, 0, 0)
    Eh = subbands[1] / const.h
    Ee = subbands[2] / const.h
    mu = subbands[3]

    l_k = np.size(k)  # length of k array

    fff = np.linspace(0.0, 1 * const.e / const.h, 100)
    l_t = 1000  # length of time array

    # ---------------------------- time -------------------------------

    t_min = 0.0  # min time
    t_max = 3.0e-10  # max time

    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # ------------------------- Polarization --------------------------

    A = np.zeros(l_k, dtype=np.complex)
    pp = np.zeros((l_t, l_k), dtype=np.complex)
    P = np.zeros(l_t, dtype=np.complex)
    E_ft = np.zeros(l_t, dtype=np.complex)

    # # ---------------- Formation of arrays ------------------

    stk = k[4] - k[3]

    # ---------------------------------------------------------

    damp = 0.01 * const.e  # damping

    # ---------------------Transition frequency----------------
    omega = Ee - Eh

    # -----------------Distribution functions------------------

    Ef_e = 0.1 * const.e
    Ef_h = 0.1 * const.e
    Tempr = 5

    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 - 1.0 / (1 + np.exp(-(Eh * const.h - Ef_h) / const.kb / Tempr))

    Eg = const.h * omega[0]

    def e_field(tt):
        pulse_widths = 0.09e-10
        pulse_delay = 0.5e-10

        a = np.exp(-((tt - pulse_delay) ** 2) / (2 * pulse_widths ** 2))  # * np.exp(1j * (bs.mat.Eg / const.h) * tt)

        return np.nan_to_num(a)

    # -----------Solving paramsuctor Bloch equations------------

    def rhs(pp, tt, omega, ne, nh, mu, damp, e_field):
        dydt = 1j * (omega - Eg / const.h) * pp + \
               1j * (ne - nh) * mu * e_field(tt) / const.h - \
               damp * pp / const.h

        return dydt

    line = 1
    # Tile k for vectorization

    y0 = np.zeros((l_k,), dtype=np.complex)

    sol = solve_ivp(rhs, (0, np.max(t)), y0, args=(omega, ne, nh, mu, damp, e_field),
                    t_eval=t, method='BDF', max_step=5e-12, first_step=2e-12, atol=1e-14)

    # if (j2 % 400) == 0.0:
    #     del line
    #     line, = plt.plot(np.real(P[:j2 - 1]))
    #     plt.pause(0.05)

    # ----------------- Fourier transformation ------------------
    #
    # ES = np.fft.fftshift(np.fft.fft(E_ft))
    # PS = np.fft.fftshift(np.fft.fft(P))
    # PSr = (fff + Eg / h) * np.imag(PS / ES) / (c * n_reff)

    # ES = np.zeros(l_f, dtype=np.complex)
    # PS = np.zeros(l_f, dtype=np.complex)
    # PSr = np.zeros(l_f)
    #
    # for j in range(l_f):
    #     for j1 in range(l_t):
    #         ES[j] += E_ft[j1] * np.exp(1j * fff[j] * t[j1]) * stt
    #         PS[j] += P[j1] * np.exp(1j * fff[j] * t[j1]) * stt / (4.0 * const.pi * const.eps0 * eps)
    #     PSr[j] = (fff[j] + Eg / const.h) * np.imag(PS[j] / ES[j]) / (const.c * n_reff)
    #
    # # ---------------------- Visualization ----------------------
    #
    # fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    # from matplotlib.gridspec import GridSpec
    #
    # gs = GridSpec(2, 3, figure=fig)
    # ax1 = fig.add_subplot(gs[:, 0])
    # ax2 = fig.add_subplot(gs[0, 1:])
    # ax3 = fig.add_subplot(gs[1, 1:])
    #
    # ax2.plot(t / 1e-12, np.real(P) / np.max(np.abs(P)))
    # ax2.plot(t / 1e-12, np.imag(P) / np.max(np.abs(P)))
    # ax2.set_xlabel('Time (ps)')
    # ax2.set_ylabel('Polarization (a.u.)')
    #
    # ax1.imshow(np.real(pp[1000:0:-1, :]))
    # ax1.axis('off')
    #
    # ax3.plot(fff * const.h / const.e / 0.0042, np.imag(PS / ES) / np.max(np.abs(PS / ES)))
    # ax3.set_xlabel('Scaled energy (E-Eg)/Eb (a.u.)')
    # ax3.set_ylabel('Absorption (a.u.)')
    # plt.show()

    plt.contourf(np.abs(sol.y))
    plt.show()

    return 1, 1, 1


def make_optical_data(k):

    class Graphene(object):
        def __init__(self):
            self.dim = 2
            self.Eg = 0  # nominal band gap
            self.eps = 1
            self.n_reff = 1

    graphene = Graphene()

    gamma0 = 3.03
    a = 2.46e-10
    gamma = np.sqrt(3) / 2 * gamma0 * a

    def ec(k):
        return gamma * k  # * k * k * k / 1e28

    def ev(k):
        return -gamma * k  # * k * k * k / 1e28

    def dip(k):
        return 1e20 / (k + 1e-15) + 1e10 * 0

    bs = BandStructure(mat=graphene, cond_band=[ec], val_band=[ev], dipoles=[[dip]])

    # ----------------------- parse inputs -----------------------

    data = bs.get_optical_transition_data(k, 0, 0)
    Eh = data[1] / const.h
    Ee = data[2] / const.h
    mu = np.array(data[3])

    return Eh, Ee, mu, graphene


def absorption_graphene(fff, Tempr, Ef_h, Ef_e, damp, vis=False):

    k = np.linspace(0, 0.3e9, 300)
    Eh, Ee, mu, graphene = make_optical_data(k)

    l_k = np.size(k)  # length of k array
    # length of time array
    stk = k[4] - k[3]  # step in the k-space grid

    eps = graphene.eps
    n_reff = graphene.n_reff

    # -------------------------- time ----------------------------

    t_min = 0.0  # min time
    t_max = 1e-12  # max time
    l_t = 30000
    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # ------------------------------------------------------------

    omega = Ee - Eh
    Eg = const.h * omega[0]

    # ----------------- Distribution functions -------------------

    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 / (1 + np.exp(-(Eh * const.h - Ef_h) / const.kb / Tempr))

    # --------------------------- Exchange energy ----------------

    # exce = exchange(k, ne, nh, V)
    exce = np.zeros((l_k,))
    V = np.zeros((l_k, l_k))

    dim = graphene.dim
    damp *= const.e
    pulse_delay = 1e-13
    pulse_widths = 0.01e-13
    pulse_amp = 1e-31
    e_phot = 0

    # Call the Fortran routine to caluculate the required arrays.
    P, pp, ne_k, nh_k = P_f2py_loop.loop(dim, l_t, l_k, t, k, stt, stk,
                                         omega, Eg, exce, ne, nh,
                                         np.abs(mu), damp, const.h, V,
                                         pulse_delay, pulse_widths, pulse_amp, e_phot)

    l_f = np.size(fff)  # length of frequency array

    def e_field(tt):
        a = pulse_amp * np.exp(-((tt - pulse_delay) ** 2) / (2 * pulse_widths ** 2))  # * np.exp(1j * (bs.mat.Eg / const.h) * tt)
        return np.nan_to_num(a)

    E_ft = e_field(t)

    PSr, PSi = fft_f2py_loop.loop(E_ft, l_t, l_f, stt, t, fff,
                                  P, const.eps0, eps, Eg, const.h,
                                  const.c, n_reff)

    # norm = np.max(PSi)
    # PSr /= norm
    # PSi /= norm

    sus = PSr + 1j*PSi

    # ---------------------- Visualization ----------------------
    if vis:
        plt.contourf(k / 1e9, t[l_k * 20: 0: -1] / 1e-12, np.abs(pp[l_k * 20: 0: -1, :]), 100, cmap='seismic')
        plt.xlabel(r'Wave vector (nm$^{-1}$)')
        plt.ylabel('Time (ps)')
        plt.title('Absolute value of the microscopic polarization')
        plt.colorbar()
        plt.show()

        plt.contourf(k / 1e9, t[l_k * 20: 0: -1] / 1e-12, np.real(pp[l_k * 20: 0: -1, :]), 100,
                     cmap='seismic',
                     norm=colors.TwoSlopeNorm(0))
        plt.xlabel(r'Wave vector (nm$^{-1}$)')
        plt.ylabel('Time (ps)')
        plt.title('Real part of the microscopic polarization')
        plt.colorbar()
        plt.show()

        plt.plot(fff * const.h / const.e, np.real(sus), 'k')
        plt.plot(fff * const.h / const.e, np.imag(sus), 'k--')
        plt.xlabel('Photon energy (eV)')
        plt.ylabel('Susceptibility (a.u.)')
        plt.title("Dielectric susceptibility, $\chi(\omega)$ @ T=" + str(Tempr) + " K, Ef=" + str(Ef_e/const.e) + " eV")
        plt.legend(['$\Re[\chi(\omega)]$', '$\Im[\chi(\omega)]$'])
        plt.show()

    cond = -1j * sus * fff
    # norm = np.max(fff)
    # cond *= norm

    return cond


def absorption_graphene_stationary(fff, Tempr, Ef_h, Ef_e, damp, vis=False):

    k = np.linspace(0, 0.3e9, 300)
    Eh, Ee, mu, graphene = make_optical_data(k)

    l_k = np.size(k)  # length of k array
    # length of time array
    stk = k[4] - k[3]  # step in the k-space grid

    eps = graphene.eps
    n_reff = graphene.n_reff

    # ------------------------------------------------------------

    omega = Ee - Eh

    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 / (1 + np.exp(-(Eh * const.h - Ef_h) / const.kb / Tempr))
    damp *= const.e

    from sbe.stationary import ksi0, vertex
    V = np.zeros((l_k, l_k))

    # ksi = ksi0(fff*const.h, omega - omega[0], nh, ne, damp)
    #
    # plt.contourf(np.abs(ksi))
    # plt.show()
    #
    # plt.plot(np.sum(np.imag(ksi), axis=1))
    # plt.show()

    M = vertex(fff * const.h, omega - omega[0], nh, ne, damp, mu, V, 0)
    cond = np.sum(M * np.tile(mu * k, (len(fff), 1)), axis=1) * stk / (2 * np.pi) / (4.0 * np.pi * const.eps0 * eps)

    # norm = np.max(np.imag(cond))
    # cond /= norm
    return -1j * cond * fff


def main():

    Ef_e = 0.05 * const.e
    Ef_h = -0.05 * const.e
    Tempr = 10
    damp = 0.003

    fff = np.linspace(0.0, 0.3 * const.e / const.h, 100)
    cond = absorption_graphene_stationary(fff, Tempr, Ef_h, Ef_e, damp, vis=True)
    cond1 = absorption_graphene(fff, Tempr, Ef_h, Ef_e, damp, vis=True)

    plt.figure(1)
    plt.plot(fff * const.h / const.e, np.real(cond), 'k')
    plt.plot(fff * const.h / const.e, np.imag(cond), 'k--')
    plt.plot(fff * const.h / const.e, np.real(cond1), 'r')
    plt.plot(fff * const.h / const.e, np.imag(cond1), 'r--')
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Conductivity (a.u.)')
    plt.title("Conductivity, $\sigma(\omega)$ @ T=" + str(Tempr) + " K, Ef=" + str(Ef_e/const.e) + " eV")
    plt.legend(['$\Re[\sigma(\omega)]$', '$\Im[\sigma(\omega)]$'])

    # plt.figure(2)
    # plt.plot(fff * const.h / const.e, np.real(cond1), 'r')
    # plt.plot(fff * const.h / const.e, np.imag(cond1), 'r--')
    # plt.xlabel('Photon energy (eV)')
    # plt.ylabel('Conductivity (a.u.)')
    # plt.title("Conductivity, $\sigma(\omega)$ @ T=" + str(Tempr) + " K, Ef=" + str(Ef_e/const.e) + " eV")
    # plt.legend(['$\Re[\sigma(\omega)]$', '$\Im[\sigma(\omega)]$'])
    plt.show()


if __name__ == '__main__':
    main()
