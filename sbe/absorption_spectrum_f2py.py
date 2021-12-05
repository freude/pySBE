import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lgmres
import matplotlib.pyplot as plt
import logging
from sbe.constants import h, e, c, eps0
from sbe.int_matrix import int_matrix_1D as int_matrix
from sbe.int_matrix import int_matrix_anisotr
from sbe.stationary import polarization as polarization1
from sbe.stationary import vertex1
from sbe.Pol_FFT_f2py import polarization
from sbe.semiconductors import GaAs, Tc
from sbe.semiconductors import BandStructure3D, get_Fermi_levels_2D, BandStructureQW, BandStructure
from sbe.c import gen_pot, gen_pot1
import multiprocessing


def plasmon_pole(q, omega, damp, omega_pl):
    eps = 1.0 + (omega_pl ** 2) / ((omega + 1j * damp) ** 2 - omega_pl ** 2)

    return 1 / eps


def absorption_matrix_inverse(bs, scratch=False, save=True):
    """
    Computes absorption spectra using the matrix inverse method
    (stationary solutions of the Liouville-von-Neumann equation)

    :param bs:             an object containing information about band structure and
                           dipole matrix elements
    :param scratch:
    :return:
    """

    dim = bs.dim

    # -------------------- arrays definition ---------------------

    l_k = 200  # length of k array
    l_f = 400  # length of frequency array
    Tempr = 10
    conc = 5.0e17

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, 4.3e9, l_k)
    freq_array = np.linspace(-1.9 * bs.mat.Eg, 1.5 * bs.mat.Eg, l_f) / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * e * 2.4, 0.5 * e * 2.4

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    V = int_matrix(wave_vector, 5)
    # V = np.zeros((l_k, l_k))

    flag = False
    subbandss = []
    ps = []

    logging.info("Loop over pairs of subbands:")
    # for j1 in [0]:
    #     for j2 in [0]:
    for j1 in [0, 1, 2, 3, 4, 5]:
        for j2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:

            if (j1 % 2) != (j2 % 2):

                # scr = 1

                # if j1 < 2:
                #     scr1 = 1.0
                # elif 2 <= j1 < 4:
                #     scr1 = 3.0
                # else:
                #     scr1 = 3.0
                #
                # if j2 < 2:
                #     scr2 = 1.0
                # elif 2 <= j2 < 4:
                #     scr2 = 3.0
                # elif 4 <= j2 < 6:
                #     scr2 = 3.0
                # else:
                #     scr2 = 3.0
                #
                # scr = scr1 * scr2

                logging.info("        cb = {}, vb = {}".format(j1, j2))
                subbands = bs.get_optical_transition_data(wave_vector, j2, j1)
                # plt.plot(subbands[3])
                subbandss.append(subbands[2] - subbands[1])

                if scratch:
                    ps1 = np.load('ps' + str(j1) + str(j2) + '_17_18.npy')
                else:
                    ps1 = polarization1(freq_array, dim, bs.mat, subbands,
                                        Ef_h, Ef_e,
                                        Tempr,
                                        V)
                    np.save('ps' + str(j1) + str(j2) + '_17_18.npy', ps1)

                ps1 = 2.0 * ps1
                ps1 = np.nan_to_num(ps1)

                if flag:

                    pad_width = int((subbands[2][0] - subbands[1][0] - Eg) / ((freq_array[2] - freq_array[1]) * h))
                    # print(pad_width)
                    if pad_width > 0:
                        ps1 = np.pad(ps1, pad_width, mode='edge')
                        ps1 = ps1[:np.size(freq_array)]

                else:
                    flag = True
                    Eg = subbands[2][0] - subbands[1][0]

                ps.append(ps1)

    # print(subbandss)

    ps_tot = np.sum(np.array(ps), axis=0)

    plt.figure()
    for item in ps:
        plt.plot((freq_array * h + Eg) / e, item / 1e5)
    plt.plot((freq_array * h + Eg) / e, ps_tot / 1e5, 'k')
    plt.fill_between((freq_array * h + Eg) / e, ps_tot / 1e5, facecolor='gray', alpha=0.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    plt.gca().legend(('heavy holes', 'light holes', 'total'))
    plt.show()

    energy = (freq_array * h + Eg) / e

    if save:
        np.save('abs.npy', ps_tot)
        np.save('energy.npy', energy)

    return energy, ps_tot


def absorption(bs, scratch=False, save=True):
    """
    Computes absorption spectra

    :param bs:             band structure
    :param scratch:
    :return:
    """

    dim = bs.dim

    # -------------------- arrays definition ---------------------

    l_k = 300  # length of k array
    l_f = 400  # length of frequency array
    Tempr = 10
    conc = 5.0e17

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, 6.3e9, l_k)
    freq_array = np.linspace(-1.9 * bs.mat.Eg, 1.5 * bs.mat.Eg, l_f) / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * e * 2.4, 0.5 * e * 2.4

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    # V = 0.5e-29*int_matrix(wave_vector, bs.mat.eps, dim=dim)
    # V = int_matrix(wave_vector, 5)
    V = np.zeros((l_k, l_k))

    pulse_widths = 0.01e-14
    pulse_delay = 10 * pulse_widths
    # pulse_amp = 1.0e7
    pulse_amp = 1.0e-3
    e_phot = 0.1 * bs.mat.Eg * 0

    def e_field(t):
        pulse_widths = 0.01e-14
        pulse_delay = 10 * pulse_widths
        # pulse_amp = 1.0e7
        pulse_amp = 1.0e-3
        e_phot = 0.1 * bs.mat.Eg * 0
        a = pulse_amp * np.exp(-((t - pulse_delay) ** 2) / (2 * pulse_widths ** 2)) * np.exp(1j * (e_phot / h) * t)
        return np.nan_to_num(a)

    flag = False
    subbandss = []
    ps = []

    logging.info("Loop over pairs of subbands:")
    for j1 in [0, 1, 2, 3, 4, 5]:
        for j2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:

            # for j1, j2 in [(0, 1), (2, 3), (1, 0), (3, 2), (4, 5), (5, 4)]:
            # for j1, j2 in [(0, 0), (1, 0), (0, 1), (1, 1),
            #                (2, 2), (2, 3), (3, 2), (3, 3),
            #                (4, 4), (4, 5), (5, 4), (5, 5)]:

            # for j1, j2 in [(1, 0), (0, 1),
            #                (2, 3), (3, 2),
            #                (4, 5), (5, 4)]:

            # for j1, j2 in [(0, 0), (1, 1),
            #                (2, 2), (3, 3),
            #                (4, 4), (5, 5)]:

            # for j1 in [4]:
            #     for j2 in [6]:
            # if (j1 % 2) == 0 and (j2 % 2) == 0:
            # if (j1 % 2) != (j2 % 2):
            # if j1 == j2:

            # for j1 in range(bs.n_sb_e):
            #     for j2 in range(bs.n_sb_h):

            logging.info("        cb = {}, vb = {}".format(j1, j2))
            subbands = bs.get_optical_transition_data(wave_vector, j2, j1)
            # plt.plot(subbands[3])
            subbandss.append(subbands[2] - subbands[1])

            scr = 1

            if j1 < 2:
                scr1 = 1.0
            elif 2 <= j1 < 4:
                scr1 = 2.0
            else:
                scr1 = 2.0

            if j2 < 2:
                scr2 = 1.0
            elif 2 <= j2 < 4:
                scr2 = 2.0
            elif 4 <= j2 < 6:
                scr2 = 2.0
            else:
                scr2 = 2.0

            scr = scr1 * scr2
            #
            # scr = scr1

            if scratch:
                ps1 = np.load('ps' + str(j1) + str(j2) + '_17_18.npy')
            else:
                ps1 = polarization1(freq_array, dim, bs.mat, subbands,
                                    Ef_h, Ef_e,
                                    Tempr,
                                    V / scr,
                                    e_field, pulse_widths, pulse_delay, pulse_amp, e_phot,
                                    debug=False)
                np.save('ps' + str(j1) + str(j2) + '_17_18.npy', ps1)

            ps1 = 2.0 * ps1

            # if (j1 == 1 and j2 == 4) or (j1 == 5 and j2 == 0):
            #     ps1 = 0.3 * ps1
            #
            # if (j1 == 4 and j2 == 5) or (j1 == 5 and j2 == 4):
            #     ps1 = 0.3 * ps1

            ps1 = np.nan_to_num(ps1)

            if flag:

                pad_width = int((subbands[2][0] - subbands[1][0] - Eg) / ((freq_array[2] - freq_array[1]) * h))
                # print(pad_width)
                if pad_width > 0:
                    ps1 = np.pad(ps1, pad_width, mode='edge')
                    ps1 = ps1[:np.size(freq_array)]

            else:
                flag = True
                Eg = subbands[2][0] - subbands[1][0]

            ps.append(ps1)

    # print(subbandss)

    ps_tot = np.sum(np.array(ps), axis=0)

    plt.figure()
    for item in ps:
        plt.plot((freq_array * h + Eg) / e, item / 1e5)
    plt.plot((freq_array * h + Eg) / e, ps_tot / 1e5, 'k')
    plt.fill_between((freq_array * h + Eg) / e, ps_tot / 1e5, facecolor='gray', alpha=0.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    plt.gca().legend(('heavy holes', 'light holes', 'total'))
    plt.show()

    energy = (freq_array * h + Eg) / e

    if save:
        np.save('abs.npy', ps_tot)
        np.save('energy.npy', energy)

    return energy, ps_tot


def main(scratch=False):
    gaas = GaAs()

    bs1 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band1.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band1.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles1.pkl')

    bs2 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band2.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band2.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles2.pkl')

    bs3 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band3.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band3.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles3.pkl')

    # -------------------- arrays definition ---------------------

    l_k = 100  # length of k array
    l_f = 400  # length of frequency array
    Tempr = 10
    conc = 5.0e17

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, 5.3e9, l_k)
    freq_array = np.linspace(-1.9 * bs1.mat.Eg, 1.5 * bs1.mat.Eg, l_f) / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * e * 2.4, 0.5 * e * 2.4

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    V = int_matrix(wave_vector, 7)
    # V = np.block([[V, np.eye(l_k, l_k), np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), V, np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), np.eye(l_k, l_k), V]])

    scr = 0.1
    scr1 = 0.05 * np.max(V)
    scr2 = 0.05 * np.max(V)

    # V = np.block([[V, V*scr, V*scr],
    #               [V*scr, V, V*scr],
    #               [V*scr, V*scr, V]])

    V = np.block([[V, np.ones((l_k, l_k)) * scr1 + 0.1 * V, np.ones((l_k, l_k)) * scr1 + 0.1 * V],
                  [np.ones((l_k, l_k)) * scr1 + 0.1 * V, V, np.ones((l_k, l_k)) * scr2 + 0.1 * V],
                  [np.ones((l_k, l_k)) * scr1 + 0.1 * V, np.ones((l_k, l_k)) * scr2 + 0.1 * V, V]])

    # V = np.zeros((l_k, l_k))

    flag = False
    subbandss = []
    ps = []

    logging.info("Loop over pairs of subbands:")
    # for j1 in [0]:
    #     for j2 in [0]:
    for j1 in [0, 1, 2, 3, 4, 5]:
        for j2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            # if (j1 % 2) == (j2 % 2):

            logging.info("        cb = {}, vb = {}".format(j1, j2))
            subbands1 = bs1.get_optical_transition_data(wave_vector, j2, j1)
            subbands2 = bs2.get_optical_transition_data(wave_vector, j2, j1)
            subbands3 = bs3.get_optical_transition_data(wave_vector, j2, j1)
            # plt.plot(subbands[3])
            subbandss.append(subbands1[2] - subbands1[1])

            subbands = []
            subbands.append(subbands1[0])
            subbands.append(np.hstack((subbands1[1], subbands2[1], subbands3[1])))
            subbands.append(np.hstack((subbands1[2], subbands2[2], subbands3[2])))
            aaa = np.hstack((np.ones(l_k), 3 * np.ones(l_k), 2 * np.ones(l_k)))
            subbands.append(aaa * np.hstack((subbands1[3], subbands2[3], subbands3[3])))

            if scratch:
                ps1 = np.load('ps' + str(j1) + str(j2) + '_17_18.npy')
            else:
                ps1 = polarization(freq_array, 1, bs1.mat, subbands,
                                   Ef_h, Ef_e,
                                   Tempr,
                                   V)
                np.save('ps' + str(j1) + str(j2) + '_17_18.npy', ps1)

            ps1 = 2.0 * ps1
            ps1 = np.nan_to_num(ps1)

            if flag:

                pad_width = int((subbands[2][0] - subbands[1][0] - Eg) / ((freq_array[2] - freq_array[1]) * h))
                # print(pad_width)
                if pad_width > 0:
                    ps1 = np.pad(ps1, pad_width, mode='edge')
                    ps1 = ps1[:np.size(freq_array)]

            else:
                flag = True
                Eg = subbands[2][0] - subbands[1][0]

            ps.append(ps1)

    # print(subbandss)

    ps_tot = np.sum(np.array(ps), axis=0)

    plt.figure()
    for item in ps:
        plt.plot((freq_array * h + Eg) / e, item / 1e5)
    plt.plot((freq_array * h + Eg) / e, ps_tot / 1e5, 'k')
    plt.fill_between((freq_array * h + Eg) / e, ps_tot / 1e5, facecolor='gray', alpha=0.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    plt.gca().legend(('heavy holes', 'light holes', 'total'))
    plt.show()

    energy = (freq_array * h + Eg) / e

    if save:
        np.save('abs.npy', ps_tot)
        np.save('energy.npy', energy)

    return energy, ps_tot


def main1(scratch=False, save=True):
    gaas = GaAs()

    bs1 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band0.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band0.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles0.pkl')

    bs1.plot()

    bs2 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band2.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band2.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles2.pkl')

    bs2.plot()
    # -------------------- arrays definition ---------------------

    l_k = 50  # length of k array
    l_f = 400  # length of frequency array
    Tempr = 10
    conc = 5.0e17

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, 6.0e9, l_k)
    freq_array = np.linspace(-1.9 * bs1.mat.Eg, 1.5 * bs1.mat.Eg, l_f) / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * e * 2.4, 0.5 * e * 2.4

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    # V = int_matrix(wave_vector, 7, R=1e21)
    # V1 = int_matrix(wave_vector, 7, R=1e21)
    # V2 = int_matrix(wave_vector, 7, R=1e21)

    lat_lc = np.pi / 0.9e-9

    eps1 = 4
    eps2 = 3
    dist = lat_lc * 0.5

    k_lim1 = 0
    k_lim2 = lat_lc * 0.5
    k_lim3 = lat_lc * 0.5
    k_lim4 = lat_lc

    # V1 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim1, k_lim2) * 1.0e-18
    V1 = gen_pot(wave_vector, eps1, eps2, k_lim1, 2 * k_lim2, k_lim1, 2 * k_lim2) * 0.2e-18
    V2 = gen_pot(wave_vector, eps1, eps2, k_lim3, k_lim4, k_lim3, k_lim4) * 0.2e-18
    V3 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4) * 0.2e-18

    # V1, V2, V3 = int_matrix_anisotr(wave_vector, eps1, eps2, dist, k_lim1, k_lim2, R=3)

    # V = np.block([[V, np.eye(l_k, l_k), np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), V, np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), np.eye(l_k, l_k), V]])

    # scr = 0.1
    # scr1 = 0.05 * np.max(V) *0
    # scr2 = 0.05 * np.max(V) *0

    # V = np.block([[V, V*scr, V*scr],
    #               [V*scr, V, V*scr],
    #               [V*scr, V*scr, V]V1])

    V = np.block([[V1, 0 * V3],
                  [0 * V3, 0 * V2 / 3]])

    # V = np.zeros((l_k, l_k))

    flag = False
    subbandss = []
    ps = []

    logging.info("Loop over pairs of subbands:")
    # for j1 in [0, 1]:
    #     for j2 in [0, 1]:
    for j1 in [0, 1, 2, 3, 4, 5]:
        for j2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            # if ((j1 % 2) == (j2 % 2)) and (j2 != 4):
            if (j1 % 2) != (j2 % 2):
                logging.info("        cb = {}, vb = {}".format(j1, j2))
                subbands1 = bs1.get_optical_transition_data(wave_vector, j2, j1)
                subbands2 = bs2.get_optical_transition_data(wave_vector, j2, j1)
                # plt.plot(subbands[3])
                subbandss.append(subbands1[2] - subbands1[1])

                subbands = []
                subbands.append(subbands1[0])
                subbands.append(np.hstack((subbands1[1], subbands2[1])))
                subbands.append(np.hstack((subbands1[2], subbands2[2])))
                subbands.append(np.hstack((subbands1[3], subbands2[3])))

                if scratch:
                    ps1 = np.load('ps' + str(j1) + str(j2) + '_17_18.npy')
                else:
                    ps1 = polarization1(freq_array, 1, bs1.mat, subbands,
                                        Ef_h, Ef_e,
                                        Tempr,
                                        V)
                    np.save('ps' + str(j1) + str(j2) + '_17_18.npy', ps1)

                ps1 = 2.0 * ps1
                ps1 = np.nan_to_num(ps1)

                if flag:

                    pad_width = int((subbands[2][0] - subbands[1][0] - Eg) / ((freq_array[2] - freq_array[1]) * h))
                    # print(pad_width)
                    if pad_width > 0:
                        ps1 = np.pad(ps1, pad_width, mode='edge')
                        ps1 = ps1[:np.size(freq_array)]

                else:
                    flag = True
                    Eg = subbands[2][0] - subbands[1][0]

                ps.append(ps1)

    # print(subbandss)

    ps_tot = np.sum(np.array(ps), axis=0)

    bse1 = np.loadtxt('/home/mk/perovsk_project/BSE.txt')
    bse2 = np.loadtxt('/home/mk/perovsk_project/BSE2.txt')

    plt.figure()
    # for item in ps:
    #     plt.plot((freq_array * h + Eg) / e, 10*item / 1e5)

    ps_tot *= 0.65e23
    plt.plot((freq_array * h + Eg) / e, ps_tot, 'r')
    # plt.fill_between((freq_array * h + Eg) / e, 10*ps_tot / 1e5, facecolor='gray', alpha=0.5)

    plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.fill_between(bse1[:, 0], bse2[:, 2], facecolor='gray', alpha=0.5)

    plt.xlabel('Energy (eV)')
    plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    plt.gca().legend(('heavy holes', 'light holes', 'total'))

    # plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.show()

    energy = (freq_array * h + Eg) / e

    if save:
        np.save('abs2.npy', ps_tot)
        np.save('energy.npy', energy)

    return energy, ps_tot


def main2(scratch=False, save=False):
    gaas = GaAs()

    bs1 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band1_f.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band1_f.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles1_f.pkl')

    # bs1.plot(kk = np.linspace(0, 0.5, 100))
    bs1.plot()

    bs2 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band2_f.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band2_f.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles2_f.pkl')

    bs2.plot()
    # -------------------- arrays definition ---------------------

    l_k = 30  # length of k array
    l_f = 400  # length of frequency array
    Tempr = 10
    conc = 5.0e17

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, 5.0e9, l_k)
    freq_array = np.linspace(-1.9 * bs1.mat.Eg, 1.5 * bs1.mat.Eg, l_f) / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * e * 2.4, 0.5 * e * 2.4

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    # V = int_matrix(wave_vector, 7, R=1e21)
    # V1 = int_matrix(wave_vector, 7, R=1e21)
    # V2 = int_matrix(wave_vector, 7, R=1e21)

    lat_lc = np.pi / 0.9e-9

    eps1 = 9
    eps2 = 5
    dist = lat_lc

    k_lim1 = 0
    k_lim2 = lat_lc * 0.5
    k_lim3 = lat_lc * 0.5
    k_lim4 = lat_lc

    # V1 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim1, k_lim2) * 1.0e-18

    VV1 = np.pi * k_lim2 ** 2 - np.pi * k_lim1 ** 2
    VV2 = np.pi * k_lim4 ** 2 - np.pi * k_lim3 ** 2

    V0 = gen_pot1(wave_vector, eps1, eps2)

    V1 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim1, k_lim2) / VV1
    V2 = gen_pot(wave_vector, eps1, eps2, k_lim3, k_lim4, k_lim3, k_lim4) / VV2
    V3 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4) / VV1
    V3c = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4) / VV2

    # V1, V2, V3 = int_matrix_anisotr(wave_vector, eps1, eps2, dist, k_lim1, k_lim2, R=3)

    # V = np.block([[V, np.eye(l_k, l_k), np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), V, np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), np.eye(l_k, l_k), V]])

    # scr = 0.1
    # scr1 = 0.05 * np.max(V) *0
    # scr2 = 0.05 * np.max(V) *0

    # V = np.block([[V, V*scr, V*scr],
    #               [V*scr, V, V*scr],
    #               [V*scr, V*scr, V]V1])

    # V = np.block([[V1, V3 / 3],
    #               [V3, V2 / 3]])

    V = np.block([[V1, V3],
                  [V3c, V2]]) * 10

    plt.plot(wave_vector / 1e9, V1[0, :] / e * (wave_vector[2] - wave_vector[1]), 'k')
    plt.plot(wave_vector / 1e9, V2[0, :] / e * (wave_vector[2] - wave_vector[1]), 'k--')
    plt.plot(wave_vector / 1e9, V3[0, :] / e * (wave_vector[2] - wave_vector[1]), 'k-.')
    plt.plot(wave_vector / 1e9, V0[0, :] / e * (wave_vector[2] - wave_vector[1]), 'r-.')
    plt.gca().legend(('$k_{\perp} \in I, k_{\perp}\' \in I$',
                      '$k_{\perp} \in II, k_{\perp}\' \in II$',
                      '$k_{\perp} \in I, k_{\perp}\' \in II$'))
    plt.yscale('log')
    # plt.contourf(wave_vector/1e9, wave_vector/1e9, V1, 300, locator=ticker.LogLocator())
    plt.xlabel(r'Wave vector (nm$^{-1}$)')
    plt.ylabel(r'Coulomb potential (eV)')
    # cbar = plt.colorbar()
    # cbar.set_label('Coulomb coupling matrix', rotation=270)
    plt.show()

    # V = np.zeros((l_k, l_k))

    flag = False
    subbandss = []
    ps = []

    logging.info("Loop over pairs of subbands:")
    # for j1 in [0, 1]:
    #     for j2 in [0, 1]:
    for j1 in [0, 1, 2, 3]:
        for j2 in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]:
            # if ((j1 % 2) == (j2 % 2)) and (j2 != 4):
            # if (j1 % 2) != (j2 % 2):

            scr = 1

            if j1 < 2:
                scr1 = 1.0
            elif 2 <= j1 < 4:
                scr1 = 10.3
            else:
                scr1 = 100.0

            if j2 < 2:
                scr2 = 1.1
            elif 2 <= j2 < 4:
                scr2 = 1.2
            elif 4 <= j2 < 6:
                scr2 = 1.3
            elif 6 <= j2 < 8:
                scr2 = 1.2
            elif 8 <= j2 < 10:
                scr2 = 1.1
            elif 10 <= j2 < 12:
                scr2 = 1.5
            else:
                scr2 = 400.0

            scr = scr1 * scr2

            logging.info("        cb = {}, vb = {}".format(j1, j2))
            subbands1 = bs1.get_optical_transition_data(wave_vector, j2, j1)
            subbands2 = bs2.get_optical_transition_data(wave_vector, j2, j1)
            # plt.plot(subbands[3])
            subbandss.append(subbands1[2] - subbands1[1])

            subbands = []
            subbands.append(subbands1[0])
            subbands.append(np.hstack((subbands1[1], subbands2[1])))
            subbands.append(np.hstack((subbands1[2], subbands2[2])))
            subbands.append(np.hstack((subbands1[3], subbands2[3])))

            if scratch:
                ps1 = np.load('ps' + str(j1) + str(j2) + '_17_18.npy')
            else:
                ps1 = polarization1(freq_array, 1, bs1.mat, subbands,
                                    Ef_h, Ef_e,
                                    Tempr,
                                    V / scr, VV1, VV1)
                np.save('ps' + str(j1) + str(j2) + '_17_18.npy', ps1)

            ps1 = 2.0 * ps1
            ps1 = np.nan_to_num(ps1)

            if flag:

                pad_width = int((subbands[2][0] - subbands[1][0] - Eg) / ((freq_array[2] - freq_array[1]) * h))
                # print(pad_width)
                if pad_width > 0:
                    ps1 = np.pad(ps1, pad_width, mode='edge')
                    ps1 = ps1[:np.size(freq_array)]

            else:
                flag = True
                Eg = subbands[2][0] - subbands[1][0]

            ps.append(ps1 * 0.55e23)

    # print(subbandss)

    ps_tot = np.sum(np.array(ps), axis=0)

    bse1 = np.loadtxt('/home/mk/perovsk_project/BSE.txt')
    bse2 = np.loadtxt('/home/mk/perovsk_project/BSE_EPS2.txt')

    energy = (freq_array * h + Eg) / e

    plt.figure()
    # for item in ps:
    #     plt.plot((freq_array * h + Eg) / e, 2*item)

    plt.plot((freq_array * h + Eg) / e, ps_tot, 'r')
    # plt.fill_between((freq_array * h + Eg) / e, 10*ps_tot / 1e5, facecolor='gray', alpha=0.5)

    plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.fill_between(bse1[:, 0], bse2[:, 2], facecolor='gray', alpha=0.5)

    plt.xlabel('Energy (eV)')
    plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    plt.gca().legend(('heavy holes', 'light holes', 'total'))

    # plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.show()

    if save:
        np.save('abs1.npy', ps_tot)
        np.save('energy.npy', energy)

    return energy, ps_tot


def main3(scratch=False, save=False):
    gaas = GaAs()

    bs1 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band1_f.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band1_f.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles1_f.pkl')

    # bs1.plot(kk = np.linspace(0, 0.5, 100))
    # bs1.plot()

    bs2 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band2_f.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band2_f.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles2_f.pkl')

    bs2.plot()
    # -------------------- arrays definition ---------------------

    l_k = 50  # length of k array
    l_f = 150  # length of frequency array
    Tempr = 10
    conc = 5.0e17

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, 6.0e9, l_k)
    freq_array = np.linspace(bs1.mat.Eg - 0.7 * bs1.mat.Eg, bs1.mat.Eg + 2.0 * bs1.mat.Eg, l_f) / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * e * 2.4, 0.5 * e * 2.4

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    # V = int_matrix(wave_vector, 7, R=1e21)
    # V1 = int_matrix(wave_vector, 7, R=1e21)
    # V2 = int_matrix(wave_vector, 7, R=1e21)

    lat_lc = np.pi / 0.9e-9

    eps1 = 9.2
    eps2 = 3.3
    dist = lat_lc

    k_lim1 = 0
    k_lim2 = lat_lc * 0.5
    k_lim3 = lat_lc * 0.5
    k_lim4 = lat_lc

    # V1 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim1, k_lim2) * 1.0e-18

    VV1 = np.pi * k_lim2 ** 2 - np.pi * k_lim1 ** 2
    VV2 = np.pi * k_lim4 ** 2 - np.pi * k_lim3 ** 2
    VV0 = np.pi * k_lim4 ** 2

    V1 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim1, k_lim2) / VV1
    V2 = gen_pot(wave_vector, eps1, eps2, k_lim3, k_lim4, k_lim3, k_lim4) / VV2
    V3 = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4) / VV1
    V3c = gen_pot(wave_vector, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4) / VV2

    # V1, V2, V3 = int_matrix_anisotr(wave_vector, eps1, eps2, dist, k_lim1, k_lim2, R=3)

    # V = np.block([[V, np.eye(l_k, l_k), np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), V, np.eye(l_k, l_k)],
    #               [np.eye(l_k, l_k), np.eye(l_k, l_k), V]])

    # scr = 0.1
    # scr1 = 0.05 * np.max(V) *0
    # scr2 = 0.05 * np.max(V) *0

    # V = np.block([[V, V*scr, V*scr],
    #               [V*scr, V, V*scr],
    #               [V*scr, V*scr, V]V1])

    # V = np.block([[V1, V3 / 3],
    #               [V3, V2 / 3]])

    V = np.block([[V1, V3],
                  [V3c, V2]]) * np.pi * np.pi * 1.45  * 0

    G = 7.4e9

    # V1 = gen_pot(wave_vector+G, eps1, eps2, k_lim1, k_lim2, k_lim1, k_lim2) / VV1
    # V2 = gen_pot(wave_vector+G, eps1, eps2, k_lim3, k_lim4, k_lim3, k_lim4) / VV2
    # V3 = gen_pot(wave_vector+G, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4) / VV1
    # V3c = gen_pot(wave_vector+G, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4) / VV2

    V0 = gen_pot1(wave_vector, eps1, eps2, G)
    V0 = np.block([[V0 * VV1, V0 * VV1],
                   [V0 * VV2, V0 * VV2]]) * 4 * 0

    l_k1 = V.shape[0]

    plt.plot(wave_vector / 1e9, V1[0, :] / e * (wave_vector[2] - wave_vector[1]), 'k')
    plt.plot(wave_vector / 1e9, V2[0, :] / e * (wave_vector[2] - wave_vector[1]), 'k--')
    plt.plot(wave_vector / 1e9, V3[0, :] / e * (wave_vector[2] - wave_vector[1]), 'k-.')
    plt.gca().legend(('$k_{\perp} \in I, k_{\perp}\' \in I$',
                      '$k_{\perp} \in II, k_{\perp}\' \in II$',
                      '$k_{\perp} \in I, k_{\perp}\' \in II$'))
    plt.yscale('log')
    # plt.contourf(wave_vector/1e9, wave_vector/1e9, V1, 300, locator=ticker.LogLocator())
    plt.xlabel(r'Wave vector (nm$^{-1}$)')
    plt.ylabel(r'Coulomb potential (eV)')
    # cbar = plt.colorbar()
    # cbar.set_label('Coulomb coupling matrix', rotation=270)
    plt.show()

    # V = np.zeros((l_k, l_k))

    flag = False
    subbandss = []
    ps = []

    cb_inds = [0, 1, 2, 3]
    vb_inds = [0, 1, 2, 3, 4, 5, 6, 7]

    pairs = []
    counter = 0
    pairs_to_ind = np.zeros((max(cb_inds) + 1, max(vb_inds) + 1), dtype=np.int) - 100

    for j1 in cb_inds:
        for j2 in vb_inds:
            pairs.append((j1, j2))
            pairs_to_ind[j1, j2] = counter
            counter += 1

    for jj, ind in enumerate(pairs):
        print(jj, ind)

    num_pairs = len(pairs)
    ps = np.zeros((l_f, 1), dtype=np.complex)
    norm = np.hstack((np.ones(len(wave_vector)), np.ones(len(wave_vector)) * 3))

    # pool = multiprocessing.Pool(4)
    # out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))

    for j, fr in enumerate(freq_array):
        logging.info("   {} out of {}".format(j, l_f))
        M = lil_matrix((l_k1 * num_pairs, l_k1 * num_pairs), dtype=np.complex)
        mu = np.zeros((l_k1 * num_pairs, 1), dtype=np.complex)
        mu1 = np.zeros((l_k1 * num_pairs, 1), dtype=np.complex)
        logging.info("Loop over pairs of subbands:")
        for jj, ind in enumerate(pairs):
                j1 = ind[0]
                j2 = ind[1]
            # if j1 % 2 != j2 % 2:

                # logging.info("        cb = {}, vb = {}".format(j1, j2))
                subbands1 = bs1.get_optical_transition_data(wave_vector, j2, j1)
                subbands2 = bs2.get_optical_transition_data(wave_vector, j2, j1)
                # plt.plot(subbands[3])
                subbandss.append(subbands1[2] - subbands1[1])

                subbands = []
                subbands.append(subbands1[0])
                subbands.append(np.hstack((subbands1[1], subbands2[1])))
                subbands.append(np.hstack((subbands1[2], subbands2[2])))
                subbands.append(np.hstack((subbands1[3], subbands2[3])))

                aaa = vertex1(fr, 1, bs1.mat, subbands,
                              Ef_h, Ef_e,
                              Tempr,
                              V)

                M[jj * l_k1:(jj + 1) * l_k1, jj * l_k1:(jj + 1) * l_k1] = aaa
                mu[jj * l_k1:(jj + 1) * l_k1, 0] = subbands[3]
                mu1[jj * l_k1:(jj + 1) * l_k1, 0] = subbands[3] * norm

                # if j1 < 2:
                #     index1 = pairs_to_ind[j1 + 2, j2]
                #     index2 = jj
                #     M[index1 * l_k1:(index1 + 1) * l_k1, index2 * l_k1:(index2 + 1) * l_k1] = 1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.45
                #     M[index2 * l_k1:(index2 + 1) * l_k1, index1 * l_k1:(index1 + 1) * l_k1] = -1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.45
                #
                # if j2 < 2:
                #     index1 = pairs_to_ind[j1, j2 + 2]
                #     index2 = jj
                #     M[index1 * l_k1:(index1 + 1) * l_k1, index2 * l_k1:(index2 + 1) * l_k1] = 1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.07
                #     M[index2 * l_k1:(index2 + 1) * l_k1, index1 * l_k1:(index1 + 1) * l_k1] = -1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.07

                # if 4 <= j2 < 6:
                #     index1 = pairs_to_ind[j1, j2+6]
                #     index2 = jj
                #     M[index1 * l_k1:(index1 + 1) * l_k1, index2 * l_k1:(index2 + 1) * l_k1] = 1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.15
                #     M[index2 * l_k1:(index2 + 1) * l_k1, index1 * l_k1:(index1 + 1) * l_k1] = -1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.15

                # if 6 <= j2 < 8:
                #     index1 = pairs_to_ind[j1, j2+2]
                #     index2 = jj
                #     M[index1 * l_k1:(index1 + 1) * l_k1, index2 * l_k1:(index2 + 1) * l_k1] = 1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.1
                #     M[index2 * l_k1:(index2 + 1) * l_k1, index1 * l_k1:(index1 + 1) * l_k1] = -1j * V0 * (
                #             wave_vector[3] - wave_vector[2]) * 0.1

        pp, _ = lgmres(M, mu1 * 1e30)
        p = 1.0 + 4 * np.pi * np.sum(pp * mu) * (wave_vector[3] - wave_vector[2])
        ps[j] = p / (4.0 * np.pi * eps0 * bs1.mat.eps)

    # plt.spy(M)
    # plt.imshow(np.abs(M.todense()))
    # plt.show()

    ps = np.imag(ps)

    bse1 = np.loadtxt('/home/mk/perovsk_project/BSE.txt')
    bse2 = np.loadtxt('/home/mk/perovsk_project/BSE_EPS2.txt')

    print(np.max(ps))

    amp = 7.0e-12
    amp = 1.0 / np.max(ps)

    plt.fill_between(bse1[:, 0], bse2[:, 2] / np.max(bse2[:, 2]), facecolor='gray', alpha=0.5)
    plt.plot(freq_array * h / e, ps * amp, 'r')
    plt.show()

    ps_tot = np.sum(np.array(ps), axis=0)

    # energy = (freq_array * h + Eg) / e

    plt.figure()
    for item in ps:
        plt.plot(freq_array * h / e, 2 * item)

    plt.plot(freq_array * h / e, ps_tot, 'r')
    # plt.fill_between((freq_array * h + Eg) / e, 10*ps_tot / 1e5, facecolor='gray', alpha=0.5)

    plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.fill_between(bse1[:, 0], bse2[:, 2], facecolor='gray', alpha=0.5)

    plt.xlabel('Energy (eV)')
    plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    plt.gca().legend(('heavy holes', 'light holes', 'total'))

    # plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.show()

    if save:
        np.save('abs1.npy', ps_tot)
        np.save('energy.npy', energy)

    return energy, ps_tot


if __name__ == '__main__':

    l_f = 150

    gaas = GaAs()

    bs1 = BandStructure(gaas,
                        val_band='/home/mk/perovsk_project/val_band1_f.pkl',
                        cond_band='/home/mk/perovsk_project/cond_band1_f.pkl',
                        dipoles='/home/mk/perovsk_project/dipoles1_f.pkl')

    freq_array = np.linspace(bs1.mat.Eg - 0.7 * bs1.mat.Eg, bs1.mat.Eg + 2.0 * bs1.mat.Eg, l_f) / h

    bse1 = np.loadtxt('/home/mk/perovsk_project/BSE.txt')
    bse2 = np.loadtxt('/home/mk/perovsk_project/BSE_EPS2.txt')
    ps = np.load('ps.npy')
    ps0 = np.load('ps0.npy')
    ps1 = np.load('ps1.npy')

    # plt.fill_between(bse1[:, 0], bse2[:, 2] / np.max(bse2[:, 2]), facecolor='gray', alpha=0.5)
    # plt.plot(freq_array * h / e,  0.9 * ps / np.max(ps), 'k')
    # plt.plot(freq_array * h / e, ps0 / np.max(ps0), 'k--')
    # # plt.plot(freq_array * h / e, ps1 / np.max(ps1), 'r')
    # plt.xlim([1.5, 4.5])
    # plt.ylim([0, 1.4])
    # plt.xlabel('Energy (eV)', fontsize=14)
    # plt.ylabel('Absorption (a.u.)', fontsize=14)
    # plt.show()

    # gaas = GaAs()
    # bs = BandStructure3D(material=gaas)
    # energy, ans = absorption(bs)

    energy, ps = main3()

    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    #
    # data_loaded = True
    #
    # try:
    #     ans1 = np.load('abs.npy')
    #     energy1 = np.load('energy.npy')
    # except:
    #     data_loaded = False
    #
    # gaas = GaAs()
    # # bs = BandStructure3D(material=gaas)
    # bs = BandStructure(gaas,
    #                    val_band='/home/mk/perovsk_project/val_band0.pkl',
    #                    cond_band='/home/mk/perovsk_project/cond_band0.pkl',
    #                    dipoles='/home/mk/perovsk_project/dipoles0.pkl')
    #
    # bs.plot()
    # energy, ans = absorption_matrix_inverse(bs, scratch=False, save=False)
    #
    # bs = BandStructure(gaas,
    #                    val_band='/home/mk/perovsk_project/val_band2.pkl',
    #                    cond_band='/home/mk/perovsk_project/cond_band2.pkl',
    #                    dipoles='/home/mk/perovsk_project/dipoles2.pkl')
    #
    # bs.plot()
    # energy1, ans1 = absorption_matrix_inverse(bs, scratch=False, save=False)
    #
    # bs = BandStructure(gaas,
    #                    val_band='/home/mk/perovsk_project/val_band3.pkl',
    #                    cond_band='/home/mk/perovsk_project/cond_band3.pkl',
    #                    dipoles='/home/mk/perovsk_project/dipoles3.pkl')
    #
    # bs.plot()
    # energy2, ans2 = absorption_matrix_inverse(bs, scratch=False, save=False)
    #
    # bse1 = np.loadtxt('/home/mk/perovsk_project/BSE.txt')
    # bse2 = np.loadtxt('/home/mk/perovsk_project/BSE2.txt')
    #
    # # plt.plot(bse[:, 0], bse[:, 2])
    # # plt.plot(energy,    0.08*1e14*ans)
    # #
    # # np.save('absorp0.npy', 0.3*1e18*ans)
    # #
    # # # plt.plot(energy, 8e12 * abs1 - 1)
    # # plt.show()
    #
    # #abs0 = np.load('absorp0.npy')
    # #abs1 = np.load('absorp1.npy')
    #
    # plt.figure()
    # plt.plot(energy, 1e16*ans)
    # # plt.plot(energy, (0.5 * (np.tanh(8*(energy - 2.1)) + 1.0) * 1e-17 + ans)/30e-20)
    # # plt.plot(energy, 3e-5 * abs0)
    # if data_loaded:
    #     plt.plot(energy1, 1e16*(ans+0.6*ans1), 'g')
    #     # plt.plot(energy1, 1e16 * ans1, 'g')
    # # plt.plot(bse1[:, 0], np.imag(np.sqrt(bse1[:, 2] + 1j*bse2[:, 2])/bse1[:, 0]), 'k')
    # # plt.fill_between(bse1[:, 0], np.imag(np.sqrt(bse1[:, 2] + 1j*bse2[:, 2])/bse1[:, 0]), facecolor='gray', alpha=0.5)
    # # plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    # # plt.fill_between(bse1[:, 0], bse2[:, 2], facecolor='gray', alpha=0.5)
    # plt.xlabel('Energy (eV)')
    # plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    # plt.gca().legend(('1D with Coulomb effects', '1D without Coulomb effects', 'BSE'))
    # plt.show()

    bse1 = np.loadtxt('/home/mk/perovsk_project/BSE.txt')
    bse2 = np.loadtxt('/home/mk/perovsk_project/BSE2.txt')
    # energy = np.load('energy.npy')
    # abs0 = np.load('abs0.npy')
    # abs1 = np.load('abs1.npy')
    # abs2 = np.load('abs2.npy')
    #
    #
    # plt.figure()
    # plt.plot(energy, abs1, 'k')
    # plt.plot(energy, abs0, 'k--')
    # plt.plot(energy, abs2)

    data = np.vstack((energy, np.imag(ps))).T

    # np.save('eps.npy', data)
    # np.savetxt('eps.txt', data)
    # data = np.load('eps.npy')

    plt.plot(data[:, 0], data[:, 1], 'b')
    # plt.plot(data1[:, 0], data1[:, 1], 'r')
    plt.fill_between(bse1[:, 0], bse2[:, 2], facecolor='gray', alpha=0.5)

    plt.xlabel('Energy (eV)')
    plt.ylabel(r'$\varepsilon_{yy}$ ($\omega$)')
    # plt.gca().legend(('with Coulomb coupling', 'without Coulomb coupling', 'averaging', 'GW approx.'))
    plt.xlim([1.5, 5])

    # plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.show()

    absorp = 2 * np.imag(energy * e / h / c * np.sqrt(ps)) * 1e10
    plt.plot(energy, absorp, 'r')
    plt.show()

    data = np.vstack((energy, absorp)).T
    # np.save('abs_spectrum.npy', data)
    # np.savetxt('abs_spectrum.txt', data)
    # data = np.load('abs_spectrum.npy')

    plt.plot(data[:, 0], data[:, 1], 'r')

    plt.xlabel('Energy (eV)')
    plt.ylabel(r'$\varepsilon_{yy}$ ($\omega$)')
    # plt.gca().legend(('with Coulomb coupling', 'without Coulomb coupling', 'averaging', 'GW approx.'))
    plt.xlim([1.5, 5])

    # plt.plot(bse1[:, 0], bse2[:, 2], 'k')
    plt.show()
