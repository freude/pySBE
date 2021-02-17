import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lgmres
import matplotlib.pyplot as plt
import logging
from sbe.constants import h, e, c, eps0
from sbe.stationary import vertex1
from sbe.semiconductors import Tc
from sbe.semiconductors import BandStructure


def potential_mat(k, G, eps1, eps2):

    l_k = k.shape[0]
    V = np.zeros((l_k, l_k))

    for j1 in range(l_k):
        for j2 in range(l_k):
            q = eps1 * np.abs(k[j1, 0] - k[j2, 0])**2 + \
                eps2 * np.abs(k[j1, 1] - k[j2, 1])**2 + \
                eps1 * np.abs(k[j1, 2] - k[j2, 2])**2 + \
                G**2 + \
                1.0e17
            V[j1][j2] = 1.0 / q

    return V * e ** 2 / (4.0 * np.pi * (2*np.pi) * eps0)


def plasmon_pole(q, omega, damp, omega_pl):
    eps = 1.0 + (omega_pl ** 2) / ((omega + 1j * damp) ** 2 - omega_pl ** 2)

    return 1 / eps


def main(scratch=False, save=False):
    tc = Tc()

    bs = BandStructure(tc,
                       val_band='/home/mk/perovsk_project/val_band_3d.pkl',
                       cond_band='/home/mk/perovsk_project/cond_band_3d.pkl',
                       dipoles='/home/mk/perovsk_project/dipoles_3d.pkl')

    # num_point = 100
    # kx = np.zeros(num_point)
    # ky = np.linspace(0, 1.0, 100) * 3.0e9
    # kk = np.vstack((kx, ky, kx)).T
    # bs1.plot(kk)

    # -------------------- arrays definition ---------------------

    wave_vector_grid = np.array([8, 8, 8])
    wave_vector_grid_center = wave_vector_grid // 2
    kk_max = np.array([3.2, 6.5, 2.4]) * 1e9

    cube_kk_x = np.linspace(-kk_max[0], kk_max[0], wave_vector_grid[0])
    cube_kk_y = np.linspace(-kk_max[1], kk_max[1], wave_vector_grid[1])
    cube_kk_z = np.linspace(-kk_max[2], kk_max[2], wave_vector_grid[2])
    k_x, k_y, k_z = np.meshgrid(cube_kk_x, cube_kk_y, cube_kk_z, indexing='ij')
    wave_vector = np.vstack((k_x.flatten(), k_y.flatten(), k_z.flatten())).T
    l_k = wave_vector.shape[0]                 # length of k array
    stk = (cube_kk_x[1] - cube_kk_x[0]) * (cube_kk_y[1] - cube_kk_y[0]) * (cube_kk_z[1] - cube_kk_z[0])

    # ------------------------------------------------------------

    Tempr = 10

    # ------------------------------------------------------------

    l_f = 50  # length of frequency array
    freq_array = np.linspace(bs.mat.Eg - 0.7 * bs.mat.Eg, bs.mat.Eg + 2.0 * bs.mat.Eg, l_f) / h
    freq_array = np.linspace(bs.mat.Eg - 0.5 * bs.mat.Eg, bs.mat.Eg + 1.7 * bs.mat.Eg, l_f) / h

    # ------------------------------------------------------------

    # Fermi levels
    Ef_h, Ef_e = 0.5 * e * 2.4, 0.5 * e * 2.4

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    # screening
    eps1 = 5
    eps2 = 5.3

    G = 3.4e9

    V = potential_mat(wave_vector, 0, eps1, eps2) / 10
    V1 = potential_mat(wave_vector, G, eps1, eps2) / 10
    # V = np.zeros((l_k, l_k))

    subbandss = []
    cb_inds = [0, 1, 2, 3]
    vb_inds = [0, 1, 2, 3, 4, 5, 6, 7]
    # cb_inds = [0, 1]
    # vb_inds = [0, 1]

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

    interband_c = np.ones((len(cb_inds), len(cb_inds))) - np.identity(len(cb_inds)) * 3.0
    interband_v = np.ones((len(vb_inds), len(vb_inds))) - np.identity(len(vb_inds)) * 1.0

    interband_c = np.load('/home/mk/c_mix.npy') / 30.0
    interband_v = np.load('/home/mk/v_mix.npy') / 60.0

    num_pairs = len(pairs)
    ps = np.zeros((l_f, 1), dtype=np.complex)

    for j, fr in enumerate(freq_array):
        logging.info("   {} out of {}".format(j, l_f))
        # M = lil_matrix((l_k * num_pairs, l_k * num_pairs), dtype=np.complex)
        M = np.zeros((l_k * num_pairs, l_k * num_pairs), dtype=np.complex)
        mu = np.zeros((l_k * num_pairs, 1), dtype=np.complex)
        # ---------------------------------------------------------------------------------------------
        logging.info("Loop over pairs of subbands:")
        for index1, ind1 in enumerate(pairs):
            j1 = ind1[0]
            j2 = ind1[1]
            # if j1 % 2 == j2 % 2:
            if True:

                # logging.info("        cb = {}, vb = {}".format(j1, j2))
                subbands = bs.get_optical_transition_data(wave_vector, j2, j1)
                subbandss.append(subbands[2] - subbands[1])

                aaa = vertex1(fr, 1, bs.mat, subbands,
                              Ef_h, Ef_e,
                              Tempr,
                              V,
                              stk)

                M[index1 * l_k:(index1 + 1) * l_k, index1 * l_k:(index1 + 1) * l_k] = aaa
                mu[index1 * l_k:(index1 + 1) * l_k, 0] = subbands[3]

                for index2, ind2 in enumerate(pairs):
                    if index2 > index1:
                        jj1 = ind2[0]
                        jj2 = ind2[1]
                        # if jj1 % 2 == jj2 % 2 and np.abs(j1-jj1) == 2 and np.abs(j2-jj2) == 2:
                        # if jj1 % 2 == jj2 % 2:
                        if True:
                            M[index1 * l_k:(index1 + 1) * l_k, index2 * l_k:(index2 + 1) * l_k] = \
                                                               V1 * stk * interband_c[j1, jj1] * interband_v[j2, jj2]
                            M[index2 * l_k:(index2 + 1) * l_k, index1 * l_k:(index1 + 1) * l_k] = \
                                                               V1 * stk * np.conj(interband_c[j1, jj1]) * np.conj(interband_v[j2, jj2])

        # ---------------------------------------------------------------------------------------------

        pp, _ = lgmres(M, mu * 1e30)
        p = 1.0 + 4 * np.pi * np.sum(pp * mu) * stk
        ps[j] = p / (4.0 * np.pi * eps0 * bs.mat.eps)

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
    # for item in ps:
    #     plt.plot(freq_array * h / e, 2 * item)

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

    # l_f = 150
    #
    # gaas = GaAs()
    #
    # bs1 = BandStructure(gaas,
    #                     val_band='/home/mk/perovsk_project/val_band_3d.pkl',
    #                     cond_band='/home/mk/perovsk_project/cond_band_3d.pkl',
    #                     dipoles='/home/mk/perovsk_project/dipoles_3d.pkl')
    #
    #
    # num_point = 100
    # kx = np.zeros(num_point)
    # ky = np.linspace(0, 1.0, 100) * 3.0e9
    #
    # kk = np.vstack((kx, ky, kx)).T
    # bs1.plot(kk)

    # freq_array = np.linspace(bs1.mat.Eg - 0.7 * bs1.mat.Eg, bs1.mat.Eg + 2.0 * bs1.mat.Eg, l_f) / h
    #
    # bse1 = np.loadtxt('/home/mk/perovsk_project/BSE.txt')
    # bse2 = np.loadtxt('/home/mk/perovsk_project/BSE_EPS2.txt')
    # ps = np.load('ps.npy')
    # ps0 = np.load('ps0.npy')
    # ps1 = np.load('ps1.npy')
    #
    # plt.fill_between(bse1[:, 0], bse2[:, 2] / np.max(bse2[:, 2]), facecolor='gray', alpha=0.5)
    # plt.plot(freq_array * h / e,  0.9 * ps / np.max(ps), 'k')
    # plt.plot(freq_array * h / e, ps0 / np.max(ps0), 'k--')
    # # plt.plot(freq_array * h / e, ps1 / np.max(ps1), 'r')
    # plt.xlim([1.5, 4.5])
    # plt.ylim([0, 1.4])
    # plt.xlabel('Energy (eV)', fontsize=14)
    # plt.ylabel('Absorption (a.u.)', fontsize=14)
    # plt.show()

    energy, ps = main()

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
