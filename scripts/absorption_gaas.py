import logging
import numpy as np
import matplotlib.pyplot as plt
from sbe.constants import h, e
from sbe.int_matrix import int_matrix
from sbe.Pol_FFT_f2py import polarization
from sbe.semiconductors import BandStructureQW, BandStructure3D, GaAs
from sbe.aux_functions import yaml_parser
import sbe.constants as const


config_file = """

verbosity:  True
damp:      0.003    # dephasing factor in eV

# ---------------- k grids ------------------

l_k:         300    # length of k array
k_max:    1.0e+9    # k-vector cutoff

# ------------- frequency array -------------

l_f:         400    # number of points in the frequency array
f_min:       0.7    # frequency array limits in units of the band gap
f_max:       1.1    # frequency array limits in units of the band gap

# --------- environment parameters ----------

tempr:         1    # temperature in K 
conc:    5.0e+14    # carrier concentration

# --------- external field parameters -------

pulse_width: 0.01   # pulse width in femtoseconds
pulse_delay:  100   # pulse delay in the units of the pulse width
pulse_amp: 1.e-27   # amplitude
e_phot:         0   # photon energy in the units of the fundamental band gap

# ----------- data management ---------------

scratch:   False
save:      False

"""


def absorption(bs, **kwargs):
    """
    Computes absorption spectra

    :param bs:             band structure
    :param scratch:
    :return:
    """

    dim = bs.dim

    # -------------------- arrays definition ---------------------
    verbosity = kwargs.get('verbosity')
    damp = kwargs.get('damp') * const.e   # damping
    l_k = kwargs.get('l_k')  # length of k array
    l_f = kwargs.get('l_f')  # length of frequency array
    Tempr = kwargs.get('tempr')
    conc = kwargs.get('conc')
    k_max = kwargs.get('k_max')
    f_min = kwargs.get('f_min')
    f_max = kwargs.get('f_max')
    pulse_width = kwargs.get('pulse_width')
    pulse_delay = kwargs.get('pulse_delay')
    pulse_amp = kwargs.get('pulse_amp')
    e_phot = kwargs.get('e_phot')
    pulse_widths = 1e-15 * pulse_width
    pulse_delay = pulse_delay * pulse_widths
    e_phot = e_phot * bs.mat.Eg
    scratch = kwargs.get('scratch')
    save = kwargs.get('save')

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, k_max, l_k)
    freq_array = np.linspace(f_min * bs.mat.Eg, f_max * bs.mat.Eg, l_f) - bs.mat.Eg
    freq_array = freq_array / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * bs.mat.Eg, 0.5 * bs.mat.Eg

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    # V = 0.5e-29*int_matrix(wave_vector, bs.mat.eps, dim=dim)
    V = int_matrix(wave_vector, bs.mat.eps, dim=bs.dim)
    # V = np.zeros((l_k, l_k))

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
    for j1 in [0]:
        for j2 in [0]:

            logging.info("        cb = {}, vb = {}".format(j1, j2))
            subbands = bs.get_optical_transition_data(wave_vector, j2, j1)
            # plt.plot(subbands[3])
            subbandss.append(subbands[2] - subbands[1])

            if scratch:
                ps1 = np.load('ps' + str(j1) + str(j2) + '_17_18.npy')
            else:
                ps1 = polarization(freq_array, dim, bs.mat, subbands,
                                   Ef_h, Ef_e,
                                   Tempr,
                                   V,
                                   damp,
                                   e_field, pulse_widths, pulse_delay, pulse_amp, e_phot,
                                   debug=verbosity)

                np.save('ps' + str(j1) + str(j2) + '_17_18.npy', ps1)

            ps1 = 2.0 * ps1
            ps1 = np.nan_to_num(ps1)

            if flag:

                pad_width = int((subbands[2][0] - subbands[1][0] - Eg) / ((freq_array[2] - freq_array[1]) * h))
                if pad_width > 0:
                    ps1 = np.pad(ps1, pad_width, mode='edge')
                    ps1 = ps1[:np.size(freq_array)]

            else:
                flag = True
                Eg = subbands[2][0] - subbands[1][0]

            ps.append(ps1)

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


def main2D():
    params = yaml_parser(config_file)

    gaas = GaAs()
    bs = BandStructureQW(material=gaas)
    energy, ans = absorption(bs, **params)

    plt.plot(energy, ans)


def main3D():
    params = yaml_parser(config_file)

    gaas = GaAs()
    bs = BandStructure3D(material=gaas)
    energy, ans = absorption(bs, **params)

    plt.plot(energy, ans)


if __name__ == '__main__':
    main2D()
    main3D()
