import logging
import numpy as np
import matplotlib.pyplot as plt
from sbe.constants import h, e
from sbe.int_matrix import int_matrix
from sbe.Pol_FFT_f2py import polarization_app
from sbe.semiconductors import BandStructure2D, BandStructure3D, GaAs
from sbe.aux_functions import yaml_parser
import sbe.constants as const


config_file = """verbosity:  True
damp:       0.003    # dephasing factor in eV

# ---------------- k grids ------------------

l_k:         300    # length of k array
k_max:    1.0e+9    # k-vector cutoff

# ------------- frequency array -------------

l_f:         2000    # number of points in the frequency array
f_min:       1.45    # frequency array limits in units of the band gap
f_max:       1.6    # frequency array limits in units of the band gap

# --------- environment parameters ----------

tempr:         1    # temperature in K 
conc:    1.0e+14    # carrier concentration

# --------- external field parameters -------

pulse_width:  5.0   # pulse width in femtoseconds
pulse_delay:  100   # pulse delay in the units of the pulse width
pulse_amp:  5.e-5   # amplitude
e_phot:         0   # photon energy in the units of the fundamental band gap

# ----------- data management ---------------

scratch:   False
save:      False
file_label:  "1"    # id added to the file name 
temp_dir:    "~"    # storage for graphic information

"""

mat_file = """# -------- Band structure parameters ---------

Eg:      1.519      # band gap
Eso:    -0.341      # spin-orbit splitting

gamma1:   6.98      # Luttinger parameters
gamma2:   2.06
gamma3:   2.93
        
me:     0.0665      # electron effective mass
mso:     0.172      # spin-orbit effective mass

# ----------- Dielectric screening -----------

eps:     12.93      # dielectric constant
n_reff:   3.16      # refractive index

# ------------ Varshni parameters ------------

alpha:   0.605      # meV / K    
betha:     204      # meV
"""


def absorption(bs, cc=True, **kwargs):
    """
    Computes absorption spectra

    :param bs:             band structure
    :param scratch:
    :return:
    """

    dim = bs.dim

    # -------------------- arrays definition ---------------------
    verbosity = kwargs.get('verbosity', 1)
    damp = kwargs.get('damp', 0.005) * const.e   # damping
    l_k = kwargs.get('l_k')  # length of k array
    l_f = kwargs.get('l_f', 100)  # length of frequency array
    Tempr = kwargs.get('tempr', 15)
    conc = kwargs.get('conc')
    k_max = kwargs.get('k_max', 1e9)
    f_min = kwargs.get('f_min', 0) * const.e
    f_max = kwargs.get('f_max', 0.5) * const.e
    pulse_width = kwargs.get('pulse_width', 1.0)
    pulse_delay = kwargs.get('pulse_delay', 100)
    pulse_amp = kwargs.get('pulse_amp', 1e2)
    e_phot = kwargs.get('e_phot', 0)
    pulse_widths = 1e-15 * pulse_width
    pulse_delay = pulse_delay * pulse_widths
    e_phot = e_phot * bs.mat.Eg
    scratch = kwargs.get('scratch', 0)
    save = kwargs.get('save', 0)

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, k_max, l_k)
    freq_array = np.linspace(f_min, f_max, l_f) - bs.mat.Eg
    freq_array = freq_array / h

    # ------------------------------------------------------------

    # Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    Ef_h, Ef_e = 0.5 * bs.mat.Eg, 0.5 * bs.mat.Eg

    logging.info('Fermi levels are: Ef_h = ' + str(Ef_h / e) + ' eV' + ' and Ef_e = ' + str(Ef_e / e) + ' eV')

    # V = 0.5e-29*int_matrix(wave_vector, bs.mat.eps, dim=dim)
    if cc:
        V = int_matrix(wave_vector, bs.mat.eps, dim=bs.dim)
    else:
        V = np.zeros((l_k, l_k))

    def e_field(t):
        # pulse_widths = 0.01e-14
        # pulse_delay = 10 * pulse_widths
        # # pulse_amp = 1.0e7
        # pulse_amp = 1.0e-3
        # e_phot = 0.1 * bs.mat.Eg * 0
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
                ps1, data = polarization_app(freq_array, dim, bs.mat, subbands,
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

    # plt.figure()
    # for item in ps:
    #     plt.plot((freq_array * h + Eg) / e, item / 1e5)
    # plt.plot((freq_array * h + Eg) / e, ps_tot / 1e5, 'k')
    # plt.fill_between((freq_array * h + Eg) / e, ps_tot / 1e5, facecolor='gray', alpha=0.5)
    # plt.xlabel('Energy (eV)')
    # plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    # plt.gca().legend(('heavy holes', 'light holes', 'total'))
    # plt.show()

    energy = (freq_array * h + Eg) / e

    if save:
        np.save('abs.npy', ps_tot)
        np.save('energy.npy', energy)

    return energy, ps_tot, data


def main2D():
    params = yaml_parser(config_file)

    gaas = GaAs()
    bs = BandStructure2D(material=gaas)
    energy, ans, _ = absorption(bs, **params)
    energy, ans1, fig = absorption(bs, cc=False, **params)

    plt.plot((energy - gaas.Eg/e)/gaas.E_0*e, ans / np.max(ans), 'k')
    plt.plot((energy - gaas.Eg/e)/gaas.E_0*e, ans1 / np.max(ans), 'k--')
    plt.xlim([-20, 20])
    # plt.xlim([1.4, 1.7])
    # plt.ylim([0, 1.4])
    plt.xlabel(r'(E-E$_g$)/E$_0$ (a.u.)', fontsize=14)
    plt.ylabel('Absorption (a.u.)', fontsize=14)
    plt.show()


def main3D():
    params = yaml_parser(config_file)

    gaas = GaAs()
    bs = BandStructure3D(material=gaas)
    energy, ans, fig = absorption(bs, **params)
    energy, ans1, fig = absorption(bs, cc=False, **params)

    plt.plot((energy - gaas.Eg/e)/gaas.E_0*e, ans / np.max(ans), 'k')
    plt.plot((energy - gaas.Eg/e)/gaas.E_0*e, ans1 / np.max(ans), 'k--')
    plt.xlim([-20, 20])
    # plt.xlim([1.4, 1.7])
    # plt.ylim([0, 1.4])
    plt.xlabel(r'(E-E$_g$)/E$_0$ (a.u.)', fontsize=14)
    plt.ylabel('Absorption (a.u.)', fontsize=14)
    plt.show()


if __name__ == '__main__':
    # main2D()
    main3D()
