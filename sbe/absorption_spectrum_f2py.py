import numpy as np
import matplotlib.pyplot as plt
from sbe.constants import h, e
from sbe.int_matrix import int_matrix
from sbe.polarization_f2py import polarization
from sbe.semiconductors import GaAs, Tc
from sbe.semiconductors import BandStructure3D, get_Fermi_levels_2D


def absorption(gaas, bs):

    dim = bs.dim

    # -------------------- arrays definition ---------------------

    l_k = 300                       # length of k array
    l_f = 500                       # length of frequency array
    Tempr = 10
    conc = 5.85e07

    # ------------------------------------------------------------

    wave_vector = np.linspace(0, 1.0e9, l_k)
    freq_array = np.linspace(-0.1 * gaas.Eg, 0.1 * gaas.Eg, l_f) / h

    # ------------------------------------------------------------

    Ef_h, Ef_e = bs.get_Fermi_levels(Tempr, conc)
    V = int_matrix(wave_vector, gaas.eps, dim=dim)
    # V = np.zeros((l_k, l_k))

    pulse_widths = 0.15e-14
    pulse_delay = 10 * pulse_widths
    pulse_amp = 1.0e29


    def e_field(t):
        a = pulse_amp * np.exp(-((t - pulse_delay) ** 2) / (2 * pulse_widths ** 2))  # * np.exp(1j*(0.0 * gaas.Eg / h) * t)
        return np.nan_to_num(a)


    flag = False
    subbandss = []
    ps = []

    for j1 in range(bs.n_sb_e):
        for j2 in range(bs.n_sb_h):
            print("The conduction subband index is {}, and the valence subband index is {}".format(j1, j2))
            subbands = bs.get_optical_transition_data(wave_vector, j2, j1)
            subbandss.append(subbands[2]-subbands[1])
            ps1 = polarization(freq_array, dim, gaas, subbands,
                               Ef_h, Ef_e,
                               Tempr,
                               V,
                               e_field, pulse_widths, pulse_delay, pulse_amp,
                               debug=True)

            ps1 = 2.0 * ps1

            # align absorption spectra according to positions of band edges
            if flag:

                pad_width = int((subbands[2][0] - subbands[1][0] - Eg) / ((freq_array[2] - freq_array[1]) * h))
                print(pad_width)
                if pad_width > 0:
                    ps1 = np.pad(ps1, pad_width, mode='edge')
                    ps1 = ps1[:np.size(freq_array)]

            else:
                flag = True
                Eg = subbands[2][0] - subbands[1][0]

            ps.append(ps1)

    ps_tot = np.sum(np.array(ps), axis=0)

    fig2 = plt.figure()
    for item in ps:
        plt.plot((freq_array * h + Eg) / e, item/1e5)
    plt.plot((freq_array * h + Eg) / e, ps_tot/1e5, 'k')
    plt.fill_between((freq_array * h + Eg) / e, ps_tot/1e5, facecolor='gray', alpha=0.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel(r'Absorption ($10^5$ m$^{-1}$)')
    plt.gca().legend(('heavy holes', 'light holes', 'total'))
    plt.show()

    return (freq_array * h + Eg) / e, ps_tot


if __name__ == '__main__':

    gaas = GaAs()
    bs = BandStructure3D(material=gaas)
    energy, ans = absorption(gaas, bs)
