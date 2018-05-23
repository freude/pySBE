import numpy as np
import matplotlib.pyplot as plt
from constants import *
from int_matrix import int_matrix
from polarization import polarization


class GaAs(object):
    """
    The class is a data structure for the material parameters of a semiconductor

    Parameters are taken from
    [I. Vurgaftman, J. R. Meyer, and L. R. Ram-Mohan, J. Appl. Phys., 89 (11), 2001]
    """

    def __init__(self, dim=2):

        self.dim = dim

        self.Eg = 1.519 * e       # nominal band gap
        self.me = 0.067 * m0      # electrons effective mass
        self.mh = 0.5 * m0        # holes effective mass
        self.eps = 12.93          # permitivity
        self.n_reff = 3.61        # refraction index

        # ------------------- scaling constants -------------------

        self.mr = self.me / (self.mh + self.me) * self.mh
        self.a_0 = h / e * eps0 * self.eps * h / e / self.mr * 4 * np.pi
        self.E_0 = (e / eps0 / self.eps) * (e / (2 * self.a_0)) / e_Ry


class BandStructure(object):

    def __init__(self, **kwargs):

        self.mat = kwargs.get('material', None)

        if self.mat is not None:
            self.n_sb_e = 1
            self.n_sb_h = 1
        else:
            self.n_sb_e = 1
            self.n_sb_h = 1

    def _cond_band(self, j, k, units='eV'):

        if j >= self.n_sb_e:
            raise ValueError("Band index exceeds maximal value")

        if self.mat is not None:
            energy = self.mat.Eg + h**2 * k**2 / (2 * self.mat.me)
        else:
            energy = None

        return energy

    def _val_band(self, j, k, units='eV'):

        if j >= self.n_sb_e:
            raise ValueError("Band index exceeds maximal value")

        if self.mat is not None:
            energy = h**2 * k**2 / (2 * self.mat.mh)
        else:
            energy = None

        return energy

    def _dipole(self, j1, j2, k, units='eV'):

        if j1 >= self.n_sb_h:
            raise ValueError("Band index exceeds maximal value")

        if j2 >= self.n_sb_e:
            raise ValueError("Band index exceeds maximal value")

        return 1.0e-21 * np.ones(k.shape)

    def get_pairs_of_subbands(self, kk, j1, j2):
        return kk, self._val_band(j1, kk), self._cond_band(j2, kk), self._dipole(j1, j2, kk)

    def get_Fermi_levels(self, wave_vector, num_val_subbans, num_cond_subbans, Tempr, conc):

        return 0.5 * self.mat.Eg, 0.5 * self.mat.Eg


if __name__ == '__main__':

    semicond = GaAs()

    # -------------------- arrays defining ---------------------

    l_k = 300   # length of k array
    l_f = 500  # length of frequency array

    wave_vector = np.linspace(0, 1.5e9, l_k)
    freq_array = np.linspace(-0.05*semicond.Eg, 0.05*semicond.Eg, l_f)/h

    bs = BandStructure(material=semicond)

    Tempr = 10
    conc = 8.85e11

    Ef_h, Ef_e = bs.get_Fermi_levels(wave_vector, 1, 1, Tempr, conc)
    V = int_matrix(wave_vector, semicond.eps)
    # V = np.zeros((l_k, l_k))

    step = 0.01e-14

    def e_field(t):

        a = np.exp(-((t - 50 * step) ** 2) / (2 * step ** 2)) # * np.exp(1j*(0.0 * semicond.Eg / h) * t)

        return np.nan_to_num(a)


    ps = np.zeros(l_f)

    for j1 in range(bs.n_sb_e):
        for j2 in range(bs.n_sb_e):
            print("The conduction subband index is {}, and the valence subband index is {}".format(j1, j2))
            subbands = bs.get_pairs_of_subbands(wave_vector, j1, j2)
            ps += polarization(freq_array, semicond, subbands, Ef_h, Ef_e, Tempr, conc, V, e_field)

    plt.plot(ps)

    print('hi')



