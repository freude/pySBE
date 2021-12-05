import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sbe.constants as const
from sbe.abstract_interfaces import BandStructure


def fd(energy, ef, tempr):
    """
    Fermi-Dirac function

    :param energy:   energy in eV
    :param ef:       Fermi level in eV
    :param temp:     temperature in K
    :return:         numpy.ndarray of the FD distribution
    """

    kb = 8.61733e-5  # Boltzmann constant in eV
    return 1.0 / (1.0 + np.exp((energy - ef) / (kb * tempr)))


class SemicondYAML(object):
    """
    The class is a data structure for the material parameters of a semiconductor

    Parameters are taken from
    [I. Vurgaftman, J. R. Meyer, and L. R. Ram-Mohan, J. Appl. Phys., 89 (11), 2001]
    """

    def __init__(self, **kwargs):

        self.dim = 3

        # --------------- band structure parameters ---------------

        tempr = kwargs.get('tempr', 0)
        self.tempr_dep = kwargs.get('tempr_dep', 'varshni')

        self.Eg = kwargs.get('Eg', 1.519)
        self.Eg *= const.e

        self.Eso = kwargs.get('Eso', -0.341)
        self.Eso *= const.e

        self.gamma1 = kwargs.get('gamma1', 6.98)
        self.gamma2 = kwargs.get('gamma2', 2.06)
        self.gamma3 = kwargs.get('gamma3', 2.93)

        self.gamma = 0.5 * (self.gamma2 + self.gamma2)

        self.me = kwargs.get('me', 0.0665)
        self.me *= const.m0                                   # electrons effective mass
        self.mhh = const.m0 / (self.gamma1 - 2 * self.gamma)  # holes effective mass
        self.mlh = const.m0 / (self.gamma1 + 2 * self.gamma)  # holes effective mass
        self.mso = kwargs.get('mso', 0.172)
        self.mso *= self.mso                                 # holes effective mass
        self.mh = (self.mhh, self.mlh, self.mso)

        # ----------------- dielectric screening ------------------

        self.eps = kwargs.get('eps', 12.93)                   # permitivity
        self.n_reff = kwargs.get('n_reff', 3.16)              # refraction index

        # ------------------- scaling constants -------------------

        self.mr = self.me / (self.mhh + self.me) * self.mhh
        self.a_0 = const.h / const.e * const.eps0 * self.eps * const.h / const.e / self.mr * 4 * const.pi
        self.E_0 = (const.e / const.eps0 / self.eps) * (const.e / (2 * self.a_0)) / const.e_Ry

        # --------------------- Varshni formula -------------------

        # Varshni perameters
        self.alpha = kwargs.get('alpha', 0.605)               # meV/K
        self.betha = kwargs.get('betha', 204)                 # K

        if self.tempr_dep == 'varshni':
            self.Eg = self.Eg - const.e * 0.001 * self.alpha * tempr * tempr / (tempr + self.betha)

        # ---------------  O’Donnell-Chen formula -----------------
        # ------------- Appl. Phys. Lett. 58 (25) (1991) ----------

        # O’Donnell-Chen formula
        self.coupling = 3.0  # meV/K
        self.phonon_energy = 26.7  # meV

        if self.tempr_dep == 'odonnell':
            self.Eg = self.Eg - const.e * 0.001 * self.phonon_energy * self.coupling * \
                      (1.0 / np.tanh(self.phonon_energy * const.e * 0.001 / (2 * const.kb * tempr)) - 1.0)

        # ------------ energy of momentum matrix element -----------
        # ----------- between conduction and valence bands ---------

        self.e_P = 28.8  # eV


class GaAs(object):
    """
    The class is a data structure for the material parameters of a semiconductor

    Parameters are taken from
    [I. Vurgaftman, J. R. Meyer, and L. R. Ram-Mohan, J. Appl. Phys., 89 (11), 2001]
    """

    def __init__(self, tempr=0, tempr_dep='varshni'):

        self.dim = 3

        self.tempr_dep = tempr_dep
        self.tempr = tempr

        # --------------- band structure parameters ---------------

        self.Eg = 1.519 * const.e  # nominal band gap
        self.Eso = -0.341 * const.e  # nominal band gap

        self.gamma1 = 6.98
        self.gamma2 = 2.06
        self.gamma3 = 2.93
        self.gamma = 0.5 * (self.gamma2 + self.gamma2)

        self.me = 0.0665 * const.m0  # electrons effective mass
        self.mhh = const.m0 / (self.gamma1 - 2 * self.gamma)  # holes effective mass
        self.mlh = const.m0 / (self.gamma1 + 2 * self.gamma)  # holes effective mass
        self.mso = 0.172 * const.m0  # holes effective mass
        self.mh = (self.mhh, self.mlh, self.mso)

        # ----------------- dielectric screening ------------------

        self.eps = 12.93  # permitivity
        self.n_reff = 3.61  # refraction index

        # ------------------- scaling constants -------------------

        self.mr = self.me / (self.mhh + self.me) * self.mhh
        self.a_0 = const.h / const.e * const.eps0 * self.eps * const.h / const.e / self.mr * 4 * const.pi
        self.E_0 = (const.e / const.eps0 / self.eps) * (const.e / (2 * self.a_0)) / const.e_Ry

        # --------------------- Varshni formula -------------------

        # Varshni perameters
        self.alpha = 0.605  # meV/K
        self.betha = 204  # K

        if self.tempr_dep == 'varshni':
            self.Eg = self.Eg - const.e * 0.001 * self.alpha * tempr * tempr / (tempr + self.betha)

        # ---------------  O’Donnell-Chen formula -----------------
        # ------------- Appl. Phys. Lett. 58 (25) (1991) ----------

        # O’Donnell-Chen formula
        self.coupling = 3.0  # meV/K
        self.phonon_energy = 26.7  # meV

        if self.tempr_dep == 'odonnell':
            self.Eg = self.Eg - const.e * 0.001 * self.phonon_energy * self.coupling * \
                      (1.0 / np.tanh(self.phonon_energy * const.e * 0.001 / (2 * const.kb * tempr)) - 1.0)

        # ------------ energy of momentum matrix element -----------
        # ----------- between conduction and valence bands ---------

        self.e_P = 28.8  # eV


class Tc(object):
    def __init__(self, dim=2):
        self.dim = dim

        self.Eg = 1.519 * const.e  # nominal band gap
        self.me = 2 * const.m0  # electrons effective mass
        self.mh = 2 * const.m0  # holes effective mass
        self.eps = 24.93  # permitivity
        self.n_reff = 3.61  # refraction index

        # ------------------- scaling constants -------------------

        self.mr = self.me / (self.mh + self.me) * self.mh
        self.a_0 = const.h / const.e * const.eps0 * self.eps * const.h / const.e / self.mr * 4 * const.pi
        self.E_0 = (const.e / const.eps0 / self.eps) * (const.e / (2 * self.a_0)) / const.e_Ry


class BandStructure3D(BandStructure, object):
    """
    Parabolic band approximation
    """

    def __init__(self, **kwargs):

        self.dim = 3

        self.mat = kwargs.get('material', GaAs())
        self.edges_c = np.array(kwargs.get('edges_c', np.array([0])))
        self.edges_v = np.array(kwargs.get('edges_v', np.array([0, 0])))

        self.edges_c = self.edges_c + self.mat.Eg

        self.n_sb_e = len(self.edges_c)
        self.n_sb_h = len(self.edges_v)

        self.ef = {}

    def _cond_band(self, j, k, units='eV'):

        if j >= self.n_sb_e:
            raise ValueError("Band index exceeds maximal value")

        if self.mat is not None:
            energy = self.edges_c[j] + const.h ** 2 * k ** 2 / (2 * self.mat.me)
        else:
            energy = None

        return energy

    def _val_band(self, j, k, units='eV'):

        if j >= self.n_sb_h:
            raise ValueError("Band index exceeds maximal value")

        if self.mat is not None:
            if j == 0:
                energy = self.edges_v[j] - const.h ** 2 * k ** 2 / (2 * self.mat.mhh)
            elif j == 1:
                energy = self.edges_v[j] - const.h ** 2 * k ** 2 / (2 * self.mat.mlh)
            elif j == 2:
                energy = self.edges_v[j] - const.h ** 2 * k ** 2 / (2 * self.mat.mso)
            else:
                energy = None
        else:
            energy = None

        return energy

    def _dipole(self, j1, j2, k, units='eV'):

        if j1 >= self.n_sb_h:
            raise ValueError("Band index exceeds maximal value")

        if j2 >= self.n_sb_e:
            raise ValueError("Band index exceeds maximal value")

        p = np.sqrt(const.e * self.mat.e_P * const.h / const.m0 * const.h / 2)
        d = p * self.mat.Eg / (self._cond_band(j2, k) - self._val_band(j1, k))

        return d

    def get_optical_transition_data(self, kk, j1, j2):
        return kk, self._val_band(j1, kk), self._cond_band(j2, kk), self._dipole(j1, j2, kk)

    def dos(self, energy, carriers="electrons"):

        dos = np.zeros(energy.shape)

        if carriers == "electrons":
            for j in range(self.n_sb_e):
                dos += _dos_single_subband(energy - self.edges_c[j] / const.e, self.mat.me, dim=self.dim)
        else:
            for j in range(self.n_sb_h):
                dos += _dos_single_subband(energy + self.edges_v[j] / const.e, self.mat.mh[j], dim=self.dim)

        return dos

    def get_Fermi_levels(self, tempr, dens):
        """
        Computes Fermi energy

        :param meff:    effective mass
        :param tempr:   temperature
        :param dens:    electron density
        :return:        Fermi energy in eV relative to the band edge
        """

        if tempr not in self.ef:

            cbb = np.min(self.edges_c) / const.e
            vbt = np.min(-self.edges_v) / const.e

            self.ef[tempr] = {}
            probe_fermi = np.linspace(self.mat.Eg / const.e * 0.5, cbb + 1.5, 350)
            conc = np.zeros(probe_fermi.shape)
            energy = np.linspace(cbb - 0.5, cbb + 5, 3000)
            for jj, ef in enumerate(probe_fermi):
                conc[jj] = np.trapz(fd(energy, ef, tempr) * self.dos(energy, carriers="electrons"),
                                    x=energy * const.e)

            self.ef[tempr]["elec"] = interpolate.interp1d(conc, probe_fermi, fill_value=0, kind='cubic')

            probe_fermi = np.linspace(-self.mat.Eg / const.e * 0.5, vbt + 1.0, 550)
            conc = np.zeros(probe_fermi.shape)
            energy = np.linspace(vbt, vbt + 10, 350)
            for jj, ef in enumerate(probe_fermi):
                conc[jj] = np.trapz(fd(energy, ef, tempr) * self.dos(energy, carriers="holes"),
                                    x=energy * const.e)

            self.ef[tempr]["holes"] = interpolate.interp1d(conc, probe_fermi, fill_value="extrapolate")

        ef_h = self.ef[tempr]["holes"](dens)
        ef_el = self.ef[tempr]["elec"](dens)

        return -ef_h * const.e, ef_el * const.e

    def plot(self):

        kk = np.linspace(0, 1.0, 100) * 3e9

        for j1 in range(self.n_sb_e):
            for j2 in range(self.n_sb_e):
                _, val, cond, dip = self.get_optical_transition_data(kk, j1, j2)

                plt.plot(kk, val / const.e)
                plt.plot(kk, cond / const.e)

        plt.show()


class BandStructure2D(BandStructure3D, object):
    """
    Parabolic band approximation
    """

    def __init__(self, **kwargs):
        super(BandStructure2D, self).__init__(**kwargs)
        self.dim = 2


def _dos_single_subband(energy, meff, dim=3, units='eV'):
    """
    Density of states of a single parabolic subband for 1, 2 and 3D electron gas

    :param energy:     energy array
    :param meff:       effective mass
    :param dim:        space dimensionality
    :param units:      energy units
    :return:           DOS array per a unit volume, area, or length per Joule
    """

    dos = np.zeros(energy.shape)

    # integrated space angle
    if dim == 1:
        omega_D = 2
    elif dim == 2:
        omega_D = 2 * np.pi
    else:
        omega_D = 4 * np.pi

    # units conversion
    if units == 'eV':
        alpha = const.e
    else:
        alpha = 1.0

    # Eq. (6.17) from [Haug, Koch, Quantum Theory of the Optical and
    # Electronic Properties of Semiconductors, 2004]
    dos += np.nan_to_num(omega_D / ((2 * np.pi) ** dim) *
                         ((2 * meff / const.h / const.h) ** (dim / 2)) *
                         ((energy * alpha) ** ((dim - 2) / 2)) *
                         (np.sign(energy) + 1.0) * 0.5)

    return dos


def get_list(obj):

    if isinstance(obj, list):
        return obj
    elif isinstance(obj, str):
        if os.path.getsize(obj) > 0:
            with open(obj, 'rb') as file:
                ans = pickle.load(obj)
            return ans
    else:
        raise ValueError('Wrong band structure data')


class BandStructure(object):
    """
    Object containing the information on the band structure
    """

    def __init__(self, mat=None, **kwargs):

        self.mat = mat
        self.dim = kwargs.get('dim', 1)
        val_file = kwargs.get('val_band', None)
        cond_file = kwargs.get('cond_band', None)
        dipoles_file = kwargs.get('dipoles', None)

        self.val_bands = get_list(val_file)
        self.cond_bands = get_list(cond_file)
        self.dipoles = get_list(dipoles_file)

        self.n_sb_e = len(self.cond_bands)
        self.n_sb_h = len(self.val_bands)

        self.ef = {}

    def _cond_band(self, j, k, units='eV'):

        if j >= self.n_sb_e:
            raise ValueError("Band index exceeds maximal value")

        return self.cond_bands[j](k) * const.e

    def _val_band(self, j, k, units='eV'):

        if j >= self.n_sb_h:
            raise ValueError("Band index exceeds maximal value")

        return self.val_bands[j](k) * const.e

    def _dipole(self, j1, j2, k, units='eV'):

        if j1 >= self.n_sb_h:
            raise ValueError("Band index exceeds maximal value")

        if j2 >= self.n_sb_e:
            raise ValueError("Band index exceeds maximal value")

        return (self.dipoles[j1][j2](k)) * 1e-31

    def get_optical_transition_data(self, kk, j1, j2):
        return kk, self._val_band(j1, kk), self._cond_band(j2, kk), self._dipole(j1, j2, kk)

    def dos(self, energy, carriers="electrons"):
        pass

    def get_Fermi_levels(self, tempr, dens):
        """
        Computes Fermi energy

        :param meff:    effective mass
        :param tempr:   temperature
        :param dens:    electron density
        :return:        Fermi energy in eV relative to the band edge
        """

        if tempr not in self.ef:

            cbb = np.min(self.edges_c) / const.e
            vbt = np.min(-self.edges_v) / const.e

            self.ef[tempr] = {}
            probe_fermi = np.linspace(self.mat.Eg / const.e * 0.5, cbb + 1.5, 50)
            conc = np.zeros(probe_fermi.shape)
            energy = np.linspace(cbb - 0.5, cbb + 5, 3000)
            for jj, ef in enumerate(probe_fermi):
                conc[jj] = np.trapz(fd(energy, ef, tempr) * self.dos(energy, carriers="electrons"),
                                    x=energy * const.e)

            self.ef[tempr]["elec"] = interpolate.interp1d(conc, probe_fermi, fill_value="extrapolate")

            probe_fermi = np.linspace(-self.mat.Eg / const.e * 0.5, vbt + 1.0, 550)
            conc = np.zeros(probe_fermi.shape)
            energy = np.linspace(vbt, vbt + 10, 350)
            for jj, ef in enumerate(probe_fermi):
                conc[jj] = np.trapz(fd(energy, ef, tempr) * self.dos(energy, carriers="holes"),
                                    x=energy * const.e)

            self.ef[tempr]["holes"] = interpolate.interp1d(conc, probe_fermi, fill_value="extrapolate")

        ef_h = self.ef[tempr]["holes"](dens)
        ef_el = self.ef[tempr]["elec"](dens)

        return -ef_h * const.e, ef_el * const.e

    def plot(self, kk=None):

        if kk is None:
            kk = np.linspace(0, 1.0, 100) * 7e9

        for j1 in range(self.n_sb_h):
            for j2 in range(self.n_sb_e):
                _, val, cond, dip = self.get_optical_transition_data(kk, j1, j2)

                plt.plot(kk, val / const.e)
                plt.plot(kk, cond / const.e)

        plt.show()


def get_Fermi_levels_2D(meff, tempr, dens):
    """
    Computes Fermi energy relative to the band edge
    for 2D quantum gas of fermions in the parabolic band approximation

    :param meff:    effective mass
    :param tempr:   temperature
    :param dens:    electron density
    :return:        Fermi energy in eV relative to the band edge
    """
    betha = 1.0 / const.kb / tempr
    aaa = const.h * betha * np.pi * dens * const.h / meff
    print(aaa)
    ef = const.kb / const.e * tempr * np.log(np.exp(aaa) - 1.0)
    return ef


def main():
    import matplotlib.pyplot as plt

    conc = np.array([1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17,
                     1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25])
    meff = GaAs().me
    tempr = 10
    ef = get_Fermi_levels_2D(meff, tempr, conc)
    plt.plot(conc, ef)
    plt.show()

    ef = []

    bs = BandStructure3D(material=GaAs(), edges_c=[0, 0.3], edges_v=[0, -0.1])
    k = np.linspace(0.0, 1e11, 100)
    d = bs._dipole(0, 0, k)
    for concentr in conc:
        ef_h, ef_el = bs.get_Fermi_levels(tempr, concentr)
        ef.append(ef_el)
    print(bs.mat.Eg / 1.6e-19)
    plt.plot(conc, np.array(ef) / 1.6e-19 - 0 * bs.mat.Eg / 1.6e-19)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def main1():
    import matplotlib.pyplot as plt

    bs = BandStructure(val_band='/home/mk/perovsk_project/val_band.pkl',
                       cond_band='/home/mk/perovsk_project/cond_band.pkl',
                       dipoles='/home/mk/perovsk_project/dipoles.pkl')
    k = np.linspace(0.0, 4e9, 100)

    _, vb, cb, d = bs.get_optical_transition_data(k, 0, 0)
    _, vb1, cb1, d = bs.get_optical_transition_data(k, 1, 1)
    _, vb2, cb2, d = bs.get_optical_transition_data(k, 2, 2)
    _, vb3, cb3, d = bs.get_optical_transition_data(k, 3, 3)

    plt.plot(k, vb)
    plt.plot(k, cb)
    plt.plot(k, vb1)
    plt.plot(k, cb1)
    plt.plot(k, vb2)
    plt.plot(k, cb2)
    plt.plot(k, vb3)
    plt.plot(k, cb3)
    plt.show()


if __name__ == '__main__':
    main1()
