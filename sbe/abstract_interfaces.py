from abc import ABC, abstractmethod


class BandStructure(ABC):
    """
    Parabolic band approximation
    """

    @abstractmethod
    def _cond_band(self, j, k, units='eV'):
        return k - k

    @abstractmethod
    def _val_band(self, j, k, units='eV'):
        return k - k

    @abstractmethod
    def _dipole(self, j1, j2, k, units='eV'):
        return k - k

    @abstractmethod
    def get_optical_transition_data(self, kk, j1, j2):
        return kk, kk - kk, kk - kk, kk - kk
