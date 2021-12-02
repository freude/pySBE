from abc import ABC, abstractmethod


class BandStructure(ABC):
    """
    Parabolic band approximation
    """

    @abstractmethod
    def _cond_band(self, j, k, units='eV'):
        raise NotImplementedError

    @abstractmethod
    def _val_band(self, j, k, units='eV'):
        raise NotImplementedError

    @abstractmethod
    def _dipole(self, j1, j2, k, units='eV'):
        raise NotImplementedError

    @abstractmethod
    def get_optical_transition_data(self, kk, j1, j2):
        raise NotImplementedError


class AbstractPolarization(ABC):
    """
    Abstract class for the polarization
    """

    @abstractmethod
    def comp(self):
        raise NotImplementedError
