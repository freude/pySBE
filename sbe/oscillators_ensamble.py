from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from sbe.aux_functions import import_check
import sbe.oscillators


def listify(ans):
    if not isinstance(ans, (list, np.ndarray)):
        ans = [ans]
    return ans


class AbsOscillators(ABC):

    @abstractmethod
    def add(self, omega=0, mu=1, nh=0, ne=0, k=0):
        raise NotImplemented

    @abstractmethod
    def setup_solver(self, backend, field):
        raise NotImplemented

    @abstractmethod
    def time_propagation(self, time):
        raise NotImplemented


@import_check("sbe.oscillators")
class OscillatorsNative(AbsOscillators):

    def __init__(self):

        self.num = None                   # number of oscillators
        self.omega = None                 # frequncies
        self.mu = None                    # initial phases
        self.nh = None
        self.ne = None
        self.k = None
        self.is_set = False

        self.dim = None
        self.t = None
        self.damp = None
        self.V = None

        self.pulse_delay = None
        self.pulse_widths = None
        self.pulse_amp = None
        self.e_phot = None

    def add(self, omega=0, mu=1, nh=0, ne=0, k=0):
        self.omega = listify(omega)
        self.mu = listify(mu)
        self.nh = listify(nh)
        self.ne = listify(ne)
        self.k = listify(k)
        self.num = len(omega)

    def setup_solver(self, **kwargs):

        self.dim = kwargs.get('dim', 1)
        self.t = kwargs.get('time', 1)
        self.damp = kwargs.get('damp', 1)
        self.V = kwargs.get('V', 1)

        self.pulse_delay = kwargs.get('pulse_delay', 1)
        self.pulse_widths = kwargs.get('pulse_widths', 1)
        self.pulse_amp = kwargs.get('pulse_amp', 1)
        self.e_phot = kwargs.get('e_phot', 1)

    def time_propagation(self, time, dim=1, V=0):

        if not self.is_set:
            raise ValueError("The solver is not set")

        if self.num == 0:
            raise ValueError("There is no oscillators in the amsable.")

        P, pp, ne_k, nh_k = sbe.oscillators.oscillators(dim, self.t, len(self.t),
                                                        self.k, len(self.k), self.omega, self.ne, self.nh, self.mu,
                                                        self.damp, self.V,
                                                        self.pulse_delay,
                                                        self.pulse_widths,
                                                        self.pulse_amp,
                                                        self.e_phot)

    def report_parameters(self):
        pass


@import_check('qutip')
class OscillatorsQuTiP(AbsOscillators):

    def add(self, omega=0, mu=1, nh=0, ne=0):

        self.omega = listify(omega)
        self.mu = listify(mu)
        self.nh = listify(nh)
        self.ne = listify(ne)


    def setup_solver(self, backend, field):
        self.is_set = True

    def time_propagation(self, time):

        if not self.is_set:
            raise ValueError("The solver is not set")

        if self.num == 0:
            raise ValueError("There is no oscillators in the amsable.")


class Oscillators(object):

    def __init__(self):
        backends = AbsOscillators.__subclasses__()
        names = [item.__name__ for item in backends]
        self.backends = dict(zip(names, backends))

    def make_ansamble(self, name):
        return self.backends[name]

    def avalible_backends(self):
        print(*self.backends.keys())


if __name__ == '__main__':

    Oscillators().avalible_backends()
