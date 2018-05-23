import numpy as np
import matplotlib.pyplot as plt
from constants import *
from int_matrix import int_matrix
from polarization import polarization
from semiconductors import GaAs
from semiconductors import BandStructure


semicond = GaAs()

# -------------------- arrays definition ---------------------

l_k = 300   # length of k array
l_f = 500  # length of frequency array

wave_vector = np.linspace(0, 1.5e9, l_k)
freq_array = np.linspace(-0.05 * semicond.Eg, 0.05 * semicond.Eg, l_f) /  h

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



