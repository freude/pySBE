import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# pop_vv_vec=[1,0]


# dim=2
d_matrix = np.zeros((2, 2))
# print(d_matrix)

pvc_before_pulse = 0
pcv_before_pulse = 0
pvv_before_pulse = 1  # Ground state
pcc_before_pulse = 0  # excited state

matrix_before_pulse = [[pvc_before_pulse, pvv_before_pulse], [pcv_before_pulse, pcc_before_pulse]]
# print(matrix_before_pulse)


pvc_after_pulse = [1, 0]
pcv_after_pulse = np.transpose(pvc_after_pulse)

pvv_after_pulse = 0
pcc_after_pulse = 1  # excited state

# ------------------ time scale -------------------
ti = 0
tf = 20E-12  # s
N = 5000
tev = np.linspace(ti, tf, N)
# ---- constants and parameters of the system -----
hbar = 6.582119569E-16  # eV*s
ihbar = 1j * hbar
E_lvl_spacing = 1  # eV
trans_dipole_mom = 1
dephase_E = 1E-3  # eV
pop_cc_ = 0
pop_vv_i = 1

# -------- parameters of optical pulse -------------
Amp = 1
t0 = 10E-12  # seconds
pulse_width = 1000E-16  # seconds
# pulse_width=(pulse_width_fs)*(10**-15)
om = E_lvl_spacing / hbar  # Hz

# -------------- initial conditions ----------------
p0 = 0 + 0 * 1j


# function computing the EM pulse
def e_pulse(t, t0, om, Amp, pulse_width):
    return Amp * np.exp((-(t - t0) ** 2) / ((2 * pulse_width) ** 2)) * np.exp(1j * om * t)


# RHS of the equation (version 1)
def f(t, p, E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, om, Amp, pulse_width):
    dp = (1 / ihbar) * ((-1j * dephase_E - E_lvl_spacing) * p) + (1 / ihbar) * trans_dipole_mom * e_pulse(t, t0, om, Amp, pulse_width)
    # print(dp)
    return dp


# RHS of the equation (version 2)
def f1(t, p, E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, om, Amp, pulse_width):
    dp = (1 / ihbar) * ((-1j * dephase_E - E_lvl_spacing + om * hbar) * p) +\
         (1 / ihbar) * trans_dipole_mom * e_pulse(t, t0, om*0, Amp, pulse_width)
    # print(dp)
    return dp

# --------------------- solver ---------------------
sol = solve_ivp(f, (ti, tf), [p0], t_eval=tev, first_step=tf/N,
                args=(E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, om, Amp, pulse_width))

t = sol.t
p = sol.y
norm = np.max(np.abs(p[0]))

plt.plot(t/1e-12, e_pulse(t, t0, om, Amp, pulse_width))
plt.plot(t/1e-12, np.real(p[0])/norm)
plt.plot(t/1e-12, np.imag(p[0])/norm)
plt.legend(['EM pump pulse', 'Polarization, real part', 'Polarization, imaginary part'])
plt.xlabel("Time (ps)")
plt.ylabel("Polarization (a.u.)")
plt.show()