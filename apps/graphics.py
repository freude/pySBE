import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.palettes import PiYG, Magma, Spectral


plot.output_backend = "svg"


def make_fig(t, fff, max_k, absorb, real_sig, imag_sig, real_pol, imag_pol, pp, ne_k, nh_k):

    ax6 = figure(x_axis_label='Time (ps)', y_axis_label='Pump signal (a.u.)',
                 plot_width=370, plot_height=190)
    ax6.line(t, real_sig)              # t / 1e-12, np.real(E_ft) / np.max(np.abs(E_ft))
    ax6.line(t, imag_sig)              # t / 1e-12, np.imag(E_ft) / np.max(np.abs(E_ft))
    ax6.toolbar.logo = None

    ax4 = figure(x_axis_label='Time (ps)', y_axis_label='Polarization (a.u.)',
                 plot_width=370, plot_height=190)
    ax4.line(t, real_pol)  # t / 1e-12, np.real(P) / np.max(np.abs(P))
    ax4.line(t, imag_pol)   # t / 1e-12, np.imag(P) / np.max(np.abs(P))
    ax4.toolbar.logo = None

    ax3 = figure(title='Microscopic \n polarization',
                 x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                 y_axis_label='Time (ps)',
                 sizing_mode="stretch_both")
    ax3.image(image=[pp], x=0, y=0, dw=max_k, dh=np.max(t), palette=PiYG[10])
    ax3.toolbar.logo = None

    ax2 = figure(title='Electron \n distribution',
                 x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                 y_axis_label='Time (ps)',
                 sizing_mode="stretch_both")
    ax2.image(image=[ne_k], x=0, y=0, dw=max_k, dh=np.max(t), palette=Spectral[10])
    ax2.toolbar.logo = None

    ax1 = figure(title='Hole \n distribution',
                 x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                 y_axis_label='Time (ps)',
                 sizing_mode="stretch_both")
    ax1.image(image=[nh_k], x=0, y=0, dw=max_k, dh=np.max(t), palette=Spectral[10])
    ax1.toolbar.logo = None

    ax5 = figure(x_axis_label='Scaled energy (E-Eg)/Eb (a.u.)', y_axis_label='Absorption (a.u.)',
                 plot_width=370, plot_height=190)
    ax5.line(fff, absorb)   # fff * const.h / const.e / 0.0042, PSr / np.max(PSr)
    ax5.toolbar.logo = None

    return layout([[ax1, ax2, ax3, [ax6, ax4, ax5]]], width=1150, height=770)
