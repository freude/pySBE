import time
import numpy as np
import matplotlib.pyplot as plt
import sbe.constants as const
from sbe.int_matrix import exchange
import sbe.P_loop as P_f2py_loop
import sbe.fft_loop as fft_f2py_loop


def polarization(fff, dim, params, bs, Ef_h, Ef_e, Tempr, V, damp, E_field, pulse_widths, pulse_delay, pulse_amp, e_phot, debug):

    # ----------------------- parse inputs -----------------------

    k = np.array(bs[0])
    Eh = bs[1] / const.h
    Ee = bs[2] / const.h
    mu = np.array(bs[3])

    l_k = np.size(k)    # length of k array
    l_f = np.size(fff)  # length of frequency array
    l_t = 10000          # length of time array
    stk = k[4] - k[3]   # step in the k-space grid

    eps = params.eps
    n_reff = params.n_reff

    # -------------------------- time ----------------------------

    t_min = 0.0  # min time
    t_max = 0.1e-11  # max time
    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # ------------------------------------------------------------

    omega = Ee - Eh
    Eg = const.h * omega[0]

    # ----------------- Distribution functions -------------------

    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 / (1 + np.exp(-(Eh * const.h - Ef_h) / const.kb / Tempr))

    # --------------------------- Exchange energy ----------------

    exce = exchange(k, ne, nh, V)

    # Call the Fortran routine to caluculate the required arrays.
    P, pp, ne_k, nh_k = P_f2py_loop.loop(dim, l_t, l_k, t, k, stt, stk,
                                         omega, Eg, exce, ne, nh,
                                         np.abs(mu)-np.abs(mu)+np.abs(mu)[0], damp, const.h, V,
                                         pulse_delay, pulse_widths, pulse_amp, e_phot)

    E_ft = E_field(t)
    PSr = fft_f2py_loop.loop(E_ft, l_t, l_f, stt, t, fff,
                             P, const.eps0, eps, Eg, const.h,
                             const.c, n_reff)

    # ---------------------- Visualization ----------------------

    if debug:
        from bokeh.io import output_file, show
        # fig = plt.figure(figsize=(11, 7), constrained_layout=True)
        # from matplotlib.gridspec import GridSpec
        #
        # gs = GridSpec(3, 5, figure=fig)
        # ax1 = fig.add_subplot(gs[:, 0])
        # ax2 = fig.add_subplot(gs[:, 1])
        # ax3 = fig.add_subplot(gs[:, 2])
        # ax4 = fig.add_subplot(gs[1, 3:])
        # ax5 = fig.add_subplot(gs[2, 3:])
        # ax6 = fig.add_subplot(gs[0, 3:])
        #
        # ax6.plot(t / 1e-12, np.real(E_ft) / np.max(np.abs(E_ft)))
        # ax6.plot(t / 1e-12, np.imag(E_ft) / np.max(np.abs(E_ft)))
        # ax6.set_xlabel('Time (ps)')
        # ax6.set_ylabel('Pump signal (a.u.)')
        #
        # ax4.plot(t / 1e-12, np.real(P) / np.max(np.abs(P)))
        # ax4.plot(t / 1e-12, np.imag(P) / np.max(np.abs(P)))
        # ax4.set_xlabel('Time (ps)')
        # ax4.set_ylabel('Polarization (a.u.)')
        #
        # ax3.contourf(k/1e9, t[l_k * 10: 0: -1]/1e-12, np.real(pp[l_k * 10: 0: -1, :]), 100)
        # ax3.set_xlabel(r'Wave vector (nm$^{-1}$)')
        # ax3.set_ylabel('Time (ps)')
        # ax3.title.set_text('Microscopic \n polarization')
        # # ax3.axis('off')
        #
        # # ax2.imshow(ne_k[l_k * 4: 0: -1, :])
        # ax2.contourf(k / 1e9, t[l_k * 10: 0: -1] / 1e-12, np.real(ne_k[l_k * 10: 0: -1, :]), 100)
        # ax2.set_xlabel(r'Wave vector (nm$^{-1}$)')
        # ax2.set_ylabel('Time (ps)')
        # ax2.title.set_text('Electron \n distribution')
        # # ax2.axis('off')
        #
        # # ax1.imshow(nh_k[l_k * 4: 0: -1, :])
        # ax1.contourf(k / 1e9, t[l_k * 10: 0: -1] / 1e-12, np.real(nh_k[l_k * 10: 0: -1, :]), 100)
        # ax1.set_xlabel(r'Wave vector (nm$^{-1}$)')
        # ax1.set_ylabel('Time (ps)')
        # ax1.title.set_text('Hole \n distribution')
        # # ax1.axis('off')
        #
        # ax5.plot(fff * const.h / const.e / 0.0042, PSr / np.max(PSr))
        # ax5.set_xlabel('Scaled energy (E-Eg)/Eb (a.u.)')
        # ax5.set_ylabel('Absorption (a.u.)')
        # # plt.pause(5)
        # # plt.draw()

        from bokeh.plotting import figure
        from bokeh.layouts import layout
        from bokeh.palettes import PiYG, Magma, Spectral
        ax6 = figure(x_axis_label='Time (ps)', y_axis_label='Pump signal (a.u.)',
                     plot_width=370, plot_height=190)
        ax6.line(t / 1e-12, np.real(E_ft) / np.max(np.abs(E_ft)))
        ax6.line(t / 1e-12, np.imag(E_ft) / np.max(np.abs(E_ft)))
        ax6.toolbar.logo = None

        ax4 = figure(x_axis_label='Time (ps)', y_axis_label='Polarization (a.u.)',
                     plot_width=370, plot_height=190)
        ax4.line(t / 1e-12, np.real(P) / np.max(np.abs(P)))
        ax4.line(t / 1e-12, np.imag(P) / np.max(np.abs(P)))
        ax4.toolbar.logo = None

        ax3 = figure(title='Microscopic \n polarization',
                     x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                     y_axis_label='Time (ps)',
                     sizing_mode="stretch_both")
        ax3.image(image=[np.real(pp)], x=0, y=0, dw=2, dh=2e9, palette=PiYG[10])
        ax3.toolbar.logo = None

        ax2 = figure(title='Electron \n distribution',
                     x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                     y_axis_label='Time (ps)',
                     sizing_mode="stretch_both")
        ax2.image(image=[np.real(ne_k)], x=0, y=0, dw=2, dh=2, palette=Spectral[10])
        ax2.toolbar.logo = None

        ax1 = figure(title='Hole \n distribution',
                     x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                     y_axis_label='Time (ps)',
                     sizing_mode="stretch_both")
        ax1.image(image=[np.real(nh_k)], x=0, y=0, dw=2, dh=2, palette=Spectral[10])
        ax1.toolbar.logo = None

        ax5 = figure(x_axis_label='Scaled energy (E-Eg)/Eb (a.u.)', y_axis_label='Absorption (a.u.)',
                     plot_width=370, plot_height=190)
        ax5.line(fff * const.h / const.e / 0.0042, PSr / np.max(PSr))
        ax5.toolbar.logo = None

        fig = layout([[ax1, ax2, ax3, [ax6, ax4, ax5]]], width=1150, height=770)

        from bokeh.document import Document
        import json

        # doc = Document()
        # doc.add_root(fig)
        doc_json = fig.to_json(True)
        with open('data.txt', 'w') as outfile:
            json.dump(doc_json, outfile)

    return PSr / (fff + Eg / const.h), fig


def polarization_app(fff, dim, params, bs, Ef_h, Ef_e, Tempr, V, damp, E_field, pulse_widths, pulse_delay, pulse_amp, e_phot, debug):

    # ----------------------- parse inputs -----------------------

    k = np.array(bs[0])
    Eh = bs[1] / const.h
    Ee = bs[2] / const.h
    mu = np.array(bs[3])

    l_k = np.size(k)    # length of k array
    l_f = np.size(fff)  # length of frequency array
    l_t = 10000          # length of time array
    stk = k[4] - k[3]   # step in the k-space grid

    eps = params.eps
    n_reff = params.n_reff

    # -------------------------- time ----------------------------

    t_min = 0.0  # min time
    t_max = 0.1e-11  # max time
    t = np.linspace(t_min, t_max, l_t)
    stt = t[3] - t[2]

    # ------------------------------------------------------------

    omega = Ee - Eh
    Eg = const.h * omega[0]

    # ----------------- Distribution functions -------------------

    ne = 1.0 / (1 + np.exp((Ee * const.h - Ef_e) / const.kb / Tempr))
    nh = 1.0 / (1 + np.exp(-(Eh * const.h - Ef_h) / const.kb / Tempr))

    # --------------------------- Exchange energy ----------------

    exce = exchange(k, ne, nh, V)

    # Call the Fortran routine to caluculate the required arrays.
    P, pp, ne_k, nh_k = P_f2py_loop.loop(dim, l_t, l_k, t, k, stt, stk,
                                         omega, Eg, exce, ne, nh,
                                         np.abs(mu)-np.abs(mu)+np.abs(mu)[0], damp, const.h, V,
                                         pulse_delay, pulse_widths, pulse_amp, e_phot)

    E_ft = E_field(t)
    PSr = fft_f2py_loop.loop(E_ft, l_t, l_f, stt, t, fff,
                             P, const.eps0, eps, Eg, const.h,
                             const.c, n_reff)

    # ---------------------- Visualization ----------------------

    if debug:
        from bokeh.io import output_file, show
        # fig = plt.figure(figsize=(11, 7), constrained_layout=True)
        # from matplotlib.gridspec import GridSpec
        #
        # gs = GridSpec(3, 5, figure=fig)
        # ax1 = fig.add_subplot(gs[:, 0])
        # ax2 = fig.add_subplot(gs[:, 1])
        # ax3 = fig.add_subplot(gs[:, 2])
        # ax4 = fig.add_subplot(gs[1, 3:])
        # ax5 = fig.add_subplot(gs[2, 3:])
        # ax6 = fig.add_subplot(gs[0, 3:])
        #
        # ax6.plot(t / 1e-12, np.real(E_ft) / np.max(np.abs(E_ft)))
        # ax6.plot(t / 1e-12, np.imag(E_ft) / np.max(np.abs(E_ft)))
        # ax6.set_xlabel('Time (ps)')
        # ax6.set_ylabel('Pump signal (a.u.)')
        #
        # ax4.plot(t / 1e-12, np.real(P) / np.max(np.abs(P)))
        # ax4.plot(t / 1e-12, np.imag(P) / np.max(np.abs(P)))
        # ax4.set_xlabel('Time (ps)')
        # ax4.set_ylabel('Polarization (a.u.)')
        #
        # ax3.contourf(k/1e9, t[l_k * 10: 0: -1]/1e-12, np.real(pp[l_k * 10: 0: -1, :]), 100)
        # ax3.set_xlabel(r'Wave vector (nm$^{-1}$)')
        # ax3.set_ylabel('Time (ps)')
        # ax3.title.set_text('Microscopic \n polarization')
        # # ax3.axis('off')
        #
        # # ax2.imshow(ne_k[l_k * 4: 0: -1, :])
        # ax2.contourf(k / 1e9, t[l_k * 10: 0: -1] / 1e-12, np.real(ne_k[l_k * 10: 0: -1, :]), 100)
        # ax2.set_xlabel(r'Wave vector (nm$^{-1}$)')
        # ax2.set_ylabel('Time (ps)')
        # ax2.title.set_text('Electron \n distribution')
        # # ax2.axis('off')
        #
        # # ax1.imshow(nh_k[l_k * 4: 0: -1, :])
        # ax1.contourf(k / 1e9, t[l_k * 10: 0: -1] / 1e-12, np.real(nh_k[l_k * 10: 0: -1, :]), 100)
        # ax1.set_xlabel(r'Wave vector (nm$^{-1}$)')
        # ax1.set_ylabel('Time (ps)')
        # ax1.title.set_text('Hole \n distribution')
        # # ax1.axis('off')
        #
        # ax5.plot(fff * const.h / const.e / 0.0042, PSr / np.max(PSr))
        # ax5.set_xlabel('Scaled energy (E-Eg)/Eb (a.u.)')
        # ax5.set_ylabel('Absorption (a.u.)')
        # # plt.pause(5)
        # # plt.draw()

        num_points = 300

        step = l_t // num_points

        from bokeh.plotting import figure
        from bokeh.layouts import layout
        from bokeh.palettes import PiYG, Magma, Spectral
        ax6 = figure(x_axis_label='Time (ps)', y_axis_label='Pump signal (a.u.)',
                     plot_width=370, plot_height=190)
        ax6.line(t / 1e-12, np.real(E_ft) / np.max(np.abs(E_ft)))
        ax6.line(t / 1e-12, np.imag(E_ft) / np.max(np.abs(E_ft)))
        ax6.toolbar.logo = None

        ax4 = figure(x_axis_label='Time (ps)', y_axis_label='Polarization (a.u.)',
                     plot_width=370, plot_height=190)
        ax4.line(t / 1e-12, np.real(P) / np.max(np.abs(P)))
        ax4.line(t / 1e-12, np.imag(P) / np.max(np.abs(P)))
        ax4.toolbar.logo = None

        ax3 = figure(title='Microscopic \n polarization',
                     x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                     y_axis_label='Time (ps)',
                     sizing_mode="stretch_both")
        ax3.image(image=[np.real(pp)], x=0, y=0, dw=2, dh=2e9, palette=PiYG[10])
        ax3.toolbar.logo = None

        ax2 = figure(title='Electron \n distribution',
                     x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                     y_axis_label='Time (ps)',
                     sizing_mode="stretch_both")
        ax2.image(image=[np.real(ne_k)], x=0, y=0, dw=2, dh=2, palette=Spectral[10])
        ax2.toolbar.logo = None

        ax1 = figure(title='Hole \n distribution',
                     x_axis_label=f'Wave vector (nm\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})',
                     y_axis_label='Time (ps)',
                     sizing_mode="stretch_both")
        ax1.image(image=[np.real(nh_k)], x=0, y=0, dw=2, dh=2, palette=Spectral[10])
        ax1.toolbar.logo = None

        ax5 = figure(x_axis_label='Scaled energy (E-Eg)/Eb (a.u.)', y_axis_label='Absorption (a.u.)',
                     plot_width=370, plot_height=190)
        ax5.line(fff * const.h / const.e / 0.0042, PSr / np.max(PSr))
        ax5.toolbar.logo = None

        fig = layout([[ax1, ax2, ax3, [ax6, ax4, ax5]]], width=1150, height=770)

        from bokeh.document import Document
        import json

        # doc = Document()
        # doc.add_root(fig)
        doc_json = fig.to_json(True)
        with open('data.txt', 'w') as outfile:
            json.dump(doc_json, outfile)

    return PSr / (fff + Eg / const.h),\
           (t / 1e-12, \
           fff * const.h / const.e / 0.0042, \
           np.max(k) / 1e9, \
           PSr / (fff + Eg / const.h) / np.max(PSr) * 1e16, \
           np.real(E_ft) / np.max(np.abs(E_ft)), \
           np.imag(E_ft) / np.max(np.abs(E_ft)), \
           np.real(P) / np.max(np.abs(P)), \
           np.imag(P) / np.max(np.abs(P)), \
           np.real(pp[::step, :]), \
           ne_k[::step, :], \
           nh_k[::step, :])
