sbe package
===========

Submodules
----------

sbe.Pol\_FFT\_f2py module
-------------------------

.. automodule:: sbe.Pol_FFT_f2py
    :members:
    :undoc-members:
    :show-inheritance:

sbe.absorption\_spectrum\_f2py module
-------------------------------------

.. automodule:: sbe.absorption_spectrum_f2py
    :members:
    :undoc-members:
    :show-inheritance:

sbe.absorption\_spectrum\_vec module
------------------------------------

.. automodule:: sbe.absorption_spectrum_vec
    :members:
    :undoc-members:
    :show-inheritance:

sbe.abstract\_interfaces module
-------------------------------

.. automodule:: sbe.abstract_interfaces
    :members:
    :undoc-members:
    :show-inheritance:

sbe.aux\_functions module
-------------------------

.. automodule:: sbe.aux_functions
    :members:
    :undoc-members:
    :show-inheritance:

sbe.constants module
--------------------

.. automodule:: sbe.constants
    :members:
    :undoc-members:
    :show-inheritance:

sbe.int\_matrix module
----------------------

The non-diagnonal elements that appear in the Rabi frequency are

.. math::
   
   \begin{align}
   \label{eq:nondiagonal matrix elements Rabi}
   \sum_{\mathbf{q\ne k}}V_{\left| \mathbf{k-q} \right| }p_{q}(t)&= \cfrac{e^{2}}{8\pi^{2}\epsilon}\int_{0}^{\infty}f_qqp_{q}dq \int_{0}^{2\pi} \cfrac{d\theta}{\sqrt{k^{2}+q^2-2kq\cos\theta}}\\
   &= \int_{0}^{\infty}V^{*}(k,q) p_{q} dq
   \end{align}

The value of :math:`V^*(k,q)` is the angle averaged Coulomb potential energy, defined by


.. math::

   \begin{align}
   \label{eq:Angle Averaged Coulomb Potential Energy}
   \color{red}
   { V^*(k,q) = \cfrac{e^{2}}{8\pi^{2}\epsilon}f_qq \int_{0}^{2\pi} \cfrac{d\theta}{\sqrt{k^{2}+q^2-2kq\cos\theta}}}
   \end{align}
   
.. automodule:: sbe.int_matrix
    :members:
    :undoc-members:
    :show-inheritance:

sbe.polarization\_f2py module
-----------------------------

.. automodule:: sbe.polarization_f2py
    :members:
    :undoc-members:
    :show-inheritance:

sbe.polarization\_vec module
----------------------------

.. automodule:: sbe.polarization_vec
    :members:
    :undoc-members:
    :show-inheritance:

sbe.semiconductors module
-------------------------

.. automodule:: sbe.semiconductors
    :members:
    :undoc-members:
    :show-inheritance:

sbe.stationary module
---------------------

.. automodule:: sbe.stationary
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: sbe
    :members:
    :undoc-members:
    :show-inheritance:
