#from setuptools import setup
from numpy.distutils.core import setup
from numpy.distutils.core import Extension


setup(name='pySBE',
      version='1.0',
      authors = 'Mykhailo KLymenko, Rajavardhan Talashila',
      description='Semionductor Bloch equation solver',
      author_email='mike.klymenko@rmit.edu.au',
      license='MIT',
      packages=['apps', 'scripts', 'sbe'],
      ext_modules = [Extension( 'sbe.P_loop', ['src/P_loop.f90'] ), \
                     Extension( 'sbe.fft_loop', ['src/fft_loop.f90'] ),],
      zip_safe=False
      )
