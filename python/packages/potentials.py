
"""Module defining the PotentialEnergySurface class, used for calculating the potential
and force from two uniform, confining planes parallel to the z-axis.

Methods for plotting the shape of the confining well are also included.
"""

import inspect
import matplotlib.pyplot as plt
import numpy as np
import sys
import textwrap
from abc import ABCMeta, abstractmethod, abstractproperty
from functools import wraps
from scipy.optimize import fsolve

__author__ = "Johan G. McQuillan"
__email__ = "johan.mcquillan.13@ucl.ac.uk"

# i-PI uses atomic units
#       Bohr Radii (a0),    Hartree (Ha)
# I prefer (what I call) standard units, or su
#       Angstrom (AA),      ElectronVolt (eV)
# I am not partial to moles either

# Avogadro's Number
L = 6.022E23
# Length conversion
bohr2angs = 0.52918
angs2bohr = 1. / bohr2angs
# Energy conversion
har2ev = 27.2114
ev2har = 1. / har2ev
kJ2eV = 6.241509125E21
eV2kJ = 1. / kJ2eV
# Force conversion
foc2au = ev2har * bohr2angs
au2foc = 1. / foc2au

# Parameters for effective width calculations
sigma_TIP5P = 3.12 * angs2bohr
epsilon_stanley = 8./np.power(3., 3./2.) * 1.25 / L * kJ2eV * ev2har


class PotentialEnergySurface1D(object):
    """Abstract class for applying a one-dimensional confining potential to simulate two walls.

    Units are in Atomic Units (because that's what i-PI uses, I would have chosen otherwise).

    Attributes:
        w (float): Width of confinement; Distance between the two walls.
        c (float): Height of simulation cell.
        z0 (float): Position of lower wall.
        z1 (float): Position of upper wall.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, w, c):
        self.w = float(w)
        self.update_cell(c)
        self.ow = self.calc_ow()
    
    def update_cell(self, c):
        """Update height of cell, keeping the confinement width constant.
        
        The new height must be at least twice the width of confinement."""
        
        if self.w > c / 2.:
            message = 'Confinement width must be less than half of cell height\n'
            message += 'Conf. width = {} Angs; height / 2 = {} Angs'.format(self.w*bohr2angs, c*bohr2angs)
            raise ValueError(message)
        self.c = float(c)
        self.z0 = (self.c - self.w)/2.
        self.z1 = (self.c + self.w)/2.
    
    def pbc(self, z):
        """Apply periodic boundary conditions to reduce all coords to exist within the supercell."""
        
        while np.any(z < 0):
            z += self.c * (z < 0).astype(int)
        while np.any(z > self.c):
            z -= self.c * (z > self.c).astype(int)
        return z
    
    def check_confined(self, z):
        """Raise exception if any atoms have strained above upper wall or below lower wall.
        
        It may be important to run pbc on the coordinates first to stop false positives."""
        
        if np.any(z <= self.z0) or np.any(z >= self.z1):
            n0 = np.sum((z <= self.z0).astype(int))
            n1 = np.sum((z >= self.z1).astype(int))
            message = '{} atoms have gone below bottom wall; {} atoms have gone above top wall'.format(n0, n1)
            raise ValueError(message)
    
    def calc_ow(self):
        """Calculate the effective confinement width.
        
        This is given by the distance from the wall where the height of the potential above the
        bottom of the well is equal to the well depth from Stanley's potential."""
        
        # Use fsolve to solve equation
        function = lambda x: self.potential_form(x) - self.potential_min - epsilon_stanley
        guess = self.r_eq / 2.
        return fsolve(function, guess)
    
    def confine(function):
        """Run self.check_confined before calculating any forces or potentials."""
        @wraps(function)
        
        def confine_wrapper(self, z, checked_confined=False):
            if not checked_confined:
                self.check_confined(z)
            return function(self, z)
        return confine_wrapper
    
    @abstractmethod
    def potential_form(self, dz):
        """Return the potential of one wall at dz = 0 acting on particles at dz > 0."""
        
        pass
    
    @abstractmethod
    def gradient_form(self, dz):
        """Return the potential gradient of one wall at dz = 0 acting on particles at dz > 0."""
        
        pass
    
    @abstractproperty
    def potential_min(self):
        """Return minimum of the potential of single wall."""
        
        pass 
    
    @abstractproperty
    def r_eq(self):
        """Return the equilibrium distance of a single wall."""
        
        pass
    
    def potential_lower(self, z):
        """Return the potential due to the lower wall."""
        
        return self.potential_form(z - self.z0)
    
    def potential_upper(self, z):
        """Return the potential due to the upper wall."""
        
        return self.potential_form(self.z1 - z)
    
    def gradient_lower(self, z):
        """Return the potential gradient due to the lower wall."""
        
        return self.gradient_form(z - self.z0)
    
    def gradient_upper(self, z):
        """Return the potential gradient due to the upper wall."""
        
        return -self.gradient_form(self.z1 - z)
    
    @confine
    def potential(self, z):
        """Return the potential due to both walls."""
        
        return self.potential_lower(z) + self.potential_upper(z)
    
    @confine
    def gradient(self, z):
        """Return the potential gradient due to both walls."""
        
        return self.gradient_lower(z) + self.gradient_upper(z)
    
    def force(self, z, checked_confined=False):
        """Return the force due to both walls."""
        
        return -self.gradient(z, checked_confined)
    
    @property
    def w_eff(self):
        """Return the effective confinement width."""
        
        return self.w - (self.ow + sigma_TIP5P) / 2.
    
    @property
    def w_eff_su(self):
        """The effective confinement width in standard units."""
        return self.w_eff * bohr2angs
    
    @property
    def r_eq_su(self):
        """The equilibrium distance of a single wall in standard units."""
        return self.r_eq * bohr2angs
    
    @property
    def ow_su(self):
        return self.ow * bohr2angs
    
    def update_cell_su(self, c):
        """Update height of cell in standard units, keeping the confinement width constant.
        
        The new height must be at least twice the width of confinement."""
        self.update_cell(c * angs2bohr)
    
    def pbc_su(self, z):
        """Apply periodic boundary conditions to reduce all coords in standard units to exist within the supercell."""
        return self.pbc(z * angs2bohr)
    
    def potential_form_su(self, z):
        """Return the potential in standard units of one wall at dz = 0 acting on particles at dz > 0."""
        
        return self.potential_form(z * angs2bohr) * har2ev
    
    def gradient_form_su(self, z):
        """Return the potential gradient in standard units of one wall at dz = 0 acting on particles at dz > 0."""
        
        return self.gradient_form(z * angs2bohr) * au2foc
    
    def potential_su(self, z, checked_confined=False):
        """Return the potential in standard units due to both walls."""
        
        return self.potential(z * angs2bohr, checked_confined) * har2ev
    
    def gradient_su(self, z, checked_confined=False):
        """Return the potential gradient in stanndard units due to both walls."""
        
        return self.gradient(z * angs2bohr, checked_confined) * au2foc
    
    def force_su(self, z, checked_confined=False):
        """Return the force in standard units due to both walls."""
        
        return self.force(z * angs2bohr, checked_confined) * au2foc 


class Morse1D(PotentialEnergySurface1D):
    
    def __init__(self, w, c, au=True):
        # Parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        self.zeta = 3.85 * angs2bohr
        self.a = 0.92 / angs2bohr       # 0.92 AA^-1 converted to a0^-1
        self.D = 57.8E-3 * ev2har
        if au:
            super(Morse1D, self).__init__(w, c)
        else:
            super(Morse1D, self).__init__(w*angs2bohr, c*angs2bohr)
    
    def potential_form(self, dz):
        return self.D * ((1. - np.exp(-self.a * (dz - self.zeta)))**2 - 1.) 
    
    def gradient_form(self, dz):
        x = np.exp(-self.a * (dz - self.zeta))
        return 2 * self.a * self.D * x * (1. - x)
   
    @property
    def potential_min(self):
        return -self.D
    
    @property
    def r_eq(self):
        return self.zeta
    
    @property
    def effective_ow(self):
        #return self.zeta
        return self.zeta - np.log(2) / self.a


class LennardJones1D(PotentialEnergySurface1D):
    
    def __init__(self, w, c, au=True):
        # Parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        self.sigma = 3.0 * angs2bohr
        self.factor = 5./12.
        self.epsilon = 21.7E-3 * ev2har
        if au:
            super(LennardJones1D, self).__init__(w, c)
        else:
            super(LennardJones1D, self).__init__(w*angs2bohr, c*angs2bohr)
    
    def potential_form(self, dz):
        x = (self.sigma / dz)**3
        return self.epsilon * (self.factor*x**3 - x) 
    
    def gradient_form(self, dz):
        return self.epsilon * (-self.factor*9.*self.sigma**9/dz**10 + 3.*self.sigma**3/dz**4) 
    
    @property
    def potential_min(self):
        return -self.epsilon * 2. / (3. * np.sqrt(3. * self.factor))
    
    @property
    def r_eq(self):
        return np.power(3. * self.factor, 1./6.) * self.sigma
    
    @property
    def effective_ow(self):
        return np.power(self.factor, 1./6.) * self.sigma


class LennardJones1DStanley(PotentialEnergySurface1D):
    
    def __init__(self, w, c, au=True):
        # Parameters from Kumar et al., Phys. Rev. E, 72, 5, 051503 (2005)
        self.sigma = 2.5 * angs2bohr
        self.epsilon = 1.25 / L * kJ2eV * ev2har # 1.25 kJ/mol, converted to Ha
        if au:
            super(LennardJones1DStanley, self).__init__(w, c)
        else:
            super(LennardJones1DStanley, self).__init__(w*angs2bohr, c*angs2bohr)
    
    def potential_form(self, dz):
        x = (self.sigma / dz)**3
        return 4 * self.epsilon * (x**3 - x) 
    
    def gradient_form(self, dz):
        return 4 * self.epsilon * (-9.*self.sigma**9/dz**10 + 3.*self.sigma**3/dz**4) 
    
    @property
    def potential_min(self):
        return -8./np.power(3., 3./2.) * self.epsilon
    
    def calc_ow(self):
        return self.sigma
    
    @property
    def r_eq(self):
        return np.power(3, 1./6.) * self.sigma


def potential_names():
    return [name for name, member in inspect.getmembers(sys.modules[__name__])
            if inspect.isclass(member) and name not in ['ABCMeta', 'abstractproperty', 'PotentialEnergySurface1D']]


def potential_text():
    text = ''
    for p in sorted(potential_names()):
        text += '\n    '+p
    return textwrap.dedent(text)


def help_text():
    return textwrap.dedent('Name of external potential - possible options are below:'+potential_text())


def get_potential(name):
    try:
        if name not in potential_names():
            raise KeyError
        potential = globals()[name]
        return potential
    except KeyError:
        raise ValueError('Must give valid external potential name. Options are:'+potential_text())


def plot_potentials(au=True):
    colors = ['b', 'g', 'r']
    
    r = np.linspace(0, 10, 200)[1:-1]
    
    fig, ax = plt.subplots(1)
    
    i = 0
    for name in potential_names():
        pot = get_potential(name)(5, 30., au=au)
        if au:
            V = pot.potential_form(r)
        else:
            V = pot.potential_form_su(r)*1E3
        ax.plot(r, V, c=colors[i], label=name)
        i += 1
    ax.axhline(0, c='k', alpha=0.2)
    ax.set_xlim([0, 10])
    ax.set_ylim([-150, 500])
    ax.set_xlabel(r'$r$ [$\AA$]')
    ax.set_ylabel(r'$V(r)$ [meV]')
    ax.legend()
    plt.show()


def plot_wells(w, c, au=True):
    colors = ['b', 'g', 'r']
    
    z0 = (c - w)/2.
    z1 = (c + w)/2.
    r = np.linspace(z0, z1, 200)[1:-1]
    fig, ax = plt.subplots(1)
    
    i = 0
    for name in potential_names():
        pot = get_potential(name)(w, c, au=au) 
        if au:
            V = pot.potential(r, checked_confined=True)
            w_eff = pot.w_eff
        else:
            V = pot.potential_su(r, checked_confined=True)*1E3
            w_eff = pot.w_eff_su
        V -= np.min(V)
        ax.plot(r, V, c=colors[i], label=name)
        
        ax.axvline(c/2.+w_eff/2., c=colors[i], linestyle='--', alpha=0.2)
        ax.axvline(c/2.-w_eff/2., c=colors[i], linestyle='--', alpha=0.2)
        i += 1
    ax.set_ylim([0, 1000])
    ax.set_xlabel(r'$z$ [$\AA$]')
    ax.set_ylabel(r'$V(z)$ [meV]')
    ax.legend()
    plt.show()


def plot_pot_well(w, au=True):
    colors = ['b', 'g', 'r']

    fig, axes = plt.subplots(2)
    ax1, ax2 = axes

    c = w*2.
    z0 = (c - w)/2.
    z1 = (c + w)/2.
    r = np.linspace(0, w, 200)[1:-1]
    z = np.linspace(z0, z1, 200)[1:-1]

    i = 0
    for name in potential_names():
        pot = get_potential(name)(w, c, au=au)
        if au:
            V1 = pot.potential_form(r)
            V2 = pot.potential(z, checked_confined=True)
            V2 -= np.min(V2)
            ow = pot.calc_ow()
            w_eff = pot.w_eff
        else:
            V1 = (pot.potential_form_su(r) - pot.potential_min*har2ev)*1E3
            V2 = pot.potential_su(z, checked_confined=True)*1E3
            V2 -= np.min(V2)
            ow = pot.calc_ow() * bohr2angs
            w_eff = pot.w_eff_su
        ax1.plot(r, V1, c=colors[i], label=name)
        ax2.plot(z-c/2., V2, c=colors[i], label=name)
        
        ax1.axvline(ow, c=colors[i], linestyle='--', alpha=0.2)
        ax2.axvline(+w_eff/2., c=colors[i], linestyle='--', alpha=0.2)
        ax2.axvline(-w_eff/2., c=colors[i], linestyle='--', alpha=0.2)
        i += 1
    ax1.set_ylim([-100, 500])
    ax2.set_ylim([-100, 500])
    ax1.axhline(0, c='k', alpha=0.2, linestyle=':')
    ax1.legend()
    plt.show()
