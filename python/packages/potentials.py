
import inspect
import matplotlib.pyplot as plt
import numpy as np
import sys
import textwrap
from functools import wraps
from abc import ABCMeta, abstractmethod, abstractproperty

# i-PI uses atomic units
#       Bohr Radii (a0),    Hartree (Ha)
# Sane people use (what I call) standard units
#       Angstrom (AA),      ElectronVolt (eV)
# And I don't like moles either

L = 6.022E23                # Avogadro's Number
bohr2angs = 0.52918         # Length conversion
angs2bohr = 1. / bohr2angs
har2ev = 27.2114            # Energy conversion
ev2har = 1. / har2ev
foc2au = ev2har * bohr2angs # Force conversion
au2foc = 1. / foc2au
kJ2eV = 6.241509125E21

sigma_TIP5P = 3.12 * angs2bohr

class PotentialEnergySurface1D(object):
    '''Abstract class for applying a one-dimensional confining potential to simulate two walls.
    
    Units are in Atomic Units (because that's what i-PI uses, I would have chosen otherwise.
    
    Attributes:
        w (float): Width of confinement; Distance between the two walls
        c (float): Height of simulation cell
        z0 (float): Position of lower wall
        z1 (float): Position of upper wall
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self, w, c):
        self.w = float(w)
        self.update_cell(c)
    
    def update_cell(self, c):
        if self.w >= c / 2.:
            message = 'Confinement width must be less than half of cell height\n'
            message += 'Conf. width = {} Angs; height / 2 = {} Angs'.format(self.w*bohr2angs, c*bohr2angs)
            raise ValueError(message)
        self.c = float(c)
        self.z0 = (self.c - self.w)/2.
        self.z1 = (self.c + self.w)/2.
    
    def pbc(self, z):
        while np.any(z < 0):
            z += self.c * (z < 0).astype(int)
        while np.any(z > self.c):
            z -= self.c * (z > self.c).astype(int)
        return z
    
    def check_confined(self, z):
        if np.any(z <= self.z0) or np.any(z >= self.z1):
            n0 = np.sum((z <= self.z0).astype(int))
            n1 = np.sum((z >= self.z1).astype(int))
            message = '{} atoms have gone below bottom wall; {} atoms have gone above top wall'.format(n0, n1)
            raise ValueError(message)
   
    def confine(function):
        @wraps(function)
        def confine_wrapper(self, z, checked_confined=False):
            if not checked_confined:
                self.check_confined(z)
            return function(self, z)
        return confine_wrapper
    
    @abstractmethod
    def potential_form(self, dz):
        pass
    
    @abstractmethod
    def gradient_form(self, dz):
        pass
    
    @abstractproperty
    def effective_ow(self):
        pass
    
    @abstractproperty
    def r_eq(self):
        pass
   
    @property
    def effective_width(self):
        return self.w - (self.effective_ow + sigma_TIP5P)/2.
    
    @property
    def effective_width_su(self):
        return self.effective_width * bohr2angs
    
    @property
    def effective_ow_su(self):
        return self.effective_ow * bohr2angs
    
    @property
    def r_eq_su(self):
        return self.r_eq * bohr2angs
    
    def potential_lower(self, z):
        return self.potential_form(z - self.z0)
    
    def potential_upper(self, z):
        return self.potential_form(self.z1 - z)
    
    def gradient_lower(self, z):
        return self.gradient_form(z - self.z0)
    
    def gradient_upper(self, z):
        return -self.gradient_form(self.z1 - z)
    
    @confine
    def potential(self, z):
        return self.potential_lower(z) + self.potential_upper(z)
    
    @confine
    def gradient(self, z):
        return self.gradient_lower(z) + self.gradient_upper(z)
    
    def force(self, z):
        return -self.gradient(z, checked_confined)
    
    def update_cell_su(self, c):
        self.update_cell(c * angs2bohr)
    
    def pbc_su(self, z):
        return self.pbc(z * angs2bohr)
    
    def potential_form_su(self, z):
        return self.potential_form(z * angs2bohr) * har2ev
    
    def gradient_form_su(self, z):
        return self.gradient_form(z * angs2bohr) * au2foc
    
    def potential_su(self, z, checked_confined=False):
        return self.potential(z * angs2bohr, checked_confined) * har2ev
    
    def gradient_su(self, z, checked_confined=False):
        return self.gradient(z * angs2bohr, checked_confined) * au2foc
    
    def force_su(self, z, checked_confined=False):
        return self.force(z * angs2bohr, checked_confined) * au2foc 

class Morse1D(PotentialEnergySurface):
    
    def __init__(self, w, c, au=True):
        # Parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        if not au:
            w *= angs2bohr
            c *= angs2bohr
        self.zeta = 3.85 * angs2bohr
        self.a = 0.92 / angs2bohr       # 0.92 AA^-1 converted to a0^-1
        self.D = 57.8E-3 * ev2har
        super(Morse1D, self).__init__(w, c)
    
    def potential_form(self, dz):
        return self.D * ((1. - np.exp(-self.a * (dz - self.zeta)))**2 - 1.) 
    
    def gradient_form(self, dz):
        x = np.exp(-self.a * (dz - self.zeta))
        return 2 * self.a * self.D * x * (1. - x)
    
    @property
    def r_eq(self):
        return self.zeta
    
    @property
    def effective_ow(self):
        #return self.zeta
        return self.zeta - np.log(2) / self.a

class LennardJones1D(PotentialEnergySurface):
    
    def __init__(self, w, c, au=True):
        # Parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        if not au:
            w *= angs2bohr
            c *= angs2bohr
        self.sigma = 3.0 * angs2bohr
        self.factor = 5./12.
        self.epsilon = 21.7E-3 * ev2har
        super(LennardJones1D, self).__init__(w, c)
    
    def potential_form(self, dz):
        x = (self.sigma / dz)**3
        return self.epsilon * (self.factor*x**3 - x) 
    
    def gradient_form(self, dz):
        return self.epsilon * (-self.factor*9.*self.sigma**9/dz**10 + 3.*self.sigma**3/dz**4) 
    
    @property
    def r_eq(self):
        return np.power(3. * self.factor, 1./6.) * self.sigma
    
    @property
    def effective_ow(self):
        return np.power(self.factor, 1./6.) * self.sigma

class LennardJones1DStanley(PotentialEnergySurface):
    
    def __init__(self, w, c, au=True):
        # Parameters from Kumar et al., Phys. Rev. E, 72, 5, 051503 (2005)
        if not au:
            w *= angs2bohr
            c *= angs2bohr
        self.sigma = 2.5 * angs2bohr
        self.epsilon = 1.25 / L * kJ2eV * ev2har # 1.25 kJ/mol, converted to Ha
        super(LennardJones1DStanley, self).__init__(w, c)
    
    def potential_form(self, dz):
        x = (self.sigma / dz)**3
        return 4 * self.epsilon * (x**3 - x) 
    
    def gradient_form(self, dz):
        return 4 * self.epsilon * (-9.*self.sigma**9/dz**10 + 3.*self.sigma**3/dz**4) 
    
    @property
    def r_eq(self):
        return np.power(6, 1./6.) * self.sigma
    
    @property
    def effective_ow(self):
        return self.sigma

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
            w_eff = pot.effective_width
        else:
            V = pot.potential_su(r, checked_confined=True)*1E3
            w_eff = pot.effective_width_su
        V -= np.min(V)
        ax.plot(r, V, c=colors[i], label=name)
        
        ax.axvline(c/2.+w_eff/2., c=colors[i], linestyle='--', alpha=0.2)
        ax.axvline(c/2.-w_eff/2., c=colors[i], linestyle='--', alpha=0.2)
        i += 1
    ax.set_ylim([0, 1500])
    ax.legend()
    plt.show()

