
import numpy as np
from abc import ABCMeta, abstractmethod

# i-PI uses atomic units
#       Bohr Radii (a0),    Hartree (Ha)
# Sane people use (what I call) standard units
#       Angstrom (AA),      ElectronVolt (eV)
# And I don't like moles either

L = 6.022E23                # Avogadro's Number
bohr2angs = 0.52918         # Convert length to AU
angs2bohr = 1. / bohr2angs
har2ev = 27.2114            # Convert energy to AU
ev2har = 1. / har2ev
foc2au = ev2har * bohr2angs # Convert force to AU
au2foc = 1. / foc2au
kJ2eV = 6.241509125E21      # Convert energy to eV

class PotentialEnergySurface(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, w, c):
        self.w = w
        self.update_cell(c)
    
    def update_cell(self, c):
        if self.w >= c / 2:
            message = 'Confinement width must be less than half of cell height\n'
            message += 'Conf. width = {} Angs; height / 2 = {} Angs'.format(self.w*bohr2angs, c*bohr2angs)
            raise ValueError(message)
        self.c = c
        self.z0 = (self.c - self.w)/2
        self.z1 = (self.c + self.w)/2
    
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
            raise ValueError('{} molecules have gone over top wall and {} molecules have gone below bottom wall')
    
    @abstractmethod
    def potential(self, r, checked_confined=False):
        pass
    
    @abstractmethod
    def gradient(self, r, checked_confined=False):
        pass
    
    @abstractmethod
    def get_r_equil(self):
        pass
    
    def force(self, r, check_pbc=False):
        return -self.gradient(r, checked_confined)
    
    def update_cell_su(self, c):
        self.update_cell(c * angs2bohr)
    
    def pbc_su(self, z):
        return self.pbc(z * angs2bohr)
    
    def potential_su(self, r, checked_confined=False):
        return self.potential(r * angs2bohr, checked_confined) * au2foc
    
    def gradient_su(self, r, checked_confined=False):
        return self.gradient(r * angs2bohr, checked_confined) * au2foc
    
    def force_su(self, r, checked_confined=False):
        return self.force(r * angs2bohr, checked_confined) * au2foc
    
    def get_r_equil_su(self):
        return self.get_r_equil() * bohr2angs

class Morse1D(PotentialEnergySurface):
    
    def __init__(self, w, c, au=False):
        # parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        if not au:
            w /= bohr2angs
            c /= bohr2angs
        self.zeta = 3.85 * angs2bohr
        self.a = 0.92 * bohr2angs       # 0.92 AA^-1 converted to a0^-1
        self.D = 57.8E-3 * ev2har
        super(Morse1D, self).__init__(w, c)
    
    def potential(self, z, checked_confined=False):
        if not checked_confined:
            self.check_confined(z)
        V = np.zeros(z.shape)
        V = self.D * (( 1.0 - np.exp(-self.a*(z - self.z0 - self.zeta)))**2 +
                    (   1.0 - np.exp(-self.a*(self.z1 - z - self.zeta)))**2 - 2.0)
        return V
    
    def gradient(self, z, checked_confined=False):
        if not checked_confined:
            self.check_confined(z)
        F = np.zeros(z.shape)
        F = 2 * self.a * self.D * ( np.exp(-self.a*(z - self.z0 - self.zeta)) - np.exp(-2*self.a*(z - self.z0 - self.zeta)) - 
                                    np.exp(-self.a*(self.z1 - z - self.zeta)) + np.exp(-2*self.a*(self.z1 - z - self.zeta)))
        return F
   
    def get_r_equil(self):
        return self.zeta

class LennardJones1D(PotentialEnergySurface):
    
    def __init__(self, w, c, au=False):
        # parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        if not au:
            w /= bohr2angs
            c /= bohr2angs
        self.sigma = 3.0 / bohr2angs
        self.factor = 5./12.
        self.epsilon = 2.092 / L * kJ2eV * ev2har # 2.092 kJ/mol, converted to Ha
        self.w = w
        super(LennardJones1D, self).__init__(w, c)
    
    def potential(self, z, checked_confined=False):
        if not checked_confined:
            self.check_confined(z)
        V = np.zeros(z.shape)
        V = self.epsilon *  (self.factor*(self.sigma/(z - self.z0))**9 + (self.sigma/(z - self.z0))**3 + 
                             self.factor*(self.sigma/(self.z1 - z))**9 + (self.sigma/(self.z1 - z))**3)
        return V
    
    def gradient(self, z, checked_confined=False):
        if not checked_confined:
            self.check_confined(z)
        F = np.zeros(z.shape)
        F = self.epsilon *  (-self.factor*9.*self.sigma**9/(z - self.z0)**10 - 3.*self.sigma**3/(z - self.z0)**4 + 
                              self.factor*9.*self.sigma**9/(self.z1 - z)**10 + 3.*self.sigma**3/(self.z1 - z)**4)
        return F
    
    def get_r_equil(self):
        return self.sigma

