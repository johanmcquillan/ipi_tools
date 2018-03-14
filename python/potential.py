
import numpy as np
from abc import ABCMeta, abstractmethod

L = 6.022E23
bohr2angs = 0.52918
bohr2nm = bohr2angs / 10
ev2har = 1.0/27.2114
foc2au = ev2har*bohr2angs
kJ2eV = 6.241509125E21

class PotentialEnergySurface(object):
    __metaclass__ = ABCMeta

    def __init__(self, w, c):
        self.w = w / bohr2nm
        self.update_cell(c)

    def update_cell(self, c):
        assert self.w < c
        self.c = c
        self.z0 = (c - self.w)/2
        self.z1 = (c + self.w)/2

    def pbc(self, z):
        while np.any(z < 0):
            z += self.c * (z < 0).astype(int)
        while np.any(z > self.c):
            z -= self.c * (z > self.c).astype(int)
        return z
    
    @abstractmethod
    def potential(self, r):
        pass

    @abstractmethod
    def gradient(self, r):
        pass

class Morse1D(PotentialEnergySurface):
    
    def __init__(self, w, c):
        # parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        self.zeta = 3.85 / bohr2angs
        self.a = 0.92 * bohr2angs
        self.D = 57.8E-3 * ev2har
        super(Morse1D, self).__init__(w, c)

    def potential(self, z):
        if np.any(z <= self.z0) or np.any(z >= self.z1):
            n0 = np.sum((z <= self.z0).astype(int))
            n1 = np.sum((z >= self.z1).astype(int))
            raise ValueError('{} molecules have gone over top wall and {} molecules have gone below bottom wall'.format(n0, n1))
        V = np.zeros(z.shape)
        V = self.D * ((1.0 - np.exp(-self.a*(z - self.z0 - self.zeta)))**2 +
                    (1.0 - np.exp(-self.a*(self.z1 - z - self.zeta)))**2 - 2.0)
        return np.sum(V)

    def gradient(self, z):
        F = np.zeros(z.shape)
        F = 2 * self.a * self.D * (np.exp(-self.a*(z - self.z0 - self.zeta)) - np.exp(-2*self.a*(z - self.z0 - self.zeta)) - 
                                    np.exp(-self.a*(self.z1 - z - self.zeta)) + np.exp(-2*self.a*(self.z1 - z - self.zeta)))
        return F

class LennardJones1D(PotentialEnergySurface):
    
    def __init__(self, w, c):
        # parameters from Chen et al., Phys. Rev. B, 94, 22, 220102 (2016)
        self.zeta = 3.85 / bohr2angs
        self.factor = 5./12.
        self.epsilon = 2.092 / L * kJ2eV * ev2har # 2.092 kJ/mol, converted to Ha
        self.w = w / bohr2nm
        super(LennardJones1D, self).__init__(w, c)

    def potential(self, z):
        if np.any(z <= self.z0) or np.any(z >= self.z1):
            n0 = np.sum((z <= self.z0).astype(int))
            n1 = np.sum((z >= self.z1).astype(int))
            raise ValueError('{} molecules have gone over top wall and {} molecules have gone below bottom wall'.format(n0, n1))
        V = np.zeros(z.shape)
        V = self.epsilon *  (self.factor*(self.sigma/(z - self.z0))**9 + (self.sigma/(z - self.z0))**3 + 
                             self.factor*(self.sigma/(self.z1 - z))**9 + (self.sigma/(self.z1 - z))**3)
        return V

    def gradient(self, z):
        F = np.zeros(z.shape)
        F = self.epsilon *  (-self.factor*9.*self.sigma**9/(z - self.z0)**10 - 3.*self.sigma**3/(z - self.z0)**4 + 
                              self.factor*9.*self.sigma**9/(self.z1 - z)**10 + 3.*self.sigma**3/(self.z1 - z)**4)
        return F
