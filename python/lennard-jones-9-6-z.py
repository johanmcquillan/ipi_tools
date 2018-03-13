#!/usr/bin/env python

import socket
import struct
import sys
import numpy as np

# i-pi use annoying atomic units, make sure to be consistent
L = 6.022E23
bohr2angs = 0.52918
bohr2nm = bohr2angs / 10
ev2har = 1.0/27.2114
foc2au = ev2har*bohr2angs
kJ2eV = 6.241509125E21

have_data = False
run_flag = True

# open a socket
HDRLEN = 12
have_data = False
run_flag = True

# open a socket
if len(sys.argv) > 2:
    port = int(sys.argv[1])
    addr = sys.argv[2]
    fsoc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fsoc.connect((addr, port))
elif len(sys.argv) > 1:
    addr = "/tmp/ipi_"+sys.argv[1] # "ipiaddr" is a string you haveto make it consistent with ipi input
    fsoc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    fsoc.connect(addr)
else:
    assert 1 == 2

# PES object
class PES:
    def __init__(self, z0, z1, factor):
        self.z0 = z0 / bohr2nm
        self.z1 = z1 / bohr2nm
        self.factor = factor
        self.epsilon = 2.092 / L * kJ2eV * ev2har # 2.092 kJ/mol, converted to Ha
        self.sigma = 0.3 / bohr2nm
        self.c = 3.0 / bohr2nm

    def pbc(self, z):
        while np.any(z < 0):
            z += self.c * (z < 0).astype(int)
        while np.any(z > self.c):
            z -= self.c * (z > self.c).astype(int)
        return z

    def potential(self, z):
        assert np.all(z > self.z0)
        assert np.all(z < self.z1)
        Z = z[::3]
        V = np.zeros(z.shape)
        V[::3] = self.epsilon * (self.factor*(self.sigma/(Z - self.z0))**9 + (self.sigma/(Z - self.z0))**3 + 
                                 self.factor*(self.sigma/(self.z1 - Z))**9 + (self.sigma/(self.z1 - Z))**3)
        return V

    def gradient(self, z):
        Z = z[::3]
        F = np.zeros(z.shape)
        F[::3] = self.epsilon * (-self.factor*9.*self.sigma**9/(Z - self.z0)**10 - 3.*self.sigma**3/(Z - self.z0)**4 + 
                                  self.factor*9.*self.sigma**9/(self.z1 - Z)**10 + 3.*self.sigma**3/(self.z1 - Z)**4)
        return F

pes = PES(1.25, 1.75, 5./12.)

# force field Loop
while run_flag == True:
    msg = fsoc.recv(HDRLEN)
    
    if msg == "POSDATA     ":
        cell_h = fsoc.recv(9*8)
        cell_ih = fsoc.recv(9*8)
        
        nat = fsoc.recv(4)
        nat = struct.unpack("i", nat)[0]
        read_pos = fsoc.recv(nat*3*8)
        pos = np.zeros([nat,3])
        for atom in range(nat):
            for coord in range(3):
                pos[atom,coord] = struct.unpack("d", read_pos[atom*24+coord*8:atom*24+(coord+1)*8])[0]
        z_array = pes.pbc(pos[:, 2])
        
        V = np.sum(pes.potential(z_array))
        F = -pes.gradient(z_array)
        # V *= ev2har
        # F *= foc2au

        Fxy = np.zeros([F.shape[0], 2], dtype=F.dtype)
        F = np.concatenate([Fxy, np.array([F]).T], axis=1)
        F = F.flatten()

        have_data = True
    elif msg == "STATUS      ":
        if have_data == False:
            fsoc.send("READY       ")
        else:
            fsoc.send("HAVEDATA    ")
    elif msg == "GETFORCE    ":
        fsoc.send("FORCEREADY  ")
        fsoc.sendall(V)
        fsoc.sendall(np.int32(nat))
        fsoc.sendall(F)
        fsoc.sendall(np.zeros([3,3]))
        fsoc.sendall(np.int32(7))
        fsoc.send("nothing")
        have_data = False
    else:
        run_flag = False

