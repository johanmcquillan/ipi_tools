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

    def __init__(self, w, c):
        self.zeta = 3.85 / bohr2angs
        self.a = 0.92 * bohr2angs
        self.D = 57.8E-3 * ev2har
        self.w = w / bohr2nm
        self.update_cell(c)

    def update_cell(self, c):
        # print self.w * bohr2angs, c * bohr2angs, (c - self.w)/2 * bohr2angs, (c + self.w)/2 * bohr2angs
        assert self.w < c
        self.c = c # / bohr2nm
        self.z0 = (c - self.w)/2
        self.z1 = (c + self.w)/2

    def pbc(self, z):
        while np.any(z < 0):
            z += self.c * (z < 0).astype(int)
        while np.any(z > self.c):
            z -= self.c * (z > self.c).astype(int)
        return z

    def potential(self, z):
        assert np.all(z > self.z0)
        assert np.all(z < self.z1)

        V = np.zeros(z.shape)
        V = self.D * ((1.0 - np.exp(-self.a*(z - self.z0 - self.zeta)))**2 +
                    (1.0 - np.exp(-self.a*(self.z1 - z - self.zeta)))**2 - 2.0)
        return np.sum(V)

    def gradient(self, z):
        F = np.zeros(z.shape)
        F = 2 * self.a * self.D * (np.exp(-self.a*(z - self.z0 - self.zeta)) - np.exp(-2*self.a*(z - self.z0 - self.zeta)) - 
                                    np.exp(-self.a*(self.z1 - z - self.zeta)) + np.exp(-2*self.a*(self.z1 - z - self.zeta)))
        return F

# force field Loop
first = True
while run_flag == True:
    msg = fsoc.recv(HDRLEN)
    if msg == "POSDATA     ":
        cell_h = fsoc.recv(9*8)
        cellh = [struct.unpack("d", cell_h[i*8:(i+1)*8])[0] for i in range(9)]

        if first:
            pes = PES(0.5, cellh[8])
        else:
            pes.update_cell(cellh[8])

        cell_ih = fsoc.recv(9*8)
        
        nat = fsoc.recv(4)
        nat = struct.unpack("i", nat)[0]

        read_pos = fsoc.recv(nat*3*8)
        pos = np.zeros([nat,3])
        for atom in range(nat):
            for coord in range(3):
                pos[atom,coord] = struct.unpack("d", read_pos[atom*24+coord*8:atom*24+(coord+1)*8])[0]
        z_array = pes.pbc(pos[::3, 2])
        
        V = np.sum(pes.potential(z_array))
        F_oxygens = -pes.gradient(z_array)

        F = np.zeros(nat*3)
        F[2::9] = F_oxygens

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
