#!/usr/bin/env python

import socket
import struct
import sys
import numpy as np
from mpi4py import MPI

# Initialise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# assert size > 1
tag_Z = 16
tag_V = 32
tag_F = 48
tag_end = 60

# i-pi use annoying atomic units, make sure to be consistent
L = 6.022E23
bohr2angs = 0.52918
bohr2nm = bohr2angs / 10
ev2har = 1.0/27.2114
foc2au = ev2har*bohr2angs
kJ2eV = 6.241509125E21

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

if rank == 0:
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

# Loop to receive data from i-PI
pes = PES(0.5, 3.0/bohr2nm)
pos = np.empty((300, 3))
local_z = None
local_V = None
local_F = None
V = np.zeros(1)
F_O = np.empty(1)
z = None
array_lengths = None

run_flag = True
first = True
while rank != 0 or run_flag == True:
    if rank == 0:
        msg = fsoc.recv(HDRLEN)
        if msg == "POSDATA     ":
            # Get cell matrix
            cell_h = fsoc.recv(9*8)
            # cellh = [struct.unpack("d", cell_h[i*8:(i+1)*8])[0] for i in range(9)]

            # Update position of walls if z-height is free to change
            # if first:
            #     pes = PES(0.5, cellh[8])
            #     first = False
            # else:
            #     pes.update_cell(cellh[8])

            # Get cell matrix inverse
            cell_ih = fsoc.recv(9*8)
            
            # Get number of atoms
            nat = fsoc.recv(4)
            nat = struct.unpack("i", nat)[0]

            # Get positions
            read_pos = fsoc.recv(nat*3*8)
            pos = np.zeros([nat,3])
            for atom in range(nat):
                for coord in range(3):
                    pos[atom,coord] = struct.unpack("d", read_pos[atom*24+coord*8:atom*24+(coord+1)*8])[0]
            z = np.ascontiguousarray(pos[::3, 2])
            split = np.array_split(z, size)
            array_lengths = [len(split[i]) for i in range(len(split))]
            F_O = np.empty(z.shape)
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
    if rank != 0 or have_data:
        array_lengths = comm.bcast(array_lengths, root=0)
        local_z = np.empty(array_lengths[rank])
        comm.Scatterv(z, local_z, root=0)
        local_z = pes.pbc(local_z)
        local_V = pes.potential(local_z)
        local_F = -pes.gradient(local_z)
        # print rank, local_F.shape
        comm.Reduce(local_V, V, root=0)
        comm.Gatherv(local_F, F_O, root=0)

        if rank == 0:
            # print array_lengths
            F = np.zeros(nat*3)
            F[2::9] = F_O

            # Fxy = np.zeros((F_oxygen.shape[0]*3, 2), dtype=F_oxygen.dtype)
            # Fz = np.zeros(F_oxygen.shape[0]*3)
            # F = np.concatenate([Fxy, np.array([Fz]).T], axis=1)

