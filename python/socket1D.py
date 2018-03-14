#!/usr/bin/env python

import socket
import struct
import sys
import numpy as np
from potential import PotentialEnergySurface

# ipi uses atomic units
L = 6.022E23
bohr2angs = 0.52918
bohr2nm = bohr2angs / 10
ev2har = 1.0/27.2114
foc2au = ev2har*bohr2angs
kJ2eV = 6.241509125E21

def connect_and_run(address, port, PES):
    # initialise socket
    have_data = False
    run_flag = True
    HDRLEN = 12
    if address == 'UNIX':
        fsoc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        fsoc.connect(port)
    else:
        fsoc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        fsoc.connect((address, port))

    first = True
    while run_flag == True:
        msg = fsoc.recv(HDRLEN)
        if msg == "POSDATA     ":
            cell_h = fsoc.recv(9*8)
            cell_h = [struct.unpack("d", cell_h[i*8:(i+1)*8])[0] for i in range(9)]

            if first:
                pes = PES(0.5, cell_h[8])
                first = False
            else:
                pes.update_cell(cell_h[8])

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

