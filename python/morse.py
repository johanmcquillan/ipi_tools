#!/usr/bin/env python

import socket
import struct
import sys
import numpy as np
from potential import Morse1D

# ipi uses atomic units
L = 6.022E23
bohr2angs = 0.52918
bohr2nm = bohr2angs / 10
ev2har = 1.0/27.2114
foc2au = ev2har*bohr2angs
kJ2eV = 6.241509125E21

# initialise socket
have_data = False
run_flag = True
HDRLEN = 12
if len(sys.argv) > 2:               # if 2 arguments, open inet port
    try:
        port = int(sys.argv[1])     # both ip and port MUST be specified in ffsocket in ipi input
        assert port >= 4000
    except ValueError, AssertionError:
        raise ValueError('Port must be integer and >= 4000')
    addr = sys.argv[2]
    if addr != 'localhost' and not all(i.isdigit() for i in addr.split('.')):
        raise ValueError('Must give valid IP address of \'localhost\'')
    fsoc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fsoc.connect((addr, port))
elif len(sys.argv) > 1:             # if 1 argument, open unix port
    ipi_addr = sys.argv[1]          # this MUST be specified as <address> in ffsocket in ipi input
    addr = "/tmp/ipi_"+ipi_addr
    fsoc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    fsoc.connect(addr)
else:
    raise ValueError('Need at least one argument (address for UNIX port) or two arguments (port and IP for INET port)')

first = True
while run_flag == True:
    msg = fsoc.recv(HDRLEN)
    if msg == "POSDATA     ":
        cell_h = fsoc.recv(9*8)
        cellh = [struct.unpack("d", cell_h[i*8:(i+1)*8])[0] for i in range(9)]

        if first:
            pes = 1DMorse(0.5, cellh[8])
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

