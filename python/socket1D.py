#!/usr/bin/env python

"""Applies a one-dimensional confining potential to the oxygen atoms in an i-PI water simulation.

Connects to a UNIX or INET socket opened by i-PI and computes the force and potential of two
confining walls parallel to the z-axis.
"""

import inspect
import numpy as np
import socket
import struct
from argparse import ArgumentParser, RawTextHelpFormatter
from packages.potentials import help_text, get_potential, angs2bohr

__author__ = "Johan G. McQuillan, Wei Fang"
__email__ = "johan.mcquillan.13@ucl.ac.uk"

# Get arguments
parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument('V', type=str, help=help_text())
parser.add_argument('w', type=float, help='Confinement width in Angstroms')
parser.add_argument('port', type=str, metavar='P', help='Port number (if INET) or address name (if UNIX)')
parser.add_argument('ip', type=str, nargs='?', default=None, help='IP address - if given, opens an INET socket, else a UNIX socket')
args = parser.parse_args()

# Get external potential class
PES = get_potential(args.V)

# Check if INET or UNIX socket
# If INET, ensure IP and port are valid
if args.ip is not None:
    try:
        port = int(args.port)
        assert port >= 4000
    except ValueError, AssertionError:
        raise ValueError('If using INET, port must be integer and >= 4000')
    address = args.ip
    if address != 'localhost' and not all(i.isdigit() for i in address.split('.')):
        raise ValueError('Must give valid IP address or \'localhost\'')
else:
    port = "/tmp/ipi_"+args.port
    address = 'UNIX'

# Initialise socket
have_data = False
run_flag = True
HDRLEN = 12
if address == 'UNIX':
    fsoc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    fsoc.connect(port)
else:
    fsoc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fsoc.connect((address, port))

# Start force calculation loop
first = True
while run_flag == True:
    msg = fsoc.recv(HDRLEN)
    if msg == "POSDATA     ":       # i-PI has sent the data to the socket
        
        # Unpack cell matrix
        cell_h = fsoc.recv(9*8)
        cell_h = [struct.unpack("d", cell_h[i*8:(i+1)*8])[0] for i in range(9)]
        
        # Instantiate or update the external potential
        if first:
            pes = PES(args.w*angs2bohr, cell_h[8])
            first = False
        else:
            pes.update_cell(cell_h[8])
        # Unpack inverse of cell matrix
        cell_ih = fsoc.recv(9*8)
        
        # Unpack total number of atoms
        nat = fsoc.recv(4)
        nat = struct.unpack("i", nat)[0]
        
        # Unpack position data
        read_pos = fsoc.recv(nat*3*8)
        pos = np.zeros([nat,3])
        for atom in range(nat):
            for coord in range(3):
                pos[atom,coord] = struct.unpack("d", read_pos[atom*24+coord*8:atom*24+(coord+1)*8])[0]
        
        # Get only the z-coordinate of the oxygens
        z_array = pes.pbc(pos[::3, 2])
        
        # Calculate total potential and force on each oxygen
        V = np.sum(pes.potential(z_array))
        F_oxygens = pes.force(z_array, checked_confined=True)
        
        # Reformat force array so it has zeros for H and zeros for x & y for O
        F = np.zeros(nat*3)
        F[2::9] = F_oxygens
        
        have_data = True
    elif msg == "STATUS      ":
        if have_data == False:
            fsoc.send("READY       ")
        else:
            fsoc.send("HAVEDATA    ")
    elif msg == "GETFORCE    ":
        # Send data to i-pi
        fsoc.send("FORCEREADY  ")
        fsoc.sendall(V)
        fsoc.sendall(np.int32(nat))
        fsoc.sendall(F)
        fsoc.sendall(np.zeros([3,3]))
        fsoc.sendall(np.int32(7))       # I think this is the virial...
        fsoc.send("nothing")
        have_data = False
    else:
        run_flag = False

