#!/usr/bin/env python

import sys
from potential import Morse1D
from socket1D import connect_and_run

if len(sys.argv) > 2:               # if 2 arguments, open inet port
    try:
        port = int(sys.argv[1])     # both ip and port MUST be specified in ffsocket in ipi input
        assert port >= 4000
    except ValueError, AssertionError:
        raise ValueError('Port must be integer and >= 4000')
    addr = sys.argv[2]
    if addr != 'localhost' and not all(i.isdigit() for i in addr.split('.')):
        raise ValueError('Must give valid IP address or \'localhost\'')
elif len(sys.argv) > 1:             # if 1 argument, open unix port
    ipi_port = sys.argv[1]          # this MUST be specified as <address> in ffsocket in ipi input
    port = "/tmp/ipi_"+ipi_port
    addr = 'UNIX'
else:
    raise ValueError('Need at least one argument (address for UNIX port) or two arguments (port and IP for INET)')

connect_and_run(addr, port, Morse1D)
