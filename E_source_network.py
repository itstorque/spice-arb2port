import numpy as np

COPY_OUTPUT = False

if COPY_OUTPUT: 
    import pyperclip

# ### taper setup

# import skrf as rf
# from skrf.media import MLine

# freq = rf.Frequency(1, 100000, unit='MHz', npoints=10000)
# w1 = 20*rf.mil  # conductor width [m]
# w2 = 50*rf.mil  # conductor width [m]
# h = 50*rf.mil  # dielectric thickness [m]
# t = 0.7*rf.mil  # conductor thickness [m]
# rho = 1.724138e-8  # Copper resistivity [Ohm.m]
# ep_r = 10  # dielectric relative permittivity
# rough = 1e-6  # conductor RMS roughtness [m]
# taper_length = 10*rf.mil  # [m]

# taper_exp = rf.taper.Exponential(med=MLine, param='w', start=w1, stop=w2,
#                         length=taper_length, n_sections=50,
#                         med_kw={'frequency': freq, 'h': h, 't':t, 'ep_r': ep_r,
#                                 'rough': rough, 'rho': rho}).network

# S = taper_exp.s

# S11, S12, S21, S22 = S[:,0,0], S[:,0,1], S[:,1,0], S[:,1,1]

# Zin, Zout = abs(taper_exp.z0[0, 0]), abs(taper_exp.z0[0, 1])

# print("IMPEDANCES: ", Zin, Zout)

# freq = taper_exp.frequency.f

# ###

### LC filter

import skrf as rf
from skrf.media import MLine

freq = rf.Frequency(start=0.1, stop=10, unit='GHz', npoints=10000)
tl_media = rf.DefinedGammaZ0(freq, z0=50, gamma=1j*freq.w/rf.c)
C1 = tl_media.capacitor(3.222e-12, name='C1')
C2 = tl_media.capacitor(82.25e-15, name='C2')
C3 = tl_media.capacitor(3.222e-12, name='C3')
L2 = tl_media.inductor(8.893e-9, name='L2')
RL = tl_media.resistor(50, name='RL')
gnd = rf.Circuit.Ground(freq, name='gnd')
port1 = rf.Circuit.Port(freq, name='port1', z0=50)
port2 = rf.Circuit.Port(freq, name='port2', z0=50)

cnx = [
    [(port1, 0), (C1, 0), (L2, 0), (C2, 0)],
    [(L2, 1), (C2, 1), (C3, 0), (port2, 0)],
    [(gnd, 0), (C1, 1), (C3, 1)],
]
cir = rf.Circuit(cnx)
ntw = cir.network

# import matplotlib.pyplot as plt
# cir.plot_graph(network_labels=True, network_fontsize=15,
#                port_labels=True, port_fontsize=15,
#               edge_labels=True, edge_fontsize=10)
# plt.show()

Zin, Zout = 50, 50
freq = ntw.frequency.f

S = ntw.s
S11, S12, S21, S22 = S[:,0,0], S[:,0,1], S[:,1,0], S[:,1,1]

###

# ### Manual Test

# Zin, Zout = 50, 50

# freq = [1e6, 2e6]

# S11 = [1+2j, 1-2j]
# S12 = [1-2j, 1+2j]
# S21 = [-1-0.5j, 1]
# S22 = [2, 0]

# ###

def S_param_source(freq, Sxy):
    
    output = ""
    
    for idx in range(len(freq)):
        
        mag, phase = abs(Sxy[idx]), np.angle(Sxy[idx])
        
        output += f"+ ({str(freq[idx])}, {20*np.log10(mag)}, {phase})\n"
        
        # output += f"+ ({str(freq[idx])}, {np.real(Sxy[idx])}, {np.imag(Sxy[idx])})\n"
        
    return output

# TYPE = "R_I" # "DB, MAG, RAD, DEG..."
TYPE = "DB"

OUTPUT = f""".SUBCKT 2_PORT_TEST 1 2
R1N 1 10 {-Zin}
R1P 10 11 {2*Zout}
R2N 2 20 {-Zout}
R2P 20 21 {2*Zout}

*S11 FREQ DB PHASE
E11 11 12 FREQ {{V(10, 0)}}= {TYPE}
{S_param_source(freq, S11)}

*S12 FREQ DB PHASE
E12 12 G FREQ {{V(20, 0)}}= {TYPE}
{S_param_source(freq, S12)}

*S21 FREQ DB PHASE
E21 21 22 FREQ {{V(10, 0)}}= {TYPE}
{S_param_source(freq, S21)}

*S22 FREQ DB PHASE
E22 22 G FREQ {{V(20, 0)}}= {TYPE}
{S_param_source(freq, S22)}

.ENDS
"""

if COPY_OUTPUT: 
    print(OUTPUT, "COPIED")
    pyperclip.copy(OUTPUT)
else:
    f = open("2_PORT_TEST.lib", "w")
    f.write(OUTPUT)
    f.close()

# ### Debug Plotting

# import matplotlib.pyplot as plt

# plt.plot(freq/1e9, 20*np.log10(abs(S22)))

# # plt.plot(freq/1e9, np.angle(S22))

# plt.show()

# plt.plot(freq/1e9, np.angle(S22))
# plt.show()

# ###