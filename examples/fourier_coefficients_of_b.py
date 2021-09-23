R"""
This script facilitates plotting the Fourier coefficients of |B|.
"""
import numpy as np
import matplotlib.pyplot as plt
import booz_xform as bx


b = bx.Booz_xform()
b.verbose = 0
b.read_wout('../tests/test_files/wout_LandremanSenguptaPlunk_section5p3.nc')

b.run()

B = np.c_[b.extrapolate_on_axis_bmnc_b(), b.bmnc_b]

A0n = B[b.xm_b == 0, :]

B0n = B[b.xm_b != 0, :]

fig, ax = plt.subplots()

for y in A0n:
    ax.plot(np.append([0],b.s_b), y, 'r')

for y in B0n:
    ax.plot(np.append([0],b.s_b), y, 'k')

#  S = np.c_[b.extrapolate_on_axis_bmns_b(), b.bmns_b]

#  A0n = S[b.xm_b == 0, :]

#  xn = b.xn_b[b.xm_b == 0]

#  fig, ax = plt.subplots()

#  for y in A0n:
    #  ax.plot(np.append([0],b.s_b), y)

plt.show()

